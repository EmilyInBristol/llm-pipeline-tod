import pickle
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
import json

from multiwoz_utils.dialog_iterator import iterate_dialogues
from multiwoz_utils.data_loader import load_multiwoz
from prompts import (
    get_state_extraction_prompt,
    get_response_generation_prompt,
    DOMAIN_RECOGNITION_PROMPT
)
from multiwoz_utils.database import default_database

def process_examples(examples, input_keys, output_keys):
    output = "\n"
    for n, ex in enumerate(examples[-2:]):
        input_str = '\n'.join((f"{key if key != 'full_state' else 'state'}: {ex[key]}" for key in input_keys))
        output_str = '\n'.join((f"{key}: {ex[key]}" for key in output_keys))
        output += "---------------------" + \
                  f"Example {n}:\n" + \
                  f"{input_str}\n" + \
                  f"\n{output_str}\n"
    return output + "\n"

def main():
    # Load validation split
    data = load_multiwoz(split='validation')

    # Load vector store (if needed)
    vec_file_path = "multiwoz-context-db.vec"
    top_k = 5
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    with open(vec_file_path, "rb") as f:
        vector_store = pickle.load(f)

    state_prompts = []
    response_prompts = []
    domain_prompts = []
    
    # New: Collect pairs for BERT classification evaluation
    bert_domain_eval_pairs = []

    last_dial_id = None
    history = []

    for it, turn in enumerate(tqdm(iterate_dialogues(data, default_database), desc="Processing", unit="turns")):
        dialog_id = turn['dialogue_id']
        if dialog_id != last_dial_id:
            history = []
            last_dial_id = dialog_id

        # Current history, excluding current utterance
        history_text = "\n".join(history)
        query_text = turn['page_content']
        results = vector_store.similarity_search(query_text, k=top_k)
        examples = [{
            'context': doc.metadata.get('context', ''),
            'state': doc.metadata.get('state', ''),
            'full_state': doc.metadata.get('full_state', ''),
            'response': doc.metadata.get('response', ''),
            'database': doc.metadata.get('database', ''),
            'domain': doc.metadata.get('domain', '')
        } for doc in results]

        domain = turn['metadata']['domain']

        # 1. State extraction prompt
        examples_str = process_examples(
            examples, ['context'], ['state']
        )
        final_prompt = get_state_extraction_prompt(
            domain=domain,
            examples=examples_str,
            input=history_text,
            customer=turn['question'].strip()
        )
        # Only add when gt_state is not empty
        if turn['gt_state']:
            state_prompts.append({
                "id": turn['dialogue_id'],
                "prompt": final_prompt,
                "gold_result": {
                    "gt_state": turn['metadata']['state'],
                    "gt_full_state": turn['metadata']['full_state']
                }
            })

        # 2. Response generation prompt
        examples_str = process_examples(
            examples, ["context", "full_state", "database"], ["response"]
        )
        history_text_response = "\n".join(history[-6:])
        final_prompt = get_response_generation_prompt(
            domain=domain,
            examples=examples_str,
            input=history_text_response,
            customer=turn['question'].strip(),
            state=turn['gt_state'],
            database=turn['metadata']['database']
        )
        # Only add when gt_state is not empty
        if turn['gt_state']:
            response_prompts.append({
                "id": turn['dialogue_id'],
                "prompt": final_prompt,
                "gold_result": {
                    "response": turn['metadata']['response'],
                    "gt_full_state": turn['gt_state'],
                    "database": turn['metadata']['database']
                }
            })

        # 3. Domain recognition prompt
        history_text_domain = "\n".join(history[-2:])
        final_prompt = DOMAIN_RECOGNITION_PROMPT.format(
            history_text_domain,
            turn['question'].strip()
        )
        domain_prompts.append({
            "id": turn['dialogue_id'],
            "prompt": final_prompt,
            "gold_result": {
                "domain": turn['metadata']['domain'],
                "gt_full_state": turn['gt_state']
            }
        })
        # New: Collect BERT classification evaluation pairs
        # source is history_text_domain and current Customer input
        bert_source = history_text_domain + "\n" + f"Customer: {turn['question'].strip()}"
        bert_domain_eval_pairs.append({
            "source": bert_source,
            "target": turn['metadata']['domain']
        })

        # Update dialogue history
        history.append(f"Customer: {turn['question']}")
        history.append(f"Assistant: {turn['metadata']['response']}")

    # Output to json files
    with open('state_extraction_prompts.json', 'w', encoding='utf-8') as f:
        json.dump(state_prompts, f, ensure_ascii=False, indent=2)

    with open('response_generation_prompts.json', 'w', encoding='utf-8') as f:
        json.dump(response_prompts, f, ensure_ascii=False, indent=2)

    with open('domain_recognition_prompts.json', 'w', encoding='utf-8') as f:
        json.dump(domain_prompts, f, ensure_ascii=False, indent=2)

    # Output to txt files for easy reading
    def write_txt(filename, prompts, gold_keys):
        with open(filename, 'w', encoding='utf-8') as f:
            for item in prompts:
                f.write(f"ID: {item['id']}\n")
                f.write("Prompt:\n" + item['prompt'] + "\n")
                f.write("Gold Result:\n")
                for k in gold_keys:
                    v = item['gold_result'].get(k, '')
                    f.write(f"  {k}: {v}\n")
                f.write("\n" + "="*40 + "\n\n")

    write_txt('state_extraction_prompts.txt', state_prompts, ['gt_state', 'gt_full_state'])
    write_txt('response_generation_prompts.txt', response_prompts, ['response', 'gt_full_state', 'database'])
    write_txt('domain_recognition_prompts.txt', domain_prompts, ['domain', 'gt_full_state'])

    print("Output to state_extraction_prompts.json, response_generation_prompts.json, domain_recognition_prompts.json")
    print("Output to state_extraction_prompts.txt, response_generation_prompts.txt, domain_recognition_prompts.txt")

    # New: Output BERT classification evaluation data
    with open('bert_domain_eval_pairs.jsonl', 'w', encoding='utf-8') as f:
        for item in bert_domain_eval_pairs:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print("Output to bert_domain_eval_pairs.jsonl")

if __name__ == "__main__":
    main() 