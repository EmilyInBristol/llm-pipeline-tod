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

def write_jsonl(filename, prompt_list, gt_key):
    with open(filename, 'w', encoding='utf-8') as f:
        for item in prompt_list:
            prompt = item['prompt']
            gt = item['gold_result'][gt_key]
            if isinstance(gt, dict):
                gt = json.dumps(gt, ensure_ascii=False)
            f.write(json.dumps({"prompt": prompt, "output": gt}, ensure_ascii=False) + "\n")

def write_txt(filename, prompt_list, gt_key):
    with open(filename, 'w', encoding='utf-8') as f:
        for item in prompt_list:
            prompt = item['prompt']
            gt = item['gold_result'][gt_key]
            if isinstance(gt, dict):
                gt = json.dumps(gt, ensure_ascii=False)
            f.write(f"Prompt:\n{prompt}\nOutput:\n{gt}\n\n")

def main():
    data = load_multiwoz()
    vec_file_path = "multiwoz-context-db.vec"
    top_k = 5
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    with open(vec_file_path, "rb") as f:
        vector_store = pickle.load(f)

    state_prompts = []
    response_prompts = []
    domain_prompts = []
    
    # 新增：用于BERT分类训练的pair收集
    bert_domain_eval_pairs = []

    last_dial_id = None
    history = []
    domain_counter = {}

    for it, turn in enumerate(tqdm(iterate_dialogues(data, default_database), desc="Processing", unit="turns")):
        dialog_id = turn['dialogue_id']
        if dialog_id != last_dial_id:
            history = []
            last_dial_id = dialog_id

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
        # 统计 domain 数量
        domain_counter[domain] = domain_counter.get(domain, 0) + 1

        # 1. 状态抽取 prompt
        examples_str = process_examples(
            examples, ['context'], ['state']
        )
        final_prompt = get_state_extraction_prompt(
            domain=domain,
            examples=examples_str,
            input=history_text,
            customer=turn['question'].strip()
        )
        state_prompts.append({
            "id": len(state_prompts) + 1,
            "prompt": final_prompt,
            "gold_result": {
                "gt_state": turn['metadata']['state']
            }
        })

        # 2. 回答生成 prompt
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
        response_prompts.append({
            "id": len(response_prompts) + 1,
            "prompt": final_prompt,
            "gold_result": {
                "response": turn['metadata']['response'],
                "gt_full_state": turn['gt_state'],
                "database": turn['metadata']['database']
            }
        })

        # 3. 域识别 prompt
        history_text_domain = "\n".join(history[-2:])
        final_prompt = DOMAIN_RECOGNITION_PROMPT.format(
            history_text_domain,
            turn['question'].strip()
        )
        domain_prompts.append({
            "id": len(domain_prompts) + 1,
            "prompt": final_prompt,
            "gold_result": {
                # "domain": turn['metadata']['current_domain'],
                "domain": turn['metadata']['domain'],
                "gt_full_state": turn['gt_state']
            }
        })

        # 新增：收集BERT分类训练pair
        bert_source = history_text_domain + "\n" + f"Customer: {turn['question'].strip()}"
        bert_domain_eval_pairs.append({
            "source": bert_source,
            # "target": turn['metadata']['current_domain']
            "target": turn['metadata']['domain']
        })

        history.append(f"Customer: {turn['question']}")
        history.append(f"Assistant: {turn['metadata']['response']}")

    write_jsonl('state_extraction_train.jsonl', state_prompts, 'gt_state')
    write_jsonl('response_generation_train.jsonl', response_prompts, 'response')
    write_jsonl('domain_recognition_train.jsonl', domain_prompts, 'domain')

    write_txt('state_extraction_train.txt', state_prompts, 'gt_state')
    write_txt('response_generation_train.txt', response_prompts, 'response')
    write_txt('domain_recognition_train.txt', domain_prompts, 'domain')

    print("已输出 state_extraction_train.jsonl, response_generation_train.jsonl, domain_recognition_train.jsonl")
    print("各 domain 样本数量：")
    for domain, count in domain_counter.items():
        print(f"{domain}: {count}")

    # 输出BERT分类训练数据
    with open('bert_domain_train_pairs.jsonl', 'w', encoding='utf-8') as f:
        for item in bert_domain_eval_pairs:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print("已输出到 bert_domain_train_pairs.jsonl")

if __name__ == "__main__":
    main() 
    # data = load_multiwoz()
    # print(data)