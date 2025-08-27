import json
import re
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import random
import evaluate
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_model(base_model_path, lora_model_path, use_lora=False, load_in_8bit=False, load_in_4bit=False):
    """Load Tokenizer and LoRA model"""
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto",
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
    )
    if use_lora:
        model = PeftModel.from_pretrained(model, lora_model_path)
    return tokenizer, model

def infer_domains(tokenizer, model, prompts, max_new_tokens=128, num_samples=None):
    results = []
    if num_samples is not None and num_samples < len(prompts):
        sampled_indices = random.sample(range(len(prompts)), num_samples)
        sampled_prompts = [prompts[i] for i in sampled_indices]
    else:
        sampled_indices = list(range(len(prompts)))
        sampled_prompts = prompts
    for idx, prompt in tqdm(list(enumerate(sampled_prompts)), desc="Inferencing Domain"):
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        outputs = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
        output_ids = outputs[0][len(model_inputs.input_ids[0]):].tolist()
        generated_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        pred_domain = generated_text.lower().strip()
        results.append({
            "prompt": prompt,
            "pred_domain": pred_domain,
            "raw_output": generated_text,
            "original_index": sampled_indices[idx]
        })
    return results

def parse_state_output(state_str):
    try:
        match = re.search(r'\{.*\}', state_str, re.DOTALL)
        if match:
            state_str = match.group(0)
        state = json.loads(state_str.replace("'", '"'))
        if not isinstance(state, dict):
            return {}
        return state
    except Exception:
        return {}

def infer_states(tokenizer, model, prompts, max_new_tokens=128, num_samples=None):
    results = []
    if num_samples is not None and num_samples < len(prompts):
        sampled_indices = random.sample(range(len(prompts)), num_samples)
        sampled_prompts = [prompts[i] for i in sampled_indices]
    else:
        sampled_indices = list(range(len(prompts)))
        sampled_prompts = prompts
    for idx, prompt in tqdm(list(enumerate(sampled_prompts)), desc="Inferencing State"):
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        outputs = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
        output_ids = outputs[0][len(model_inputs.input_ids[0]):].tolist()
        generated_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        pred_state = parse_state_output(generated_text)
        results.append({
            "prompt": prompt,
            "pred_state": pred_state,
            "raw_output": generated_text,
            "original_index": sampled_indices[idx]
        })
    return results

def infer_responses(tokenizer, model, prompts, max_new_tokens=128, num_samples=None):
    """Inference response generation"""
    results = []
    if num_samples is not None and num_samples < len(prompts):
        sampled_indices = random.sample(range(len(prompts)), num_samples)
        sampled_prompts = [prompts[i] for i in sampled_indices]
    else:
        sampled_indices = list(range(len(prompts)))
        sampled_prompts = prompts
    
    for idx, prompt in tqdm(list(enumerate(sampled_prompts)), desc="Inferencing Response"):
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        outputs = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
        output_ids = outputs[0][len(model_inputs.input_ids[0]):].tolist()
        generated_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        
        results.append({
            "prompt": prompt,
            "pred_response": generated_text.strip(),
            "raw_output": generated_text,
            "original_index": sampled_indices[idx]
        })
    return results

def pipeline_infer(
    base_model_path, 
    domain_lora_path, state_lora_path, response_lora_path,
    vec_file_path="multiwoz-context-db.vec", top_k=2,
    max_new_tokens=128, num_samples=None,
    use_lora=True, load_in_8bit=False, load_in_4bit=False
):
    """
    Complete three-step pipeline inference: Domain recognition -> State prediction -> Response generation
    Reuse the same base model, load different LoRA weights, dynamically generate prompts
    
    Args:
        base_model_path: Base model path
        domain_lora_path: Domain task LoRA weights path
        state_lora_path: State task LoRA weights path  
        response_lora_path: Response task LoRA weights path
        vec_file_path: Vector database file path
        top_k: Number of retrieved examples
        max_new_tokens: Maximum tokens to generate
        num_samples: Number of samples, None means all
        use_lora: Whether to use LoRA weights
        load_in_8bit: Whether to load in 8bit quantization
        load_in_4bit: Whether to load in 4bit quantization
    
    Returns:
        dict: Dictionary containing results of three tasks
    """
    print("=== Starting pipeline inference ===")
    
    # Import necessary modules
    import pickle
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from multiwoz_utils.dialog_iterator import iterate_dialogues
    from multiwoz_utils.data_loader import load_multiwoz
    from prompts import (
        get_state_extraction_prompt,
        get_response_generation_prompt,
        DOMAIN_RECOGNITION_PROMPT
    )
    from multiwoz_utils.database import default_database
    
    # Load data and vector store
    print("Loading validation data and vector store...")
    data = load_multiwoz(split='validation')
    
    with open(vec_file_path, "rb") as f:
        vector_store = pickle.load(f)
    
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
    
    # Collect data for all turns
    print("Collecting dialogue data...")
    all_turns = []
    last_dial_id = None
    history = []
    
    for turn in iterate_dialogues(data, default_database):
        dialog_id = turn['dialogue_id']
        if dialog_id != last_dial_id:
            history = []
            last_dial_id = dialog_id
        
        # Only process turns with gt_state
        if turn['gt_state']:
            all_turns.append({
                'turn': turn,
                'history': history.copy()
            })
        
        # Update dialogue history
        history.append(f"Customer: {turn['question']}")
        history.append(f"Assistant: {turn['metadata']['response']}")
    
    # Sampling
    if num_samples is not None and num_samples < len(all_turns):
        sampled_turns = random.sample(all_turns, num_samples)
    else:
        sampled_turns = all_turns
    print(f"Will process {len(sampled_turns)} dialogue turns")
    
    # 1. Domain recognition
    print("Step 1: Loading Domain model for domain recognition...")
    domain_tokenizer, domain_model = load_model(
        base_model_path, domain_lora_path, use_lora, load_in_8bit, load_in_4bit
    )
    
    domain_prompts = []
    domain_gold_data = []
    for item in sampled_turns:
        turn = item['turn']
        history = item['history']
        history_text_domain = "\n".join(history[-2:])
        domain_prompt = DOMAIN_RECOGNITION_PROMPT.format(
            history_text_domain,
            turn['question'].strip()
        )
        domain_prompts.append(domain_prompt)
        domain_gold_data.append({
            'id': turn['dialogue_id'],
            'gold_domain': turn['metadata']['domain'],
            'gt_full_state': turn['gt_state']
        })
    
    domain_results = infer_domains(
        domain_tokenizer, domain_model, domain_prompts, max_new_tokens, len(sampled_turns)
    )
    pred_domains = [r["pred_domain"] for r in domain_results]
    print(f"Domain recognition completed, prediction result examples: {pred_domains[:3]}")
    
    # Release Domain model memory
    del domain_model, domain_tokenizer
    
    # 2. State prediction
    print("Step 2: Loading State model for state prediction...")
    state_tokenizer, state_model = load_model(
        base_model_path, state_lora_path, use_lora, load_in_8bit, load_in_4bit
    )
    
    state_prompts = []
    state_gold_data = []
    for i, item in enumerate(sampled_turns):
        turn = item['turn']
        history = item['history']
        pred_domain = pred_domains[i]  # Use predicted domain
        
        history_text = "\n".join(history)
        
        # Vector retrieval examples (if vector store is available)
        examples = []
        if vector_store is not None:
            try:
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
                # print(f"Successfully retrieved {len(examples)} examples")
            except Exception as e:
                print(f"Vector retrieval failed: {e}")
                print("Will continue with empty examples...")
                examples = []
                # Disable vector store to avoid subsequent failures
                vector_store = None
        
        # If no examples, create empty examples
        if not examples:
            examples = [{
                'context': '',
                'state': '{}',
                'full_state': '{}',
                'response': '',
                'database': '{}',
                'domain': pred_domain
            }]
        
        examples_str = process_examples(examples, ['context'], ['state'])
        state_prompt = get_state_extraction_prompt(
            domain=pred_domain,  # Use predicted domain
            examples=examples_str,
            input=history_text,
            customer=turn['question'].strip()
        )
        state_prompts.append(state_prompt)
        state_gold_data.append({
            'id': turn['dialogue_id'],
            'gold_state': turn['metadata']['state'],
            'gold_full_state': turn['metadata']['full_state']
        })
    
    state_results = infer_states(
        state_tokenizer, state_model, state_prompts, max_new_tokens, len(sampled_turns)
    )
    pred_states = [r["pred_state"] for r in state_results]
    print(f"State prediction completed, prediction result examples: {pred_states[:2]}")
    
    # Release State model memory
    del state_model, state_tokenizer
    
    # 3. Response generation
    print("Step 3: Loading Response model for response generation...")
    response_tokenizer, response_model = load_model(
        base_model_path, response_lora_path, use_lora, load_in_8bit, load_in_4bit
    )
    
    response_prompts = []
    response_gold_data = []
    for i, item in enumerate(sampled_turns):
        turn = item['turn']
        history = item['history']
        pred_domain = pred_domains[i]  # Use predicted domain
        pred_state = pred_states[i]    # Use predicted state
        
        history_text_response = "\n".join(history[-6:])
        
        # Vector retrieval examples (if vector store is available)
        examples = []
        if vector_store is not None:
            try:
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
            except Exception as e:
                print(f"Response vector retrieval failed: {e}")
                print("Will continue with empty examples...")
                examples = []
                vector_store = None
        
        # If no examples, create empty examples
        if not examples:
            examples = [{
                'context': '',
                'state': json.dumps(pred_state, ensure_ascii=False),
                'full_state': json.dumps(pred_state, ensure_ascii=False),
                'response': '',
                'database': json.dumps(turn['metadata']['database'], ensure_ascii=False),
                'domain': pred_domain
            }]
        
        examples_str = process_examples(examples, ["context", "full_state", "database"], ["response"])
        response_prompt = get_response_generation_prompt(
            domain=pred_domain,  # Use predicted domain
            examples=examples_str,
            input=history_text_response,
            customer=turn['question'].strip(),
            state=pred_state,  # Use predicted state
            database=turn['metadata']['database']
        )
        response_prompts.append(response_prompt)
        response_gold_data.append({
            'id': turn['dialogue_id'],
            'gold_response': turn['metadata']['response'],
            'gt_full_state': turn['gt_state'],
            'database': turn['metadata']['database']
        })
    
    response_results = infer_responses(
        response_tokenizer, response_model, response_prompts, max_new_tokens, len(sampled_turns)
    )
    print(f"Response generation completed, prediction result examples: {[r['pred_response'][:50] + '...' for r in response_results[:2]]}")
    
    # Release Response model memory
    del response_model, response_tokenizer
    
    print("=== Pipeline inference completed ===")
    
    # Add gold label information to results (for subsequent evaluation)
    for i in range(len(sampled_turns)):
        domain_results[i].update(domain_gold_data[i])
        state_results[i].update(state_gold_data[i])
        response_results[i].update(response_gold_data[i])
    
    return {
        "domain_results": domain_results,
        "state_results": state_results, 
        "response_results": response_results,
        "processed_turns": len(sampled_turns)
    }

def evaluate_domains(results, gold_domains, error_log_file="error_log.txt"):
    correct = 0
    empty_gold_count = 0
    error_log = []
    preds = []
    golds = []
    for i, r in enumerate(results):
        gold = gold_domains[i].lower().strip() if i < len(gold_domains) else ""
        pred = r.get("pred_domain", "").lower().strip()
        golds.append(gold)
        preds.append(pred)
        if not gold:
            empty_gold_count += 1
        if gold and pred and pred == gold:
            correct += 1
        else:
            error_log.append(
                "Prompt:\n" + r.get("prompt", "") + "\n"
                "Gold Domain: " + str(gold) + "\n"
                "Predicted Domain: " + str(pred) + "\n"
                "Raw Output:\n" + r.get("raw_output", "") + "\n"
                + "-" * 50 + "\n"
            )
    if error_log:
        with open(error_log_file, "w", encoding='utf-8') as f:
            f.writelines(error_log)
    accuracy = correct / len(results) if results else 0
    print(f"\nDomain Inference finished! Accuracy: {accuracy:.2%} ({correct}/{len(results)})")
    print(f"Number of empty gold_domain: {empty_gold_count}")
    if error_log:
        print(f"❌ Found {len(error_log)} error cases, written to {error_log_file}")
    domain_labels = sorted(list(set([g for g in golds if g])))
    print("\nDetailed evaluation report:")
    print(classification_report(golds, preds, labels=domain_labels, digits=3))
    print("Confusion Matrix:")
    print(confusion_matrix(golds, preds, labels=domain_labels))

def flatten_state(state):
    flat = set()
    if not isinstance(state, dict):
        return flat
    def norm(x):
        if isinstance(x, str):
            return x.lower().strip()
        return str(x).lower().strip()
    for k, v in state.items():
        if isinstance(v, dict):
            for slot, value in v.items():
                flat.add((norm(slot), norm(value)))
        else:
            flat.add((norm(k), norm(v)))
    return flat

def evaluate_states_v2(results, gold_states, gold_full_states):
    total = len(results)
    JGA = 0
    missing = 0
    badcase = 0
    acceptable_extra = 0
    logs = {
        "JGA": [],
        "missing": [],
        "badcase": [],
        "acceptable_extra": []
    }
    all_tp, all_fp, all_fn = 0, 0, 0
    for i, r in enumerate(results):
        gold = gold_states[i] if i < len(gold_states) else {}
        gold_full = gold_full_states[i] if i < len(gold_full_states) else {}
        pred = r.get("pred_state", {})
        gold_slots = flatten_state(gold)
        full_slots = flatten_state(gold_full)
        pred_slots = flatten_state(pred)
        tp = len(pred_slots & gold_slots)
        fp = len(pred_slots - gold_slots)
        fn = len(gold_slots - pred_slots)
        all_tp += tp
        all_fp += fp
        all_fn += fn
        if pred_slots == gold_slots:
            JGA += 1
            logs["JGA"].append(f"Prompt:\n{r.get('prompt','')}\nGold: {gold}\nPred: {pred}\nRaw: {r.get('raw_output','')}\n{'-'*50}\n")
            continue
        extra_slots = pred_slots - gold_slots
        missing_slots = gold_slots - pred_slots
        if any(slot not in full_slots for slot in extra_slots):
            badcase += 1
            logs["badcase"].append(f"Prompt:\n{r.get('prompt','')}\nGold: {gold}\nFull: {gold_full}\nPred: {pred}\nExtra predictions not in full_state: {extra_slots - full_slots}\nRaw: {r.get('raw_output','')}\n{'-'*50}\n")
        elif extra_slots and all(slot in full_slots for slot in extra_slots):
            acceptable_extra += 1
            logs["acceptable_extra"].append(f"Prompt:\n{r.get('prompt','')}\nGold: {gold}\nFull: {gold_full}\nPred: {pred}\nExtra predictions but all in full_state: {extra_slots}\nRaw: {r.get('raw_output','')}\n{'-'*50}\n")
        elif missing_slots:
            missing += 1
            logs["missing"].append(f"Prompt:\n{r.get('prompt','')}\nGold: {gold}\nFull: {gold_full}\nPred: {pred}\nMissing recall: {missing_slots}\nRaw: {r.get('raw_output','')}\n{'-'*50}\n")
        else:
            logs["missing"].append(f"Prompt:\n{r.get('prompt','')}\nGold: {gold}\nFull: {gold_full}\nPred: {pred}\nUnknown case\nRaw: {r.get('raw_output','')}\n{'-'*50}\n")
    print(f"\nState evaluation finished! (Ignore domain, only compare slot+value)")
    print(f"JGA (Joint Goal Accuracy): {JGA}/{total}")
    print(f"Missing: {missing}/{total}")
    print(f"Badcase (extra slots not in full_state): {badcase}/{total}")
    print(f"Acceptable extra: {acceptable_extra}/{total}")
    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    print(f"Slot-value level Micro Precision: {precision:.4f}")
    print(f"Slot-value level Micro Recall: {recall:.4f}")
    print(f"Slot-value level Micro F1: {f1:.4f}")
    for k, v in logs.items():
        if v:
            with open(f"{k}_log.txt", "w", encoding="utf-8") as f:
                f.writelines(v)
    print("Logs have been written to JGA_log.txt, missing_log.txt, badcase_log.txt, acceptable_extra_log.txt")

def evaluate_responses(results, gold_responses, error_log_file="error_log.txt"):
    predictions = []
    references = []
    log_lines = []
    for i, r in enumerate(results):
        gold = gold_responses[i].strip() if i < len(gold_responses) else ""
        pred = r.get("pred_response", "").strip()
        predictions.append(pred)
        references.append([gold])
        log_lines.append(
            "Prompt:\n" + r.get("prompt", "") + "\n"
            "Gold Response: " + str(gold) + "\n"
            "Predicted Response: " + str(pred) + "\n"
            "Raw Output:\n" + r.get("raw_output", "") + "\n"
            + "-" * 50 + "\n"
        )
    sacrebleu = evaluate.load('sacrebleu')
    bleu_result = sacrebleu.compute(predictions=predictions, references=references)
    bleu_score = bleu_result["score"]
    print(f"\nResponse inference finished! BLEU score: {bleu_score:.2f}")
    with open(error_log_file, "w", encoding="utf-8") as f:
        f.writelines(log_lines)
    print(f"All gold and pred comparison logs have been written to {error_log_file}")

def main():
    parser = argparse.ArgumentParser(description="Pipeline Inference: Domain -> State -> Response (Reuse base model + different LoRA)")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-4B", help="Base model path")
    parser.add_argument("--domain_lora", type=str, default="qwen3_lora_output/domain_recognition_train", help="Domain LoRA weights path")
    parser.add_argument("--state_lora", type=str, default="qwen3_lora_output/state_extraction_train_sampled", help="State LoRA weights path")
    parser.add_argument("--response_lora", type=str, default="qwen3_lora_output/response_generation_train", help="Response LoRA weights path")
    parser.add_argument("--vec_file", type=str, default="multiwoz-context-db.vec", help="Vector database file path")
    parser.add_argument("--top_k", type=int, default=2, help="Number of retrieved examples")
    parser.add_argument("--max_samples", type=int, default=100, help="Number of inference samples")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum tokens to generate")
    parser.add_argument("--no_lora", action="store_true", help="Don't use LoRA weights (default: use LoRA)")
    parser.add_argument("--load_in_8bit", action="store_true", help="Whether to load model in 8bit quantization")
    parser.add_argument("--load_in_4bit", action="store_true", help="Whether to load model in 4bit quantization")
    parser.add_argument("--error_log", type=str, default="error_log.txt", help="Error log filename")
    args = parser.parse_args()

    print(f"Vector database file: {args.vec_file}")
    print(f"Number of retrieved examples: {args.top_k}")
    print(f"Maximum samples to process: {args.max_samples}")
    
    # Execute pipeline inference
    results = pipeline_infer(
        args.base_model,
        args.domain_lora, args.state_lora, args.response_lora,
        args.vec_file, args.top_k,
        args.max_new_tokens, args.max_samples,
        not args.no_lora, args.load_in_8bit, args.load_in_4bit
    )
    
    # Extract gold data from results for evaluation
    gold_domains = [r["gold_domain"] for r in results["domain_results"]]
    gold_states = [r["gold_state"] for r in results["state_results"]]
    gold_full_states = [r["gold_full_state"] for r in results["state_results"]]
    gold_responses = [r["gold_response"] for r in results["response_results"]]
    
    print(f"\n=== Processed {results['processed_turns']} dialogue turns ===")
    print("\n=== Domain Evaluation ===")
    domain_error_log = args.error_log.replace('.txt', '_domain.txt')
    evaluate_domains(results["domain_results"], gold_domains, domain_error_log)
    print("\n=== State Evaluation ===")
    evaluate_states_v2(results["state_results"], gold_states, gold_full_states)
    print("\n=== Response Evaluation ===")
    response_error_log = args.error_log.replace('.txt', '_response.txt')
    evaluate_responses(results["response_results"], gold_responses, response_error_log)

if __name__ == "__main__":
    main()