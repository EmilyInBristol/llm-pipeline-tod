import json
import re
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import random
import evaluate  # New
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # New

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

def load_prompts(filename):
    """Load domain recognition prompts and gold_domains"""
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    prompts = [item["prompt"] for item in data if "prompt" in item]
    gold_domains = [item["gold_result"].get("domain", "") for item in data if "gold_result" in item]
    return prompts, gold_domains

def infer_domains(tokenizer, model, prompts, max_new_tokens=128, num_samples=None):
    """Batch inference, return inference results. Randomly sample num_samples from prompts for inference."""
    results = []
    if num_samples is not None and num_samples < len(prompts):
        sampled_indices = random.sample(range(len(prompts)), num_samples)
        sampled_prompts = [prompts[i] for i in sampled_indices]
    else:
        sampled_indices = list(range(len(prompts)))
        sampled_prompts = prompts

    for idx, prompt in tqdm(list(enumerate(sampled_prompts)), desc="Inferencing"):
        # Construct chat_template input
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # Turn off thinking mode
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        outputs = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
        # Only take newly generated content
        output_ids = outputs[0][len(model_inputs.input_ids[0]):].tolist()
        generated_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        # Normalization processing
        pred_domain = generated_text.lower().strip()
        results.append({
            "prompt": prompt,
            "pred_domain": pred_domain,  # Use normalized content directly
            "raw_output": generated_text,
            "original_index": sampled_indices[idx]  # Record original index for subsequent alignment with gold_domains
        })
    return results

def evaluate_domains(results, gold_domains, error_log_file="error_log.txt"):
    """Evaluate accuracy, output detailed evaluation report and error log"""
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
    print(f"\nInference finished! Accuracy: {accuracy:.2%} ({correct}/{len(results)})")
    print(f"Number of empty gold_domain: {empty_gold_count}")
    if error_log:
        print(f"❌ Found {len(error_log)} error cases, written to {error_log_file}")
    # New: Detailed evaluation output
    # Auto-statistics label set
    domain_labels = sorted(list(set([g for g in golds if g])))
    print("\nDetailed evaluation report:")
    print(classification_report(golds, preds, labels=domain_labels, digits=3))
    print("Confusion Matrix:")
    print(confusion_matrix(golds, preds, labels=domain_labels))

def load_state_prompts(filename):
    """Load state extraction prompts and gold_state"""
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    prompts = [item["prompt"] for item in data if "prompt" in item]
    gold_states = [item["gold_result"].get("gt_state", {}) for item in data if "gold_result" in item]
    gold_full_states = [item["gold_result"].get("gt_full_state", {}) for item in data if "gold_result" in item]

    return prompts, gold_states, gold_full_states

def parse_state_output(state_str):
    """Parse the state string output from model into dict"""
    try:
        # Only take the first brace content
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
    """Batch inference, return state extraction results. Randomly sample num_samples from prompts for inference."""
    results = []
    if num_samples is not None and num_samples < len(prompts):
        sampled_indices = random.sample(range(len(prompts)), num_samples)
        sampled_prompts = [prompts[i] for i in sampled_indices]
    else:
        sampled_indices = list(range(len(prompts)))
        sampled_prompts = prompts

    for idx, prompt in tqdm(list(enumerate(sampled_prompts)), desc="State Inferencing"):
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

def flatten_state(state):
    """Flatten state dictionary (may or may not have domain) into normalized (slot, value) set"""
    flat = set()
    if not isinstance(state, dict):
        return flat
    def norm(x):
        if isinstance(x, str):
            return x.lower().strip()
        return str(x).lower().strip()
    for k, v in state.items():
        if isinstance(v, dict):  # Has domain
            for slot, value in v.items():
                flat.add((norm(slot), norm(value)))
        else:  # No domain, directly slot
            flat.add((norm(k), norm(v)))
    return flat

def evaluate_states_v2(results, gold_states, gold_full_states):
    """More detailed state extraction evaluation, ignore domain, only compare by slot/value"""
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

    # New: Slot-value level F1 statistics
    all_tp, all_fp, all_fn = 0, 0, 0  # Micro F1

    for i, r in enumerate(results):
        gold = gold_states[i] if i < len(gold_states) else {}
        gold_full = gold_full_states[i] if i < len(gold_full_states) else {}
        pred = r.get("pred_state", {})

        gold_slots = flatten_state(gold)
        full_slots = flatten_state(gold_full)
        pred_slots = flatten_state(pred)

        # Slot-value level F1 statistics
        tp = len(pred_slots & gold_slots)
        fp = len(pred_slots - gold_slots)
        fn = len(gold_slots - pred_slots)
        all_tp += tp
        all_fp += fp
        all_fn += fn

        # JGA: Complete match
        if pred_slots == gold_slots:
            JGA += 1
            logs["JGA"].append(f"Prompt:\n{r.get('prompt','')}\nGold: {gold}\nPred: {pred}\nRaw: {r.get('raw_output','')}\n{'-'*50}\n")
            continue

        # Extra predicted slots
        extra_slots = pred_slots - gold_slots
        # Missing slots
        missing_slots = gold_slots - pred_slots

        # Badcase: predicted slots not in full_state
        if any(slot not in full_slots for slot in extra_slots):
            badcase += 1
            logs["badcase"].append(f"Prompt:\n{r.get('prompt','')}\nGold: {gold}\nFull: {gold_full}\nPred: {pred}\nExtra predictions not in full_state: {extra_slots - full_slots}\nRaw: {r.get('raw_output','')}\n{'-'*50}\n")
        # Acceptable extra: extra slots are all in full_state
        elif extra_slots and all(slot in full_slots for slot in extra_slots):
            acceptable_extra += 1
            logs["acceptable_extra"].append(f"Prompt:\n{r.get('prompt','')}\nGold: {gold}\nFull: {gold_full}\nPred: {pred}\nExtra predictions but all in full_state: {extra_slots}\nRaw: {r.get('raw_output','')}\n{'-'*50}\n")
        # Missing recall
        elif missing_slots:
            missing += 1
            logs["missing"].append(f"Prompt:\n{r.get('prompt','')}\nGold: {gold}\nFull: {gold_full}\nPred: {pred}\nMissing recall: {missing_slots}\nRaw: {r.get('raw_output','')}\n{'-'*50}\n")
        else:
            logs["missing"].append(f"Prompt:\n{r.get('prompt','')}\nGold: {gold}\nFull: {gold_full}\nPred: {pred}\nUnknown case\nRaw: {r.get('raw_output','')}\n{'-'*50}\n")

    # Output statistics
    print(f"\nState evaluation finished! (Ignore domain, only compare slot+value)")
    print(f"JGA (Joint Goal Accuracy): {JGA}/{total}")
    print(f"Missing: {missing}/{total}")
    print(f"Badcase (extra slots not in full_state): {badcase}/{total}")
    print(f"Acceptable extra: {acceptable_extra}/{total}")

    # New: Slot-value level F1 output
    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    print(f"Slot-value level Micro Precision: {precision:.4f}")
    print(f"Slot-value level Micro Recall: {recall:.4f}")
    print(f"Slot-value level Micro F1: {f1:.4f}")

    # Write logs
    for k, v in logs.items():
        if v:
            with open(f"{k}_log.txt", "w", encoding="utf-8") as f:
                f.writelines(v)
    print("Logs have been written to JGA_log.txt, missing_log.txt, badcase_log.txt, acceptable_extra_log.txt")

# New: Load response generation prompts and gold_response
def load_response_prompts(filename):
    """Load response generation prompts and gold_response"""
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    prompts = [item["prompt"] for item in data if "prompt" in item]
    gold_responses = [item["gold_result"].get("response", "") for item in data if "gold_result" in item]
    return prompts, gold_responses

# New: Response batch inference
def infer_responses(tokenizer, model, prompts, max_new_tokens=128, num_samples=None):
    """Batch inference, return response generation results."""
    results = []
    if num_samples is not None and num_samples < len(prompts):
        sampled_indices = random.sample(range(len(prompts)), num_samples)
        sampled_prompts = [prompts[i] for i in sampled_indices]
    else:
        sampled_indices = list(range(len(prompts)))
        sampled_prompts = prompts

    for idx, prompt in tqdm(list(enumerate(sampled_prompts)), desc="Response Inferencing"):
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
            "pred_response": generated_text,
            "raw_output": generated_text,
            "original_index": sampled_indices[idx]
        })
    return results

# New: Response evaluation
def evaluate_responses(results, gold_responses, error_log_file="error_log.txt"):
    """Use BLEU score to evaluate response generation, and record all gold and pred comparison logs."""
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
    # Calculate BLEU score
    sacrebleu = evaluate.load('sacrebleu')
    bleu_result = sacrebleu.compute(predictions=predictions, references=references)
    bleu_score = bleu_result["score"]
    print(f"\nResponse inference finished! BLEU score: {bleu_score:.2f}")
    with open(error_log_file, "w", encoding="utf-8") as f:
        f.writelines(log_lines)
    print(f"All gold and pred comparison logs have been written to {error_log_file}")

def main():
    parser = argparse.ArgumentParser(description="Domain Inference and Evaluation")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-4B", help="Base model path")
    parser.add_argument("--lora_model", type=str, default="./qwen3_lora_output", help="LoRA weights path")
    parser.add_argument("--prompt_file", type=str, default="domain_recognition_prompts.json", help="Prompt data file")
    parser.add_argument("--max_samples", type=int, default=100, help="Number of inference samples")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum tokens to generate")
    parser.add_argument("--error_log", type=str, default="error_log.txt", help="Error log filename")
    parser.add_argument("--use_lora", action="store_true", help="Whether to load LoRA weights")
    parser.add_argument("--load_in_8bit", action="store_true", help="Whether to load model in 8bit quantization")
    parser.add_argument("--load_in_4bit", action="store_true", help="Whether to load model in 4bit quantization")
    parser.add_argument("--state_prompt_file", type=str, default="state_extraction_prompts.json", help="State extraction prompt data file")
    parser.add_argument("--response_prompt_file", type=str, default="response_generation_prompts.json", help="Response generation prompt data file")
    parser.add_argument("--infer_mode", type=str, choices=["domain", "state", "response"], default="domain", help="Inference mode: domain, state, or response")
    args = parser.parse_args()

    print("Loading model...")
    tokenizer, model = load_model(
        args.base_model, args.lora_model, args.use_lora, args.load_in_8bit, args.load_in_4bit
    )

    if args.infer_mode == "domain":
        print("Loading prompt data...")
        prompts, gold_domains = load_prompts(args.prompt_file)
        if not prompts:
            raise RuntimeError("Failed to load domain recognition prompts, cannot continue.")
        print(f"Loaded {len(prompts)} prompts, preparing to infer first {args.max_samples}.")
        print("Prompt example:", prompts[0])
        print("Gold Domain example:", gold_domains[0])

        results = infer_domains(tokenizer, model, prompts, args.max_new_tokens, args.max_samples)
        # Due to sampling, need to realign gold_domains
        if args.max_samples is not None and args.max_samples < len(prompts):
            gold_domains_sampled = [gold_domains[r["original_index"]] for r in results]
        else:
            gold_domains_sampled = gold_domains
        evaluate_domains(results, gold_domains_sampled, args.error_log)

    elif args.infer_mode == "state":
        print("Loading state extraction prompt data...")
        state_prompts, gold_states, gold_full_states = load_state_prompts(args.state_prompt_file)
        if not state_prompts:
            raise RuntimeError("Failed to load state extraction prompts, cannot continue.")
        print(f"Loaded {len(state_prompts)} state prompts, preparing to infer first {args.max_samples}.")
        print("Prompt example:", state_prompts[0])
        print("Gold State example:", gold_states[0])
        print("Gold Full State example:", gold_full_states[0])
        state_results = infer_states(tokenizer, model, state_prompts, args.max_new_tokens, args.max_samples)
        # Due to sampling, need to realign gold_states and gold_full_states
        if args.max_samples is not None and args.max_samples < len(state_prompts):
            gold_states_sampled = [gold_states[r["original_index"]] for r in state_results]
            gold_full_states_sampled = [gold_full_states[r["original_index"]] for r in state_results]
        else:
            gold_states_sampled = gold_states
            gold_full_states_sampled = gold_full_states

        evaluate_states_v2(state_results, gold_states_sampled, gold_full_states_sampled)

    elif args.infer_mode == "response":
        print("Loading response generation prompt data...")
        response_prompts, gold_responses = load_response_prompts(args.response_prompt_file)
        if not response_prompts:
            raise RuntimeError("Failed to load response generation prompts, cannot continue.")
        print(f"Loaded {len(response_prompts)} response prompts, preparing to infer first {args.max_samples}.")
        print("Prompt example:", response_prompts[0])
        print("Gold Response example:", gold_responses[0])
        response_results = infer_responses(tokenizer, model, response_prompts, args.max_new_tokens, args.max_samples)
        if args.max_samples is not None and args.max_samples < len(response_prompts):
            gold_responses_sampled = [gold_responses[r["original_index"]] for r in response_results]
        else:
            gold_responses_sampled = gold_responses
        evaluate_responses(response_results, gold_responses_sampled, args.error_log)

if __name__ == "__main__":
    main()