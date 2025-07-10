import json
import re
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import random
import evaluate  # 新增

def load_model(base_model_path, lora_model_path, use_lora=False, load_in_8bit=False, load_in_4bit=False):
    """加载Tokenizer和LoRA模型"""
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
    """加载域识别prompts和gold_domains"""
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    prompts = [item["prompt"] for item in data if "prompt" in item]
    gold_domains = [item["gold_result"].get("domain", "") for item in data if "gold_result" in item]
    return prompts, gold_domains

def infer_domains(tokenizer, model, prompts, max_new_tokens=128, num_samples=None):
    """批量推理，返回推理结果。对prompts随机采样num_samples个样本进行推理。"""
    results = []
    if num_samples is not None and num_samples < len(prompts):
        sampled_indices = random.sample(range(len(prompts)), num_samples)
        sampled_prompts = [prompts[i] for i in sampled_indices]
    else:
        sampled_indices = list(range(len(prompts)))
        sampled_prompts = prompts

    for idx, prompt in tqdm(list(enumerate(sampled_prompts)), desc="推理中"):
        # 构造 chat_template 输入
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # 关闭thinking模式
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        outputs = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
        # 只取新生成的内容
        output_ids = outputs[0][len(model_inputs.input_ids[0]):].tolist()
        generated_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        # 归一化处理
        pred_domain = generated_text.lower().strip()
        results.append({
            "prompt": prompt,
            "pred_domain": pred_domain,  # 直接用归一化后的内容
            "raw_output": generated_text,
            "original_index": sampled_indices[idx]  # 记录原始索引，便于后续对齐gold_domains
        })
    return results

def evaluate_domains(results, gold_domains, error_log_file="error_log.txt"):
    """评估准确率，输出错误日志"""
    correct = 0
    empty_gold_count = 0
    error_log = []
    for i, r in enumerate(results):
        gold = gold_domains[i].lower().strip() if i < len(gold_domains) else ""
        pred = r.get("pred_domain", "").lower().strip()
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
    print(f"\n推理完成！准确率: {accuracy:.2%} ({correct}/{len(results)})")
    print(f"gold_domain 为空的数量: {empty_gold_count}")
    if error_log:
        print(f"❌ 发现 {len(error_log)} 个错误示例，已写入 {error_log_file}")

def load_state_prompts(filename):
    """加载状态抽取prompts和gold_state"""
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    prompts = [item["prompt"] for item in data if "prompt" in item]
    gold_states = [item["gold_result"].get("gt_state", {}) for item in data if "gold_result" in item]
    gold_full_states = [item["gold_result"].get("gt_full_state", {}) for item in data if "gold_result" in item]

    return prompts, gold_states, gold_full_states

def parse_state_output(state_str):
    """将模型输出的state字符串解析为dict"""
    try:
        # 只取第一个大括号内容
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
    """批量推理，返回state抽取结果。对prompts随机采样num_samples个样本进行推理。"""
    results = []
    if num_samples is not None and num_samples < len(prompts):
        sampled_indices = random.sample(range(len(prompts)), num_samples)
        sampled_prompts = [prompts[i] for i in sampled_indices]
    else:
        sampled_indices = list(range(len(prompts)))
        sampled_prompts = prompts

    for idx, prompt in tqdm(list(enumerate(sampled_prompts)), desc="State推理中"):
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
    """将state字典（可能有domain，也可能没有）拉平成归一化(slot, value)集合"""
    flat = set()
    if not isinstance(state, dict):
        return flat
    def norm(x):
        if isinstance(x, str):
            return x.lower().strip()
        return str(x).lower().strip()
    for k, v in state.items():
        if isinstance(v, dict):  # 有domain
            for slot, value in v.items():
                flat.add((norm(slot), norm(value)))
        else:  # 无domain，直接是slot
            flat.add((norm(k), norm(v)))
    return flat

def evaluate_states_v2(results, gold_states, gold_full_states):
    """更细致的state抽取评估，忽略domain，只按slot/value对比"""
    total = len(results)
    perfect = 0
    missing = 0
    badcase = 0
    acceptable_extra = 0

    logs = {
        "perfect": [],
        "missing": [],
        "badcase": [],
        "acceptable_extra": []
    }

    for i, r in enumerate(results):
        gold = gold_states[i] if i < len(gold_states) else {}
        gold_full = gold_full_states[i] if i < len(gold_full_states) else {}
        pred = r.get("pred_state", {})

        gold_slots = flatten_state(gold)
        full_slots = flatten_state(gold_full)
        pred_slots = flatten_state(pred)

        # 完全一致
        if pred_slots == gold_slots:
            perfect += 1
            logs["perfect"].append(f"Prompt:\n{r.get('prompt','')}\nGold: {gold}\nPred: {pred}\nRaw: {r.get('raw_output','')}\n{'-'*50}\n")
            continue

        # 预测多出来的slot
        extra_slots = pred_slots - gold_slots
        # 漏掉的slot
        missing_slots = gold_slots - pred_slots

        # badcase: 预测的slot有不在full_state中的
        if any(slot not in full_slots for slot in extra_slots):
            badcase += 1
            logs["badcase"].append(f"Prompt:\n{r.get('prompt','')}\nGold: {gold}\nFull: {gold_full}\nPred: {pred}\n多预测且不在full_state: {extra_slots - full_slots}\nRaw: {r.get('raw_output','')}\n{'-'*50}\n")
        # 可接受的多预测：多出来的slot都在full_state中
        elif extra_slots and all(slot in full_slots for slot in extra_slots):
            acceptable_extra += 1
            logs["acceptable_extra"].append(f"Prompt:\n{r.get('prompt','')}\nGold: {gold}\nFull: {gold_full}\nPred: {pred}\n多预测但都在full_state: {extra_slots}\nRaw: {r.get('raw_output','')}\n{'-'*50}\n")
        # 漏召回
        elif missing_slots:
            missing += 1
            logs["missing"].append(f"Prompt:\n{r.get('prompt','')}\nGold: {gold}\nFull: {gold_full}\nPred: {pred}\n漏召回: {missing_slots}\nRaw: {r.get('raw_output','')}\n{'-'*50}\n")
        else:
            logs["missing"].append(f"Prompt:\n{r.get('prompt','')}\nGold: {gold}\nFull: {gold_full}\nPred: {pred}\n未知情况\nRaw: {r.get('raw_output','')}\n{'-'*50}\n")

    # 输出统计
    print(f"\nState评估完成！（忽略domain，仅按slot+value）")
    print(f"完全正确: {perfect}/{total}")
    print(f"漏召回: {missing}/{total}")
    print(f"badcase(多预测且不在full_state): {badcase}/{total}")
    print(f"可接受的多预测: {acceptable_extra}/{total}")

    # 写日志
    for k, v in logs.items():
        if v:
            with open(f"{k}_log.txt", "w", encoding="utf-8") as f:
                f.writelines(v)
    print("日志已分别写入 perfect_log.txt, missing_log.txt, badcase_log.txt, acceptable_extra_log.txt")

# 新增：加载response生成prompts和gold_response
def load_response_prompts(filename):
    """加载response生成prompts和gold_response"""
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    prompts = [item["prompt"] for item in data if "prompt" in item]
    gold_responses = [item["gold_result"].get("response", "") for item in data if "gold_result" in item]
    return prompts, gold_responses

# 新增：response批量推理
def infer_responses(tokenizer, model, prompts, max_new_tokens=128, num_samples=None):
    """批量推理，返回response生成结果。"""
    results = []
    if num_samples is not None and num_samples < len(prompts):
        sampled_indices = random.sample(range(len(prompts)), num_samples)
        sampled_prompts = [prompts[i] for i in sampled_indices]
    else:
        sampled_indices = list(range(len(prompts)))
        sampled_prompts = prompts

    for idx, prompt in tqdm(list(enumerate(sampled_prompts)), desc="Response推理中"):
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

# 新增：response评估
def evaluate_responses(results, gold_responses, error_log_file="error_log.txt"):
    """使用BLEU分数评估response生成，并记录所有gold和pred对比日志。"""
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
    # 计算BLEU分数
    sacrebleu = evaluate.load('sacrebleu')
    bleu_result = sacrebleu.compute(predictions=predictions, references=references)
    bleu_score = bleu_result["score"]
    print(f"\nResponse推理完成！BLEU分数: {bleu_score:.2f}")
    with open(error_log_file, "w", encoding="utf-8") as f:
        f.writelines(log_lines)
    print(f"所有gold和pred对比日志已写入 {error_log_file}")

def main():
    parser = argparse.ArgumentParser(description="Domain Inference and Evaluation")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-4B", help="基础模型路径")
    parser.add_argument("--lora_model", type=str, default="./qwen3_lora_output", help="LoRA权重路径")
    parser.add_argument("--prompt_file", type=str, default="domain_recognition_prompts.json", help="Prompt数据文件")
    parser.add_argument("--max_samples", type=int, default=100, help="推理样本数")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="生成最大token数")
    parser.add_argument("--error_log", type=str, default="error_log.txt", help="错误日志文件名")
    parser.add_argument("--use_lora", action="store_true", help="是否加载LoRA权重")
    parser.add_argument("--load_in_8bit", action="store_true", help="是否以8bit量化加载模型")
    parser.add_argument("--load_in_4bit", action="store_true", help="是否以4bit量化加载模型")
    parser.add_argument("--state_prompt_file", type=str, default="state_extraction_prompts.json", help="State抽取Prompt数据文件")
    parser.add_argument("--response_prompt_file", type=str, default="response_generation_prompts.json", help="Response生成Prompt数据文件")
    parser.add_argument("--infer_mode", type=str, choices=["domain", "state", "response"], default="domain", help="推理模式：domain、state或response")
    args = parser.parse_args()

    print("加载模型中...")
    tokenizer, model = load_model(
        args.base_model, args.lora_model, args.use_lora, args.load_in_8bit, args.load_in_4bit
    )

    if args.infer_mode == "domain":
        print("加载Prompt数据中...")
        prompts, gold_domains = load_prompts(args.prompt_file)
        if not prompts:
            raise RuntimeError("未加载到域识别prompts，无法继续。")
        print(f"共加载 {len(prompts)} 条Prompt，准备推理前 {args.max_samples} 条。")
        print("Prompt示例：", prompts[0])
        print("Gold Domain示例：", gold_domains[0])

        results = infer_domains(tokenizer, model, prompts, args.max_new_tokens, args.max_samples)
        # 由于采样，需对gold_domains重新对齐
        if args.max_samples is not None and args.max_samples < len(prompts):
            gold_domains_sampled = [gold_domains[r["original_index"]] for r in results]
        else:
            gold_domains_sampled = gold_domains
        evaluate_domains(results, gold_domains_sampled, args.error_log)

    elif args.infer_mode == "state":
        print("加载State抽取Prompt数据中...")
        state_prompts, gold_states, gold_full_states = load_state_prompts(args.state_prompt_file)
        if not state_prompts:
            raise RuntimeError("未加载到state抽取prompts，无法继续。")
        print(f"共加载 {len(state_prompts)} 条State Prompt，准备推理前 {args.max_samples} 条。")
        print("Prompt示例：", state_prompts[0])
        print("Gold State示例：", gold_states[0])
        print("Gold Full State示例：", gold_full_states[0])
        state_results = infer_states(tokenizer, model, state_prompts, args.max_new_tokens, args.max_samples)
        # 由于采样，需对gold_states和gold_full_states重新对齐
        if args.max_samples is not None and args.max_samples < len(state_prompts):
            gold_states_sampled = [gold_states[r["original_index"]] for r in state_results]
            gold_full_states_sampled = [gold_full_states[r["original_index"]] for r in state_results]
        else:
            gold_states_sampled = gold_states
            gold_full_states_sampled = gold_full_states

        evaluate_states_v2(state_results, gold_states_sampled, gold_full_states_sampled)

    elif args.infer_mode == "response":
        print("加载Response生成Prompt数据中...")
        response_prompts, gold_responses = load_response_prompts(args.response_prompt_file)
        if not response_prompts:
            raise RuntimeError("未加载到response生成prompts，无法继续。")
        print(f"共加载 {len(response_prompts)} 条Response Prompt，准备推理前 {args.max_samples} 条。")
        print("Prompt示例：", response_prompts[0])
        print("Gold Response示例：", gold_responses[0])
        response_results = infer_responses(tokenizer, model, response_prompts, args.max_new_tokens, args.max_samples)
        if args.max_samples is not None and args.max_samples < len(response_prompts):
            gold_responses_sampled = [gold_responses[r["original_index"]] for r in response_results]
        else:
            gold_responses_sampled = gold_responses
        evaluate_responses(response_results, gold_responses_sampled, args.error_log)

if __name__ == "__main__":
    main()