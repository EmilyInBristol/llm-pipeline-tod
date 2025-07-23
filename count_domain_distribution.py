import json
from collections import Counter
import matplotlib.pyplot as plt


def count_gold_domain_jsonl(jsonl_file):
    counter = Counter()
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            gold = obj.get("output", None)
            if gold:
                counter[gold] += 1
    return counter


def count_gold_domain_json(json_file, max_items=None):
    counter = Counter()
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for i, obj in enumerate(data):
            if max_items is not None and i >= max_items:
                break
            gold = obj.get("gold_result", {}).get("domain", None)
            if gold:
                counter[gold] += 1
    return counter


def plot_domain_distribution(train_counter, eval_counter, train_label, eval_label):
    domains = sorted(set(train_counter.keys()) | set(eval_counter.keys()))
    train_counts = [train_counter.get(domain, 0) for domain in domains]
    eval_counts = [eval_counter.get(domain, 0) for domain in domains]

    x = range(len(domains))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar([i - width/2 for i in x], train_counts, width=width, label=train_label)
    plt.bar([i + width/2 for i in x], eval_counts, width=width, label=eval_label)
    plt.xticks(x, domains, rotation=45)
    plt.ylabel('Count')
    plt.title('Gold Domain Distribution: Train vs Eval')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_counter = count_gold_domain_jsonl("domain_recognition_train.jsonl")
    eval_counter = count_gold_domain_json("domain_recognition_prompts.json", max_items=300)

    print(f"[Train gold_domain distribution]\n{train_counter}\n")
    print(f"[Eval gold_domain distribution (first 300)]\n{eval_counter}\n")

    plot_domain_distribution(train_counter, eval_counter, "Train", "Eval (first 300)") 