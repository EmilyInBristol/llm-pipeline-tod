import json
import random

input_file = "state_extraction_train.jsonl"
output_file = "state_extraction_train_sampled.jsonl"
empty_label_keep_ratio = 0.2  # Keep 20% of empty label samples

with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        data = json.loads(line)
        output = data.get("output", "").strip()
        if output == "{}":
            if random.random() < empty_label_keep_ratio:
                fout.write(line)
        else:
            fout.write(line) 