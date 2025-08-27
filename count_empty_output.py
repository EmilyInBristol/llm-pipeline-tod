import json

file_path = "state_extraction_train_sampled.jsonl"
total = 0
empty_output = 0

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        total += 1
        try:
            data = json.loads(line)
            # Check if output is an empty dictionary string
          
            if "output" in data and data["output"].strip() == "{}":
                empty_output += 1
        except Exception as e:
            print(f"Parsing error: {e}")

if total > 0:
    print(f"Lines with empty output: {empty_output}")
    print(f"Total lines: {total}")
    print(f"Percentage: {empty_output / total:.2%}")
else:
    print("File is empty or has no valid data") 