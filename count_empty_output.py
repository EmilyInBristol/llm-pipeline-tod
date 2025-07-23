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
            # 判断 output 是否为字符串类型的空字典
          
            if "output" in data and data["output"].strip() == "{}":
                empty_output += 1
        except Exception as e:
            print(f"解析出错: {e}")

if total > 0:
    print(f"output为空字典的行数: {empty_output}")
    print(f"总行数: {total}")
    print(f"占比: {empty_output / total:.2%}")
else:
    print("文件为空或无有效数据") 