import json

input_file = 'state_extraction_prompts.json'
output_file = 'state_extraction_prompts.filtered.json'

with open(input_file, 'r', encoding='utf-8') as fin:
    data_list = json.load(fin)

total = len(data_list)
filtered = []
for item in data_list:
    gold_result = item.get('gold_result', {})
    gt_state = gold_result.get('gt_full_state', None)
    if gt_state not in (None, {}, ''):
        filtered.append(item)
kept = len(filtered)

with open(output_file, 'w', encoding='utf-8') as fout:
    json.dump(filtered, fout, ensure_ascii=False, indent=2)

print(f'Original count: {total}')
print(f'Kept count: {kept}')
print(f'Filtered count: {total - kept}') 