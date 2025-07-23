import os
from multiwoz_utils.data_loader import load_multiwoz
from multiwoz_utils.database import MultiWOZDatabase

# 1. 加载 MultiWOZ 数据集
split = 'train'  # 可选 'train', 'validation', 'test'
data = load_multiwoz(split)

# 2. 加载数据库
cur_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(cur_dir, 'multiwoz_database')
database = MultiWOZDatabase(db_path)

# 3. 输出10个完整多轮对话
count = 0
with open('agent_test_samples.txt', 'w', encoding='utf-8') as f:
    for dialog in data:
        dialogue_id = dialog['dialogue_id'].split('.')[0].lower()
        domain_gt = dialog['services'][0] if dialog['services'] else ''
        # 只取有效领域
        if domain_gt not in ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital']:
            continue
        # 输出一整个对话
        utterances = dialog['turns']['utterance']
        lines = []
        for i, utt in enumerate(utterances):
            prefix = 'Customer' if i % 2 == 0 else 'Assistant'
            lines.append(f"{prefix}: {utt}")
        f.write('\n'.join(lines) + '\n\n')
        count += 1
        if count >= 10:
            break
print("已生成10个多轮完整对话，保存在 agent_test_samples.txt") 