import os
from multiwoz_utils.data_loader import load_multiwoz
from multiwoz_utils.database import MultiWOZDatabase

# 1. Load MultiWOZ dataset
split = 'train'  # Options: 'train', 'validation', 'test'
data = load_multiwoz(split)

# 2. Load database
cur_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(cur_dir, 'multiwoz_database')
database = MultiWOZDatabase(db_path)

# 3. Output 10 complete multi-turn dialogues
count = 0
with open('agent_test_samples.txt', 'w', encoding='utf-8') as f:
    for dialog in data:
        dialogue_id = dialog['dialogue_id'].split('.')[0].lower()
        domain_gt = dialog['services'][0] if dialog['services'] else ''
        # Only take valid domains
        if domain_gt not in ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital']:
            continue
        # Output the entire dialogue
        utterances = dialog['turns']['utterance']
        lines = []
        for i, utt in enumerate(utterances):
            prefix = 'Customer' if i % 2 == 0 else 'Assistant'
            lines.append(f"{prefix}: {utt}")
        f.write('\n'.join(lines) + '\n\n')
        count += 1
        if count >= 10:
            break
print("Generated 10 complete multi-turn dialogues, saved in agent_test_samples.txt") 