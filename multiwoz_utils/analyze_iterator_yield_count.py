from data_loader import load_multiwoz
from dialog_iterator import iterate_dialogues
from database import default_database
from collections import Counter

splits = ['train', 'validation', 'test']

for split in splits:
    print(f"===== {split} split =====")
    data = load_multiwoz(split)
    # HuggingFace Dataset 需转为 list 才能被 iterate_dialogues 正确遍历
    data = list(data)
    domain_counter = Counter()
    for sample in iterate_dialogues(data, default_database):
        domain = sample['metadata']['domain']
        domain_counter[domain] += 1
    print("各domain下的turn-level样本数量：")
    for domain, count in domain_counter.items():
        print(f"  {domain}: {count}")
    print(f"Total turn-level samples from iterate_dialogues: {sum(domain_counter.values())}\n") 