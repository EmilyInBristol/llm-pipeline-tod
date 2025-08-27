from data_loader import load_multiwoz
from dialog_iterator import iterate_dialogues
from database import default_database
from collections import Counter

splits = ['train', 'validation', 'test']

for split in splits:
    print(f"===== {split} split =====")
    data = load_multiwoz(split)
    # HuggingFace Dataset needs to be converted to list for iterate_dialogues to work correctly
    data = list(data)
    domain_counter = Counter()
    for sample in iterate_dialogues(data, default_database):
        domain = sample['metadata']['domain']
        domain_counter[domain] += 1
    print("Turn-level sample counts by domain:")
    for domain, count in domain_counter.items():
        print(f"  {domain}: {count}")
    print(f"Total turn-level samples from iterate_dialogues: {sum(domain_counter.values())}\n") 