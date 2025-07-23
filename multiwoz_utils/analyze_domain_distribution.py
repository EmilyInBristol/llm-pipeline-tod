import sys
from collections import Counter, defaultdict
from data_loader import load_multiwoz
import numpy as np
from dialog_iterator import iterate_dialogues
from database import default_database

splits = ['train', 'validation', 'test']

for split in splits:
    print(f"===== {split} split =====")
    data = load_multiwoz(split)
    domain_counter = Counter()
    single_domain_count = 0
    multi_domain_count = 0
    single_domain_detail = Counter()
    # 新增：统计每个domain下的对话长度
    domain_lengths = defaultdict(list)
    for dialog in data:
        services = dialog.get('services', [])
        domain = services[0] if services else ''
        domain_counter[domain] += 1
        turns = dialog.get('turns', {}).get('utterance', [])
        dialog_len = len(turns)
        if len(services) == 1:
            single_domain_count += 1
            single_domain_detail[domain] += 1
            # 记录所有主domain的对话长度
            domain_lengths[domain].append(dialog_len)
        elif len(services) > 1:
            multi_domain_count += 1
        
    total = sum(domain_counter.values())
    for domain, count in domain_counter.most_common():
        print(f"{domain or '[EMPTY]'}: {count}")
    print(f"Total dialogues: {total}")
    print(f"Single-domain dialogues: {single_domain_count}")
    print(f"Multi-domain dialogues: {multi_domain_count}")
    print("Single-domain detail:")
    for domain, count in single_domain_detail.most_common():
        print(f"  {domain or '[EMPTY]'}: {count}")
    print("Domain dialogue length (turns):")
    for domain, lengths in domain_lengths.items():
        if not lengths:
            continue
        mean_len = np.mean(lengths)
        median_len = np.median(lengths)
        print(f"  {domain or '[EMPTY]'}: mean={mean_len:.2f}, median={median_len}")
    print()

    