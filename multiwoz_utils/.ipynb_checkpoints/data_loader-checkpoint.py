# multiwoz_utils/data_loader.py

from datasets import load_dataset, DatasetDict
from functools import lru_cache

@lru_cache(maxsize=3)
def load_multiwoz(split: str = 'train'):
    """
    Load and cache the MultiWOZ 2.2 dataset split.

    Args:
        split (str): One of 'train', 'validation', 'test'.

    Returns:
        Dataset: HuggingFace Dataset object for the given split.
    """
    dataset = load_dataset('multi_woz_v22')
    return dataset[split]
