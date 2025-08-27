# multiwoz_utils/data_loader.py

from datasets import load_dataset, DatasetDict
from functools import lru_cache
import os
import shutil

@lru_cache(maxsize=3)
def load_multiwoz(split: str = 'train'):
    """
    Load and cache the MultiWOZ 2.2 dataset split.

    Args:
        split (str): One of 'train', 'validation', 'test'.

    Returns:
        Dataset: HuggingFace Dataset object for the given split.
    """
    try:
        # Try normal loading
        dataset = load_dataset('multi_woz_v22')
        return dataset[split]
    except Exception as e:
        print(f"Normal loading failed: {e}")
        
        try:
            # Try forced re-download
            print("Trying forced re-download...")
            dataset = load_dataset('multi_woz_v22', download_mode='force_redownload')
            return dataset[split]
        except Exception as e2:
            print(f"Force download also failed: {e2}")
            
            try:
                # Clear cache and retry
                print("Clearing cache and retrying...")
                cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
                multi_woz_cache = os.path.join(cache_dir, "multi_woz_v22")
                if os.path.exists(multi_woz_cache):
                    shutil.rmtree(multi_woz_cache)
                    print(f"Cache cleared: {multi_woz_cache}")
                
                dataset = load_dataset('multi_woz_v22')
                return dataset[split]
            except Exception as e3:
                print(f"All attempts failed: {e3}")
                raise RuntimeError(f"Unable to load MultiWOZ dataset: {e3}")
