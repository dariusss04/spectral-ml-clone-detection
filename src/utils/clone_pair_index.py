# Clone/non-clone dictionary builder for the test split.

import os
from pathlib import Path
from collections import defaultdict

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("PCD_DATA_DIR", REPO_ROOT / "data"))
SAMPLES_FILE = Path(os.getenv("PCD_SAMPLES_FILE", DATA_DIR / "samples.txt"))
TEST_SPLIT_DIR = Path(os.getenv("PCD_TEST_SPLIT_DIR", DATA_DIR / "splits" / "test"))
DICT_DIR = Path(os.getenv("PCD_DICT_DIR", DATA_DIR / "dictionaries"))

DICT_DIR.mkdir(parents=True, exist_ok=True)

def parse_samples(samples_file_path):
    """
    Parse samples.txt into an index -> (project, binary) mapping.
    """
    index_to_name = {}
    with open(samples_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            index = int(lines[i].strip())
            program_info = lines[i + 1].strip()
            program_parts = program_info.split('_')
            project_name = program_parts[0].split('-')[0]
            binary_name = program_parts[-1]
            index_to_name[index] = (project_name, binary_name)
    return index_to_name

def build_clone_non_clone_dicts(indices, index_to_name):
    """
    Build clone/non-clone candidate lists for each index.
    """
    clones = defaultdict(list)
    non_clones = defaultdict(list)
    
    name_to_indices = defaultdict(list)
    for idx in indices:
        project_name, binary_name = index_to_name[idx]
        name_to_indices[(project_name, binary_name)].append(idx)

    for idx in indices:
        project_name, binary_name = index_to_name[idx]
        clone_group = name_to_indices[(project_name, binary_name)]
        
        clones[idx] = [i for i in clone_group if i != idx]
        non_clones[idx] = [i for i in indices if i not in clone_group]

    return clones, non_clones

test_program_indices = [
    int(f.replace('program', '').replace('.pt', ''))
    for f in os.listdir(TEST_SPLIT_DIR) if f.endswith('.pt')
]

index_to_name = parse_samples(SAMPLES_FILE)
clones, non_clones = build_clone_non_clone_dicts(test_program_indices, index_to_name)

torch.save(clones, os.path.join(DICT_DIR, 'clones_dict.pt'))
torch.save(non_clones, os.path.join(DICT_DIR, 'non_clones_dict.pt'))

print("Clone and non-clone dictionaries created and saved successfully.")
