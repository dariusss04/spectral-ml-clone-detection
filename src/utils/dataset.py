# Siamese dataset with caching and dynamic clone/non-clone pairing.

import os
import random
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("PCD_DATA_DIR", REPO_ROOT / "data"))
SAMPLES_FILE = Path(os.getenv("PCD_SAMPLES_FILE", DATA_DIR / "samples.txt"))
CACHE_DIR = Path(os.getenv("PCD_CACHE_DIR", DATA_DIR / "datasetCache"))

CACHE_DIR.mkdir(parents=True, exist_ok=True)

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
            
            project_name_with_version = program_parts[0]
            base_project_name = project_name_with_version.split('-')[0]
            binary_name = program_parts[-1]
            
            index_to_name[index] = (base_project_name, binary_name)
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

class SiameseDataset(Dataset):
    """
    Dataset that yields (program1, program2), label with cached features.
    """
    def __init__(self, split_folder, samples_file_path, top_k_eigenvalues=100, cache_dir=CACHE_DIR):
        self.top_k_eigenvalues = top_k_eigenvalues
        self.split_folder = split_folder
        self.cache_dir = cache_dir

        self.program_indices = [int(f.replace('program', '').replace('.pt', '')) 
                                for f in os.listdir(split_folder) if f.endswith('.pt')]

        self.index_to_name = parse_samples(samples_file_path)
        self.clones, self.non_clones = build_clone_non_clone_dicts(self.program_indices, self.index_to_name)
        self.programs = {idx: self.load_or_cache_program(idx) for idx in self.program_indices}

    def load_or_cache_program(self, idx):
        cache_path = os.path.join(self.cache_dir, f'program{idx}_cache.pt')

        if os.path.exists(cache_path):
            program_features = torch.load(cache_path)
        else:
            program_features = torch.load(os.path.join(self.split_folder, f'program{idx}.pt'))
            torch.save(program_features, cache_path)
        
        return program_features
    
    def __len__(self):
        return len(self.program_indices)
    
    def __getitem__(self, idx):
        """
        Sample a clone or non-clone pair with a fixed probability.
        """
        program1_index = self.program_indices[idx]
        program1_features = self.programs[program1_index]

        if random.random() < 0.05 and self.clones.get(program1_index):
            program2_index = random.choice(self.clones[program1_index])
            label = 1
        else:
            program2_index = random.choice(self.non_clones[program1_index])
            label = -1

        program2_features = self.programs[program2_index]
        pair = (program1_features, program2_features)
        
        return pair, torch.tensor(label, dtype=torch.float32)
