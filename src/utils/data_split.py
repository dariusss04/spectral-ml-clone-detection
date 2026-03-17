# Split programs into train/validation/test without clone leakage.

import os
import random
import shutil
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("PCD_DATA_DIR", REPO_ROOT / "data"))
SAMPLES_FILE = Path(os.getenv("PCD_SAMPLES_FILE", DATA_DIR / "meta" / "programs_metadata.txt"))
EXTRACTED_DIR = Path(os.getenv("PCD_EXTRACTED_DIR", DATA_DIR / "features"))
SPLITS_DIR = Path(os.getenv("PCD_SPLITS_DIR", DATA_DIR / "splits"))

def parse_samples(samples_file_path):
    """
    Parse samples.txt and group programs by their equivalency class using the base project name (excluding version) 
    and binary name. Programs that are clones (same project and binary name) are grouped together.
    """
    equivalency_classes = defaultdict(list)
    
    with open(samples_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            index = int(lines[i].strip())
            program_info = lines[i + 1].strip()
            
            program_parts = program_info.split('_')
            
            project_name_with_version = program_parts[0]
            base_project_name = project_name_with_version.split('-')[0]
            
            binary_name = program_parts[-1]
            
            clone_group_key = f"{base_project_name}_{binary_name}"
            
            equivalency_classes[clone_group_key].append(index)
    
    return equivalency_classes

def split_programs(equivalency_classes, train_ratio=0.8, val_ratio=0.1):
    """
    Shuffle and split program groups (equivalency classes) into train, validation, and test.
    Programs in the same equivalency class (clones) must stay together in the same split.
    """
    all_groups = list(equivalency_classes.values())
    random.shuffle(all_groups)
    
    total_groups = len(all_groups)
    train_size = int(total_groups * train_ratio)
    val_size = int(total_groups * val_ratio)

    train_groups = all_groups[:train_size]
    val_groups = all_groups[train_size:train_size + val_size]
    test_groups = all_groups[train_size + val_size:]
    
    return train_groups, val_groups, test_groups

def check_overlap(train_programs, val_programs, test_programs):
    """
    Check for overlap between the train, validation, and test splits.
    Returns a list of programs that are in more than one split.
    """
    train_set = set(train_programs)
    val_set = set(val_programs)
    test_set = set(test_programs)
    
    overlap_train_val = train_set.intersection(val_set)
    overlap_train_test = train_set.intersection(test_set)
    overlap_val_test = val_set.intersection(test_set)
    
   
    overlap = overlap_train_val.union(overlap_train_test).union(overlap_val_test)
    
    return overlap

def save_split(program_indices, data_path, target_folder):
    """
    Save the .pt files 
    """
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    for index in program_indices:
        source_file = os.path.join(data_path, f'program{index}.pt')
        target_file = os.path.join(target_folder, f'program{index}.pt')
        shutil.copy(source_file, target_file)

def count_clones(groups):
    """
    Count the total number of clone programs by summing the sizes of the groups.
    """
    return sum(len(group) for group in groups)

def split(samples_file_path, data_path, output_path, train_ratio=0.8, val_ratio=0.1):

    equivalency_classes = parse_samples(samples_file_path)
    
    train_groups, val_groups, test_groups = split_programs(equivalency_classes, train_ratio, val_ratio)
    
    train_programs = [index for group in train_groups for index in group]
    val_programs = [index for group in val_groups for index in group]
    test_programs = [index for group in test_groups for index in group]
    
    save_split(train_programs, data_path, os.path.join(output_path, 'train'))
    save_split(val_programs, data_path, os.path.join(output_path, 'validation'))
    save_split(test_programs, data_path, os.path.join(output_path, 'test'))
    
    overlap = check_overlap(train_programs, val_programs, test_programs)
    
    total_programs = len(train_programs) + len(val_programs) + len(test_programs)
    split_stats_file = os.path.join(output_path, 'split_stats.txt')
    
    clones_train = count_clones(train_groups)
    clones_val = count_clones(val_groups)
    clones_test = count_clones(test_groups)
    
    with open(split_stats_file, 'w') as f:
        f.write(f"Total number of programs: {total_programs}\n")
        f.write(f"Train size: {len(train_programs)}\n")
        f.write(f"Validation size: {len(val_programs)}\n")
        f.write(f"Test size: {len(test_programs)}\n")
        f.write(f"Number of clone groups in Train: {len(train_groups)}\n")
        f.write(f"Number of clone groups in Validation: {len(val_groups)}\n")
        f.write(f"Number of clone groups in Test: {len(test_groups)}\n")
        f.write(f"Overlap between splits: {overlap}\n")
        
        if overlap:
            f.write(f"Programs found in more than one split: {list(overlap)}\n")
        else:
            f.write("No overlap found between splits.\n")
        
        
        f.write(f"Total clone programs in Train: {clones_train}\n")
        f.write(f"Total clone programs in Validation: {clones_val}\n")
        f.write(f"Total clone programs in Test: {clones_test}\n")

if __name__ == "__main__":
    split(SAMPLES_FILE, EXTRACTED_DIR, SPLITS_DIR)
