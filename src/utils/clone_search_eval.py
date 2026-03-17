# Clone search evaluation for convergence experiments.

import os
import csv
import time
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from utils.dataset import parse_samples
from models.siamese_mlp_relu_base import BaseSiameseNetwork

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("PCD_DATA_DIR", REPO_ROOT / "data"))
RESULTS_DIR = Path(os.getenv("PCD_RESULTS_DIR", REPO_ROOT / "results"))
SAMPLES_FILE = Path(os.getenv("PCD_SAMPLES_FILE", DATA_DIR / "samples.txt"))
TEST_SPLIT_DIR = Path(os.getenv("PCD_TEST_SPLIT_DIR", DATA_DIR / "splits" / "test"))
PARAMS_PATH = Path(os.getenv(
    "PCD_PARAMS_PATH",
    REPO_ROOT / "experiments" / "basicMultiLayersArchitectures" / "architecture12L" / "parameters12L.pth"
))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def parse_samples(samples_file_path):
    """
    Parse samples.txt into an index -> metadata mapping.
    """
    with open(samples_file_path, 'r') as f:
        lines = f.readlines()

    sample_dict = {}
    for i in range(0, len(lines), 2):
        index = int(lines[i].strip())
        program_info = lines[i + 1].strip()
        sample_dict[index] = program_info
    return sample_dict


def filter_programs_by_type(samples_dict, split_folder, search_type, version):
    """
    Filter program indices by search type and version.
    """
    filtered_indices = []
    available_programs = [int(f.replace('program', '').replace('.pt', '')) for f in os.listdir(split_folder) if f.endswith('.pt')]

    for idx, info in samples_dict.items():
        if idx in available_programs:
            program_parts = info.split('_')
            if search_type == "o" and version == program_parts[-2]:
                filtered_indices.append(idx)
    return filtered_indices


def get_subset(indices, fraction):
    """
    Sample a subset of indices by fraction.
    """
    if not indices:
        return [], "0%"
    subset = random.sample(indices, max(1, int(fraction * len(indices))))
    subset_percentage_str = f"{int(fraction * 100)}%"
    return subset, subset_percentage_str


def isAClone(indexA, indexB, samples_dict):
    """
    Determine whether two indices are clones based on metadata.
    """
    programA_info = samples_dict.get(indexA, "")
    programB_info = samples_dict.get(indexB, "")

    if not programA_info or not programB_info:
        return False

    programA_parts = programA_info.split('_')
    programB_parts = programB_info.split('_')

    projectA_name = programA_parts[0].split('-')[0]
    projectB_name = programB_parts[0].split('-')[0]

    binaryA_name = programA_parts[-1]
    binaryB_name = programB_parts[-1]

    return projectA_name == projectB_name and binaryA_name == binaryB_name


def run_clone_search(target_idx, repository_indices, model, split_folder, samples_dict):
    """
    Compute best-match similarity for one target against the repository.
    """
    model.eval()

    target_features_path = os.path.join(split_folder, f'program{target_idx}.pt')
    if not os.path.exists(target_features_path):
        return None, None, target_idx, None, None

    target_features = torch.load(target_features_path)
    target_concat = torch.cat([target_features['eigenvalues'], target_features['num_edges']], dim=0).unsqueeze(0)

    highest_similarity = -1
    best_match_idx = None

    for repo_idx in repository_indices:
        repo_features_path = os.path.join(split_folder, f'program{repo_idx}.pt')
        if not os.path.exists(repo_features_path):
            continue

        repo_features = torch.load(repo_features_path)
        repo_concat = torch.cat([repo_features['eigenvalues'], repo_features['num_edges']], dim=0).unsqueeze(0)

        output1, output2 = model(target_concat, repo_concat)
        output1 = F.normalize(output1, dim=-1)
        output2 = F.normalize(output2, dim=-1)

        similarity = F.cosine_similarity(output1, output2).item()

        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match_idx = repo_idx

    ground_truth = isAClone(target_idx, best_match_idx, samples_dict) if best_match_idx is not None else False
    result = 1 if ground_truth and highest_similarity > 0.5 else 0

    return highest_similarity, result, target_idx, best_match_idx, ground_truth


def clone_search(search_type, version1, version2, model, fraction):
    """
    Run clone search and write results to CSV.
    """
    samples_dict = parse_samples(SAMPLES_FILE)

    target_indices = filter_programs_by_type(samples_dict, TEST_SPLIT_DIR, search_type, version1)
    repository_indices = filter_programs_by_type(samples_dict, TEST_SPLIT_DIR, search_type, version2)

    target_subset, subset_percentage_str = get_subset(target_indices, fraction)

    if not target_subset:
        return subset_percentage_str, 0

    results_file = os.path.join(RESULTS_DIR, f'results{subset_percentage_str}.csv')
    with open(results_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Target', 'Repository', 'Subset', 'Similarity', 'Result', 'Index1', 'Index2', 'Ground Truth', 'Runtime'])

        for target_idx in target_subset:
            start_time = time.time()
            similarity, result, idx1, idx2, ground_truth = run_clone_search(
                target_idx, repository_indices, model, TEST_SPLIT_DIR, samples_dict
            )
            end_time = time.time()
            runtime = end_time - start_time

            if similarity is not None:
                writer.writerow([version1, version2, subset_percentage_str, similarity, result, idx1, idx2, ground_truth, runtime])

    return subset_percentage_str, len(target_subset)


def main():
    print("Loading model...")
    model = BaseSiameseNetwork(input_size=200, num_layers=12).to(torch.device('cpu'))
    model.load_state_dict(torch.load(PARAMS_PATH))

    fractions = [0.5, 0.8, 1.0]

    for fraction in fractions:
        subset_percentage_str, num_targets = clone_search("o", "O0", "O3", model, fraction)
        print(f"Clone search completed for subset {subset_percentage_str} with {num_targets} targets.")


if __name__ == "__main__":
    main()
