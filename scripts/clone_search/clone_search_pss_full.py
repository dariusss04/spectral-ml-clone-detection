# Clone search using PSS on eigenvalues and edge counts.

import csv
import json
import math
import os
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("PCD_DATA_DIR", REPO_ROOT / "data"))
SAMPLES_FILE = Path(os.getenv("PCD_SAMPLES_FILE", DATA_DIR / "meta" / "programs_metadata.txt"))
CONFIG_FILE = Path(os.getenv("PCD_CLONE_SEARCH_CONFIG", DATA_DIR / "meta" / "clone_search_config.json"))
SPLITS_DIR = Path(os.getenv("PCD_SPLITS_DIR", DATA_DIR / "splits"))
NORMALIZED_TEST_DIR = Path(os.getenv("PCD_NORMALIZED_TEST_SPLIT_DIR", SPLITS_DIR / "normalizedTest"))

RESULTS_DIR = Path(os.getenv("PCD_RESULTS_DIR", REPO_ROOT / "results" / "clone_search"))
RESULTS_FILE = RESULTS_DIR / "pss_full.csv"

SIM_THRESHOLD = float(os.getenv("PCD_SIM_THRESHOLD", "0.8"))


def parse_samples(samples_file_path):
    index_to_name = {}
    with open(samples_file_path, "r") as f:
        lines = f.readlines()
    for i in range(0, len(lines), 2):
        index = int(lines[i].strip())
        program_info = lines[i + 1].strip()
        parts = program_info.split("_")
        project = parts[0].split("-")[0]
        binary = parts[-1]
        index_to_name[index] = (project, binary)
    return index_to_name


def is_clone(index_a, index_b, index_to_name):
    project_a, binary_a = index_to_name[index_a]
    project_b, binary_b = index_to_name[index_b]
    return project_a == project_b and binary_a == binary_b


def sim_cg(features_a, features_b):
    eigen_a = features_a["eigenvalues"]
    eigen_b = features_b["eigenvalues"]
    len_a = eigen_a.argmin()
    len_b = eigen_b.argmin()
    d = min(len_a, len_b)
    distance = (eigen_a[:d] - eigen_b[:d]).norm(p=2)
    return (math.sqrt(2) - distance.item()) / math.sqrt(2)


def sim_cfg(features_a, features_b):
    edges_a = features_a["num_edges"]
    edges_b = features_b["num_edges"]
    len_a = edges_a.argmin()
    len_b = edges_b.argmin()
    d = min(len_a, len_b)
    distance = (edges_a[:d] - edges_b[:d]).norm(p=2)
    return (math.sqrt(2) - distance.item()) / math.sqrt(2)


def pss_score(features_a, features_b):
    return (sim_cg(features_a, features_b) + sim_cfg(features_a, features_b)) / 2


def clone_search_for_config(config_data, index_to_name, writer):
    search_type = config_data["search_type"]
    version1 = config_data["version1"]
    version2 = config_data["version2"]
    targets = config_data["target"]
    repositories = config_data["repository"]

    for target_idx in targets:
        target_path = NORMALIZED_TEST_DIR / f"program{target_idx}.pt"
        if not target_path.is_file():
            continue
        target_features = torch.load(target_path)

        best_similarity = -1.0
        best_match_idx = None

        for repo_idx in repositories:
            repo_path = NORMALIZED_TEST_DIR / f"program{repo_idx}.pt"
            if not repo_path.is_file():
                continue
            repo_features = torch.load(repo_path)

            similarity = pss_score(target_features, repo_features)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_idx = repo_idx

        if best_match_idx is None:
            continue

        ground_truth = is_clone(target_idx, best_match_idx, index_to_name)
        result = 1 if (ground_truth and best_similarity > SIM_THRESHOLD) else 0
        writer.writerow([search_type, version1, version2, result, int(ground_truth)])


def main():
    index_to_name = parse_samples(SAMPLES_FILE)
    with open(CONFIG_FILE, "r") as f:
        clone_config = json.load(f)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_FILE.exists()
    with open(RESULTS_FILE, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(["type", "target", "repository", "result", "ground_truth"])
        for _, config_data in clone_config.items():
            clone_search_for_config(config_data, index_to_name, writer)


if __name__ == "__main__":
    main()
