# Clone search using DeepSets with eigenvalues and edge counts.

import csv
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from models.deepset_siamese import SiameseNetworkDeepSets
from utils.dataset import parse_samples

DATA_DIR = Path(os.getenv("PCD_DATA_DIR", REPO_ROOT / "data"))
SAMPLES_FILE = Path(os.getenv("PCD_SAMPLES_FILE", DATA_DIR / "meta" / "programs_metadata.txt"))
CONFIG_FILE = Path(os.getenv("PCD_CLONE_SEARCH_CONFIG", DATA_DIR / "meta" / "clone_search_config.json"))
SPLITS_DIR = Path(os.getenv("PCD_SPLITS_DIR", DATA_DIR / "splits"))
TEST_SPLIT_DIR = Path(os.getenv("PCD_TEST_SPLIT_DIR", SPLITS_DIR / "test"))

EXPERIMENTS_DIR = Path(os.getenv("PCD_EXPERIMENTS_DIR", REPO_ROOT / "experiments"))
WEIGHTS_PATH = Path(os.getenv("PCD_WEIGHTS_PATH", EXPERIMENTS_DIR / "deepset_full" / "weights.pth"))

RESULTS_DIR = Path(os.getenv("PCD_RESULTS_DIR", REPO_ROOT / "results" / "clone_search"))
RESULTS_FILE = RESULTS_DIR / "deepset_full.csv"

SIM_THRESHOLD = float(os.getenv("PCD_SIM_THRESHOLD", "0.5"))


def is_clone(index_a, index_b, index_to_name):
    project_a, binary_a = index_to_name[index_a]
    project_b, binary_b = index_to_name[index_b]
    return project_a == project_b and binary_a == binary_b


def load_set_features(path):
    data = torch.load(path, map_location="cpu")
    eigen = data["eigenvalues"].unsqueeze(0).unsqueeze(-1)
    edges = data["num_edges"].unsqueeze(0).unsqueeze(-1)
    return eigen, edges


def run_clone_search(target_idx, repository_indices, model, device):
    target_path = TEST_SPLIT_DIR / f"program{target_idx}.pt"
    if not target_path.is_file():
        return None, None

    eig_a, edge_a = load_set_features(target_path)
    eig_a = eig_a.to(device)
    edge_a = edge_a.to(device)

    best_similarity = -1.0
    best_match_idx = None

    with torch.no_grad():
        for repo_idx in repository_indices:
            repo_path = TEST_SPLIT_DIR / f"program{repo_idx}.pt"
            if not repo_path.is_file():
                continue

            eig_b, edge_b = load_set_features(repo_path)
            eig_b = eig_b.to(device)
            edge_b = edge_b.to(device)

            out_a, out_b = model(eig_a, edge_a, eig_b, edge_b)
            out_a = F.normalize(out_a, dim=-1)
            out_b = F.normalize(out_b, dim=-1)
            similarity = F.cosine_similarity(out_a, out_b, dim=-1).item()

            if similarity > best_similarity:
                best_similarity = similarity
                best_match_idx = repo_idx

    return best_similarity, best_match_idx


def clone_search_for_config(config_data, model, device, index_to_name, writer):
    search_type = config_data["search_type"]
    version1 = config_data["version1"]
    version2 = config_data["version2"]
    targets = config_data["target"]
    repositories = config_data["repository"]

    for target_idx in targets:
        similarity, best_match_idx = run_clone_search(target_idx, repositories, model, device)
        if best_match_idx is None or similarity is None:
            continue

        ground_truth = is_clone(target_idx, best_match_idx, index_to_name)
        result = 1 if (ground_truth and similarity > SIM_THRESHOLD) else 0
        writer.writerow([search_type, version1, version2, result, int(ground_truth)])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseNetworkDeepSets(input_dim=1, hidden_dim=64, output_dim=64, phi_layers=5, rho_layers=5)
    state = torch.load(WEIGHTS_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()

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
            clone_search_for_config(config_data, model, device, index_to_name, writer)


if __name__ == "__main__":
    main()
