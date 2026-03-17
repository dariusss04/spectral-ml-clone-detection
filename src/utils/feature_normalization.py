# Feature normalization for eigenvalues and edge counts.

import os
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("PCD_DATA_DIR", REPO_ROOT / "data"))
INPUT_DIR = Path(os.getenv("PCD_NORM_INPUT_DIR", DATA_DIR / "splits" / "test"))
OUTPUT_DIR = Path(os.getenv("PCD_NORM_OUTPUT_DIR", DATA_DIR / "splits" / "normalizedTest"))

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".pt"):
            input_path = os.path.join(INPUT_DIR, filename)
            output_path = os.path.join(OUTPUT_DIR, filename)

            data = torch.load(input_path)
            eigenvalues = data["eigenvalues"] 
            num_edges   = data["num_edges"]  

            eig_norm = eigenvalues.norm(p=2)
            if eig_norm > 0:
                data["eigenvalues"] = eigenvalues / eig_norm
            else:
                data["eigenvalues"] = eigenvalues

            edge_norm = num_edges.norm(p=2)
            if edge_norm > 0:
                data["num_edges"] = num_edges / edge_norm
            else:
                data["num_edges"] = num_edges

            torch.save(data, output_path)

            print(f"Normalized and saved: {filename}")

if __name__ == "__main__":
    main()
