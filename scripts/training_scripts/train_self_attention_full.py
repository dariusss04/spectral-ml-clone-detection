# Train self-attention model using eigenvalues and edge counts.

import os
import sys
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from utils.dataset import SiameseDataset
from models.deepset_self_attention import DeepSetSelfAttention

DATA_DIR = Path(os.getenv("PCD_DATA_DIR", REPO_ROOT / "data"))
SAMPLES_FILE = Path(os.getenv("PCD_SAMPLES_FILE", DATA_DIR / "meta" / "programs_metadata.txt"))
TRAIN_SPLIT_DIR = Path(os.getenv("PCD_TRAIN_SPLIT_DIR", DATA_DIR / "splits" / "train"))
VAL_SPLIT_DIR = Path(os.getenv("PCD_VAL_SPLIT_DIR", DATA_DIR / "splits" / "validation"))

EXPERIMENTS_DIR = Path(os.getenv("PCD_EXPERIMENTS_DIR", REPO_ROOT / "experiments"))
EXPERIMENT_DIR = EXPERIMENTS_DIR / "self_attention_full"
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

NUM_EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 1e-5

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

MODEL_SAVE_PATH = EXPERIMENT_DIR / "weights.pth"
METRICS_FILE = EXPERIMENT_DIR / "metrics.csv"
SUMMARY_FILE = EXPERIMENT_DIR / "summary.csv"


def train_and_evaluate() -> None:
    model = DeepSetSelfAttention(
        input_dim=1,
        hidden_dim=64,
        output_dim=64,
        phi_layers=4,
        rho_layers=4,
    ).to(DEVICE)

    criterion = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    train_dataset = SiameseDataset(split_folder=str(TRAIN_SPLIT_DIR), samples_file_path=str(SAMPLES_FILE))
    val_dataset = SiameseDataset(split_folder=str(VAL_SPLIT_DIR), samples_file_path=str(SAMPLES_FILE))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    best_val_loss = float("inf")
    best_epoch = 0

    with open(METRICS_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["epoch", "avg_train_loss", "avg_val_loss"])

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        batch_count = 0

        for pair, labels in train_loader:
            features1, features2 = pair
            labels = labels.float().to(DEVICE)

            eig1 = features1["eigenvalues"].unsqueeze(-1).to(DEVICE)
            edges1 = features1["num_edges"].unsqueeze(-1).to(DEVICE)
            eig2 = features2["eigenvalues"].unsqueeze(-1).to(DEVICE)
            edges2 = features2["num_edges"].unsqueeze(-1).to(DEVICE)

            optimizer.zero_grad()
            out1, out2 = model(eig1, edges1, eig2, edges2)
            out1 = F.normalize(out1, dim=-1)
            out2 = F.normalize(out2, dim=-1)

            loss = criterion(out1, out2, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_count += 1

        avg_train_loss = running_loss / batch_count

        model.eval()
        val_loss = 0.0
        val_batch_count = 0
        with torch.no_grad():
            for pair, labels in val_loader:
                features1, features2 = pair
                labels = labels.float().to(DEVICE)

                eig1 = features1["eigenvalues"].unsqueeze(-1).to(DEVICE)
                edges1 = features1["num_edges"].unsqueeze(-1).to(DEVICE)
                eig2 = features2["eigenvalues"].unsqueeze(-1).to(DEVICE)
                edges2 = features2["num_edges"].unsqueeze(-1).to(DEVICE)

                val_out1, val_out2 = model(eig1, edges1, eig2, edges2)
                val_out1 = F.normalize(val_out1, dim=-1)
                val_out2 = F.normalize(val_out2, dim=-1)

                loss = criterion(val_out1, val_out2, labels)
                val_loss += loss.item()
                val_batch_count += 1

        avg_val_loss = val_loss / val_batch_count

        with open(METRICS_FILE, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch + 1, avg_train_loss, avg_val_loss])

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

        scheduler.step()

    with open(SUMMARY_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["best_epoch", "best_val_loss", "num_epochs", "batch_size", "learning_rate"])
        writer.writerow([best_epoch, best_val_loss, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE])


if __name__ == "__main__":
    train_and_evaluate()
