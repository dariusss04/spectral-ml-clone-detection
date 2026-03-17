# Train baseline Siamese MLP models across a depth sweep.

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
from models.siamese_mlp_relu_base import BaseSiameseNetwork

DATA_DIR = Path(os.getenv("PCD_DATA_DIR", REPO_ROOT / "data"))
SAMPLES_FILE = Path(os.getenv("PCD_SAMPLES_FILE", DATA_DIR / "meta" / "programs_metadata.txt"))
TRAIN_SPLIT_DIR = Path(os.getenv("PCD_TRAIN_SPLIT_DIR", DATA_DIR / "splits" / "train"))
VAL_SPLIT_DIR = Path(os.getenv("PCD_VAL_SPLIT_DIR", DATA_DIR / "splits" / "validation"))

EXPERIMENTS_DIR = Path(os.getenv("PCD_EXPERIMENTS_DIR", REPO_ROOT / "experiments"))
EXPERIMENT_DIR = EXPERIMENTS_DIR / "mlp_depth_sweep"
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


def train_and_evaluate_model(num_layers: int) -> None:
    architecture_dir = EXPERIMENT_DIR / f"depth_{num_layers}l"
    architecture_dir.mkdir(parents=True, exist_ok=True)

    model_save_path = architecture_dir / "weights.pth"
    metrics_file = architecture_dir / "metrics.csv"
    summary_file = architecture_dir / "summary.csv"

    model = BaseSiameseNetwork(num_layers=num_layers, input_size=200).to(DEVICE)
    criterion = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    train_dataset = SiameseDataset(split_folder=str(TRAIN_SPLIT_DIR), samples_file_path=str(SAMPLES_FILE))
    val_dataset = SiameseDataset(split_folder=str(VAL_SPLIT_DIR), samples_file_path=str(SAMPLES_FILE))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    best_val_loss = float("inf")
    best_epoch = 0

    with open(metrics_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["epoch", "avg_train_loss", "avg_val_loss"])

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        batch_count = 0

        for pair, labels in train_loader:
            features1, features2 = pair
            labels = labels.float().to(DEVICE)

            x1 = torch.cat([features1["eigenvalues"], features1["num_edges"]], dim=1).to(DEVICE)
            x2 = torch.cat([features2["eigenvalues"], features2["num_edges"]], dim=1).to(DEVICE)

            optimizer.zero_grad()
            out1, out2 = model(x1, x2)
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

                x1 = torch.cat([features1["eigenvalues"], features1["num_edges"]], dim=1).to(DEVICE)
                x2 = torch.cat([features2["eigenvalues"], features2["num_edges"]], dim=1).to(DEVICE)

                out1, out2 = model(x1, x2)
                out1 = F.normalize(out1, dim=-1)
                out2 = F.normalize(out2, dim=-1)

                loss = criterion(out1, out2, labels)
                val_loss += loss.item()
                val_batch_count += 1

        avg_val_loss = val_loss / val_batch_count

        with open(metrics_file, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch + 1, avg_train_loss, avg_val_loss])

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_save_path)

        scheduler.step()

    with open(summary_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["best_epoch", "best_val_loss", "num_epochs", "batch_size", "learning_rate"])
        writer.writerow([best_epoch, best_val_loss, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE])


if __name__ == "__main__":
    for num_layers in range(2, 25):
        train_and_evaluate_model(num_layers)
