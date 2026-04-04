from args import get_args
import os
import re
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


class EarlyStopping:
    def __init__(self, patience=7, min_delta=1e-3):
        self.patience = patience
        self.min_delta = min_delta
        self.best_val = float('inf')
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss):
        if val_loss < self.best_val - self.min_delta:
            self.best_val = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def get_new_run_dir(base_dir: str) -> str:
    os.makedirs(base_dir, exist_ok=True)
    existing = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]

    run_nums = []
    for name in existing:
        m = re.match(r"run_(\d+)$", name)
        if m:
            run_nums.append(int(m.group(1)))

    next_num = (max(run_nums) + 1) if run_nums else 1
    run_name = f"run_{next_num:03d}"
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def train_model(model, train_loader, val_loader, device):
    args = get_args()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best_val_loss = float('inf')
    best_epoch = None

    run_dir = get_new_run_dir(args.plot_out_dir)

    config = vars(args).copy()  # argparse.Namespace -> dict
    config["run_dir"] = run_dir
    config["start_time"] = datetime.now().isoformat(timespec="seconds")

    config_path = os.path.join(run_dir, "config.txt")
    with open(config_path, "w") as f:
        for k, v in config.items():
            f.write(f"{k}: {v}\n")

    train_losses = []
    val_losses = []

    early_stopper = EarlyStopping(patience=7, min_delta=1e-3)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for images, targets in train_loader:
            images = [image.to(device=device, dtype=torch.float32) for image in images]
            targets = [
                {
                    'boxes': target['boxes'].to(device=device, dtype=torch.float32),
                    'labels': target['labels'].to(device=device, dtype=torch.int64)
                }
                for target in targets
            ]

            optimizer.zero_grad()

            loss_dict = model(images, targets)
            loss = sum(loss_value for loss_value in loss_dict.values())

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(images)

        train_epoch_loss = running_loss / len(train_loader.dataset)

        val_loss = validate_model(model, val_loader, device)

        train_losses.append(train_epoch_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {train_epoch_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            os.makedirs(args.out_dir, exist_ok=True)
            torch.save(
                model.state_dict(),
                os.path.join(args.out_dir, 'best_model.pth')
            )

        early_stopper.step(val_loss)
        if early_stopper.should_stop:
            print(
                f"Early stopping triggered at epoch {epoch + 1}. "
                f"Best val loss: {early_stopper.best_val:.4f}"
            )
            break

    # after training (stopped early or reached max epochs): plot and save
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, train_losses, label="Training loss")
    plt.plot(epochs, val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning curve: training vs validation loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "learning_curve.png"))

    loss_df = pd.DataFrame(
        {"epoch": list(epochs), "train_loss": train_losses, "val_loss": val_losses}
    )
    loss_df.to_csv(os.path.join(run_dir, "loss_history.csv"), index=False)

    with open(config_path, "a") as f:
        f.write("\n")
        f.write(f"best_val_loss: {best_val_loss:.6f}\n")
        if best_epoch is not None:
            f.write(f"best_epoch: {best_epoch}\n")


def validate_model(model, val_loader, device):
    model.train()
    val_loss_sum = 0.0
    val_count = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = [image.to(device=device, dtype=torch.float32) for image in images]
            targets = [
                {
                    'boxes': target['boxes'].to(device=device, dtype=torch.float32),
                    'labels': target['labels'].to(device=device, dtype=torch.int64)
                }
                for target in targets
            ]

            loss_dict = model(images, targets)
            loss = sum(loss_value for loss_value in loss_dict.values())

            val_loss_sum += loss.item() * len(images)
            val_count += len(images)

    val_epoch_loss = val_loss_sum / val_count
    return val_epoch_loss