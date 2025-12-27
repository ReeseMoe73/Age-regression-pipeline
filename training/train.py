import csv
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image


# -----------------------
# 1) Paths / Settings
# -----------------------
BASE_DIR = Path(r"C:\Users\might\OneDrive\Documents\AI_ML Projects\age-regression")
IMAGE_DIR = BASE_DIR / "data" / "raw"

TRAIN_CSV = BASE_DIR / "labels_train.csv"
VAL_CSV   = BASE_DIR / "labels_val.csv"
TEST_CSV  = BASE_DIR / "labels_test.csv"

OUT_MODEL = BASE_DIR / "best_model.pt"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
SEED = 42

# Windows/PyCharm: keep at 0 to avoid multiprocessing issues
NUM_WORKERS = 0


# -----------------------
# 2) Seed (reproducible runs)
# -----------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------
# 3) Dataset
# -----------------------
class AgeDataset(Dataset):
    """
    CSV must have columns: filename, age
    - Loads images from `image_dir/filename`
    - Scales age to 0..1 by dividing by 100.0
    """

    def __init__(self, csv_path: Path, image_dir: Path, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.items = []

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        if not image_dir.exists():
            raise FileNotFoundError(f"Image folder not found: {image_dir}")

        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"{csv_path} appears empty or has no header row.")

            expected = {"filename", "age"}
            if not expected.issubset(set(reader.fieldnames)):
                raise ValueError(
                    f"{csv_path} must have columns {expected}, found {set(reader.fieldnames)}"
                )

            for r in reader:
                fname = r["filename"].strip()
                try:
                    age = float(r["age"])
                except ValueError:
                    continue

                # Keep your stated valid range
                if 0 <= age <= 100:
                    self.items.append((fname, age))

        if len(self.items) == 0:
            raise ValueError(f"No valid rows found in {csv_path}. Check your CSV contents.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fname, age = self.items[idx]
        img_path = self.image_dir / fname

        if not img_path.exists():
            raise FileNotFoundError(f"Missing image file: {img_path}")

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # Scale age to 0..1
        y = torch.tensor([age / 100.0], dtype=torch.float32)
        return img, y


# -----------------------
# 4) Model
# -----------------------
class AgeRegressor(nn.Module):
    """
    ResNet18 backbone -> 1 output -> sigmoid keeps it in 0..1
    """

    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.backbone(x)
        x = self.sigmoid(x)  # output constrained to 0..1
        return x


# -----------------------
# 5) Metric (MAE in years)
# -----------------------
def mae_years(pred_scaled: torch.Tensor, y_scaled: torch.Tensor) -> torch.Tensor:
    """
    pred_scaled and y_scaled are in 0..1
    Convert to years by multiplying by 100, then compute MAE.
    """
    pred_years = pred_scaled * 100.0
    y_years = y_scaled * 100.0
    return torch.mean(torch.abs(pred_years - y_years))


# -----------------------
# 6) Train / Eval loops
# -----------------------
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    total_n = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_mae += mae_years(pred.detach(), y).item() * bs
        total_n += bs

    return total_loss / total_n, total_mae / total_n


@torch.no_grad()
def eval_one_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_n = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_mae += mae_years(pred, y).item() * bs
        total_n += bs

    return total_loss / total_n, total_mae / total_n


# -----------------------
# 7) Main
# -----------------------
def main():
    set_seed(SEED)

    print("✅ Starting training script...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Basic sanity checks before training begins
    print("BASE_DIR:", BASE_DIR)
    print("IMAGE_DIR:", IMAGE_DIR)
    print("TRAIN_CSV:", TRAIN_CSV)
    print("VAL_CSV:  ", VAL_CSV)
    print("TEST_CSV: ", TEST_CSV)

    # Transforms
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Data
    train_ds = AgeDataset(TRAIN_CSV, IMAGE_DIR, transform=train_tf)
    val_ds   = AgeDataset(VAL_CSV,   IMAGE_DIR, transform=eval_tf)
    test_ds  = AgeDataset(TEST_CSV,  IMAGE_DIR, transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print("Train samples:", len(train_ds))
    print("Val samples:  ", len(val_ds))
    print("Test samples: ", len(test_ds))

    # Model / Loss / Optimizer
    model = AgeRegressor().to(device)

    # SmoothL1 (Huber) is stable for regression
    loss_fn = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_mae = float("inf")

    # Train
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_mae = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_mae = eval_one_epoch(model, val_loader, loss_fn, device)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_MAE={train_mae:.2f}y | "
            f"val_loss={val_loss:.4f} val_MAE={val_mae:.2f}y"
        )

        # Save best model based on validation MAE
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), OUT_MODEL)
            print(f"  ✅ Saved best model -> {OUT_MODEL} (val_MAE={best_val_mae:.2f}y)")

    # Final Test
    if OUT_MODEL.exists():
        model.load_state_dict(torch.load(OUT_MODEL, map_location=device))
        test_loss, test_mae = eval_one_epoch(model, test_loader, loss_fn, device)
        print(f"\nFinal Test | loss={test_loss:.4f} | MAE={test_mae:.2f} years")
    else:
        print("⚠️ No saved model found. (Validation MAE never improved?)")

    print("Done.")


if __name__ == "__main__":
    main()
