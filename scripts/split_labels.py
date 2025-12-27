from pathlib import Path
import csv
import random
from collections import defaultdict

BASE_DIR = Path(r"C:\Users\might\OneDrive\Documents\AI_ML Projects\age-regression")
LABELS_CSV = BASE_DIR / "labels.csv"

TRAIN_CSV = BASE_DIR / "labels_train.csv"
VAL_CSV   = BASE_DIR / "labels_val.csv"
TEST_CSV  = BASE_DIR / "labels_test.csv"

random.seed(42)

# 1) Read labels.csv
rows = []
with open(LABELS_CSV, "r", newline="") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append((r["filename"], int(float(r["age"]))))

# 2) Bucket by age decade to keep distribution reasonable
buckets = defaultdict(list)
for fname, age in rows:
    bucket = min(age // 10, 9)  # 0-9 -> 0, 90-100 -> 9
    buckets[bucket].append((fname, age))

train, val, test = [], [], []

# 3) Split each bucket 80/10/10
for bucket_rows in buckets.values():
    random.shuffle(bucket_rows)
    n = len(bucket_rows)
    n_train = int(n * 0.80)
    n_val = int(n * 0.10)

    train.extend(bucket_rows[:n_train])
    val.extend(bucket_rows[n_train:n_train + n_val])
    test.extend(bucket_rows[n_train + n_val:])

# 4) Shuffle final splits
random.shuffle(train)
random.shuffle(val)
random.shuffle(test)

def write_csv(path, data):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "age"])
        w.writerows(data)

write_csv(TRAIN_CSV, train)
write_csv(VAL_CSV, val)
write_csv(TEST_CSV, test)

print("✅ Done splitting!")
print("Train:", len(train))
print("Val:  ", len(val))
print("Test: ", len(test))
print("Saved to:", BASE_DIR)
