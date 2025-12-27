from pathlib import Path
import csv

IMAGE_DIR = Path(r"C:\Users\might\OneDrive\Documents\AI_ML Projects\age-regression\data\raw")
OUTPUT_CSV = Path(r"C:\Users\might\OneDrive\Documents\AI_ML Projects\age-regression\labels.csv")

# Safety: ensure the output folder exists
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# Quick sanity check
jpgs = list(IMAGE_DIR.glob("*.jpg"))
print("Looking in:", IMAGE_DIR)
print("Found .jpg files:", len(jpgs))

rows = []
for img_path in jpgs:
    try:
        age = int(img_path.stem.split("_")[0])  # age_gender_race_date.jpg
        if 0 <= age <= 100:
            rows.append((img_path.name, age))
    except Exception:
        continue

rows.sort()

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "age"])
    writer.writerows(rows)

print(f"✅ Wrote {len(rows)} rows to: {OUTPUT_CSV}")
