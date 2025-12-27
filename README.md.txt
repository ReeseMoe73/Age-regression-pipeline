# Age Regression from Images (0–100) — Training Pipeline + FastAPI + Docker

This project predicts a person’s **age (0–100)** from a face image using a fine-tuned **ResNet18** model (PyTorch). It includes an end-to-end workflow:
- Create labels from image filenames
- Create reproducible train/val/test splits
- Train + evaluate with **MAE (years)**
- Deploy as:
  - a local CLI tool
  - a FastAPI endpoint (image upload)
  - a Docker container

---

## Results (example run)
- **Final Test MAE:** ~**5.63 years**

> Your number may vary depending on hardware, dataset composition, and random split seed.

---

## Repository layout
Recommended layout:

