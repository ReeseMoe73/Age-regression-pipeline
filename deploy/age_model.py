from pathlib import Path
from typing import Union, Tuple

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# Image sizes for all images 224
IMG_SIZE = 224

# Preprocessing must match what your model expects (same normalization)
PREPROCESS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class AgeRegressor(nn.Module):
    """
     This must match training architecture:
    - ResNet18 backbone
    - fc changed to 1 output
    - sigmoid to constrain 0..1 (then we scale to 0..100 years)
    """
    def __init__(self):
        super().__init__()
        # Important: weights=None prevents downloading ImageNet weights during deployment
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 1)
        self.m = m
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.m(x))


def load_model(model_path: Union[str, Path], device: str = None) -> Tuple[nn.Module, str]:
    """
    Loads best_model.pt and returns (model, device).
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AgeRegressor().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, device


@torch.no_grad()
def predict_age(model: nn.Module, device: str, image: Image.Image) -> float:
    """
    Input: PIL Image
    Output: predicted age in years (0..100)
    """
    x = PREPROCESS(image.convert("RGB")).unsqueeze(0).to(device)  # [1,3,224,224]
    pred_01 = model(x).item()     # 0..1
    return float(pred_01 * 100.0) # 0..100 years
