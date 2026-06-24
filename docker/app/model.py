"""Load EfficientNet-B0 checkpoint and run inference."""

from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image

CLASS_NAMES = ['oil_well', 'solar_panel', 'transformer_station', 'wastewater_treatment_plant']

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def build_model(num_classes: int = 4) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def load_model(checkpoint_path: Path) -> nn.Module:
    model = build_model()
    state = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    return model


def predict(model: nn.Module, image: Image.Image) -> dict:
    tensor = TRANSFORM(image.convert('RGB')).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze()

    scores = {name: round(float(probs[i]), 6) for i, name in enumerate(CLASS_NAMES)}
    top_class = max(scores, key=scores.get)
    return {
        'predicted_class': top_class,
        'confidence': scores[top_class],
        'probabilities': scores,
    }
