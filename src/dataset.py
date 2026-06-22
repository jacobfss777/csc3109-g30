"""PyTorch Dataset and DataLoader setup for aerial imagery classification."""

from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ImageNet normalisation (standard for transfer learning)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Class names (must match directory structure)
CLASS_NAMES = ['oil_well', 'solar_panel', 'transformer_station', 'wastewater_treatment_plant']
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}

# Transform definitions
TRANSFORM_VAL = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

TRANSFORM_TRAIN = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


class ImageDataset(Dataset):
    """Load aerial images from a directory with class subdirectories."""

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []

        # Collect all image paths
        for cls_dir in sorted(self.root_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            class_idx = CLASS_TO_IDX.get(cls_dir.name)
            if class_idx is None:
                continue

            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}:
                    self.samples.append((img_path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def get_data_loaders(data_dir, batch_size=32, num_workers=0):
    """Create train and validation DataLoaders.

    Args:
        data_dir (Path): path to directory containing 'train' and 'val' subdirs
        batch_size (int): images per batch
        num_workers (int): parallel processes for loading (0 on Windows)

    Returns:
        (train_loader, val_loader, class_to_idx, idx_to_class)
    """
    data_dir = Path(data_dir)
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'

    train_dataset = ImageDataset(train_dir, transform=TRANSFORM_TRAIN)
    val_dataset = ImageDataset(val_dir, transform=TRANSFORM_VAL)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, CLASS_TO_IDX, IDX_TO_CLASS
