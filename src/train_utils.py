"""Shared training and evaluation utilities used by all 5 approaches."""

import time
import copy
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


CLASS_NAMES = ['oil_well', 'solar_panel', 'transformer_station', 'wastewater_treatment_plant']
MODELS_DIR = Path(__file__).resolve().parent.parent / 'models'
MODELS_DIR.mkdir(exist_ok=True)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one full pass over the training set. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    """Run inference on the validation set. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def train_model(model, train_loader, val_loader, optimizer, scheduler=None,
                num_epochs=30, save_name='model', device=None):
    """Full training loop with validation, checkpointing, and history tracking.

    Args:
        model: PyTorch model
        train_loader, val_loader: DataLoader objects
        optimizer: torch optimiser
        scheduler: optional learning rate scheduler
        num_epochs: total training epochs
        save_name: filename stem for saved checkpoint (e.g. 'custom_cnn')
        device: 'cuda' or 'cpu'

    Returns:
        dict with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print(f"Training on: {device}")
    print(f"Epochs: {num_epochs}  |  Save as: models/{save_name}_best.pth")
    print("-" * 60)

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if scheduler:
            # ReduceLROnPlateau needs the val_loss; others just call step()
            if hasattr(scheduler, 'step') and 'metrics' in scheduler.step.__code__.co_varnames:
                scheduler.step(val_loss)
            else:
                scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, MODELS_DIR / f'{save_name}_best.pth')
            saved = ' * saved'
        else:
            saved = ''

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} | "
              f"{elapsed:.1f}s{saved}")

    print("-" * 60)
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    # Restore best weights
    model.load_state_dict(best_weights)
    return history


def plot_training_curves(history, title='Training Curves', save_path=None):
    """Plot loss and accuracy curves for a training run."""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Loss
    ax1.plot(epochs, history['train_loss'], label='Train Loss', color='#4C72B0')
    ax1.plot(epochs, history['val_loss'],   label='Val Loss',   color='#DD8452')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss over Epochs')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history['train_acc'], label='Train Accuracy', color='#4C72B0')
    ax2.plot(epochs, history['val_acc'],   label='Val Accuracy',   color='#DD8452')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy over Epochs')
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f'Saved: {save_path}')
    plt.show()


def get_all_predictions(model, loader, device):
    """Run inference on an entire DataLoader. Returns (all_labels, all_preds)."""
    model.eval()
    all_labels, all_preds = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(labels, preds, title='Confusion Matrix', save_path=None):
    """Plot a labelled confusion matrix heatmap."""
    cm = confusion_matrix(labels, preds)
    short_names = ['oil_well', 'solar_panel', 'transformer', 'wastewater']

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=short_names, yticklabels=short_names, ax=ax)
    ax.set_xlabel('Predicted Label', fontweight='bold')
    ax.set_ylabel('True Label', fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f'Saved: {save_path}')
    plt.show()


def print_classification_report(labels, preds):
    """Print per-class precision, recall, F1-score."""
    short_names = ['oil_well', 'solar_panel', 'transformer', 'wastewater']
    print(classification_report(labels, preds, target_names=short_names, digits=4))
