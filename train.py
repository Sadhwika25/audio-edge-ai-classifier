"""
Training Driver for Audio Intelligence & Edge AI Classifier.
Trains AudioCNN or EdgeDSCNN with data augmentation, learning rate scheduling,
and automatic checkpointing.
"""

import argparse
import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.models import AudioCNN, EdgeDSCNN
from src.data import get_dataloaders, ACOUSTIC_CLASSES
from src.utils import calculate_metrics


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for specs, labels in loader:
        specs, labels = specs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(specs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * specs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, (correct / total) * 100.0


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for specs, labels in loader:
            specs, labels = specs.to(device), labels.to(device)
            outputs = model(specs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * specs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())

    metrics = calculate_metrics(all_targets, all_preds, num_classes=len(ACOUSTIC_CLASSES))
    val_loss = total_loss / len(all_targets)
    return val_loss, metrics["accuracy"] * 100.0, metrics


def main():
    parser = argparse.ArgumentParser(description="Train Audio Intelligence Classifier")
    parser.add_argument("--model", type=str, default="edge_dscnn", choices=["audio_cnn", "edge_dscnn"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-samples", type=int, default=1600)
    parser.add_argument("--val-samples", type=int, default=400)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}")
    print(f"[*] Architecture: {args.model.upper()}")

    # Dataloaders
    print(f"[*] Initializing acoustic datasets ({args.train_samples} train / {args.val_samples} val)...")
    train_loader, val_loader = get_dataloaders(
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        batch_size=args.batch_size,
    )

    # Initialize model
    num_classes = len(ACOUSTIC_CLASSES)
    if args.model == "audio_cnn":
        model = AudioCNN(num_classes=num_classes)
    else:
        model = EdgeDSCNN(num_classes=num_classes)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] Total Trainable Parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_acc = 0.0
    best_checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.model}_best.pt")

    start_time = time.time()
    print("-" * 65)
    print(f"{'Epoch':<8}{'Train Loss':<14}{'Train Acc (%)':<16}{'Val Loss':<12}{'Val Acc (%)':<12}")
    print("-" * 65)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"{epoch:<8}{train_loss:<14.4f}{train_acc:<16.2f}{val_loss:<12.4f}{val_acc:<12.2f}", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "model_type": args.model,
                    "num_classes": num_classes,
                },
                best_checkpoint_path,
            )

    elapsed = time.time() - start_time
    print("-" * 65, flush=True)
    print(f"[+] Training completed in {elapsed:.2f} seconds.", flush=True)
    print(f"[+] Best Validation Accuracy: {best_val_acc:.2f}%", flush=True)
    print(f"[+] Saved best model to: {best_checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
