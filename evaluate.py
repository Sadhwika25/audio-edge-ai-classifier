"""
Evaluation Script for Trained Audio Intelligence Models.
Loads checkpoint, runs on validation/test sets, and prints confusion matrix and detailed metrics.
"""

import argparse
import os
import torch

from src.models import AudioCNN, EdgeDSCNN
from src.data import get_dataloaders, ACOUSTIC_CLASSES
from src.utils import calculate_metrics, compute_confusion_matrix, print_ascii_confusion_matrix


def main():
    parser = argparse.ArgumentParser(description="Evaluate Audio Model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/edge_dscnn_best.pt")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}. Please run train.py first.")

    device = torch.device("cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_type = checkpoint.get("model_type", "edge_dscnn")
    num_classes = checkpoint.get("num_classes", len(ACOUSTIC_CLASSES))

    print(f"[*] Loading {model_type} from {args.checkpoint}...")
    if model_type == "audio_cnn":
        model = AudioCNN(num_classes=num_classes)
    else:
        model = EdgeDSCNN(num_classes=num_classes)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _, test_loader = get_dataloaders(
        train_samples=100,
        val_samples=args.samples,
        batch_size=args.batch_size,
    )

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for specs, labels in test_loader:
            outputs = model(specs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy().tolist())
            all_targets.extend(labels.numpy().tolist())

    metrics = calculate_metrics(all_targets, all_preds, num_classes=num_classes)

    print("\n" + "=" * 55)
    print("           MODEL EVALUATION SUMMARY REPORT")
    print("=" * 55)
    print(f"  Accuracy (Overall):      {metrics['accuracy'] * 100.0:.2f}%")
    print(f"  Macro-Precision:         {metrics['precision_macro'] * 100.0:.2f}%")
    print(f"  Macro-Recall:            {metrics['recall_macro'] * 100.0:.2f}%")
    print(f"  Macro-F1 Score:          {metrics['f1_macro'] * 100.0:.2f}%")
    print("=" * 55)

    cm = compute_confusion_matrix(all_targets, all_preds, num_classes=num_classes)
    print("\nConfusion Matrix:")
    print_ascii_confusion_matrix(cm, ACOUSTIC_CLASSES)


if __name__ == "__main__":
    main()
