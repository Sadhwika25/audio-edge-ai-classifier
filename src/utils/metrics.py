"""
Model Evaluation Metrics & Confusion Matrix calculation.
"""

from typing import Dict, List, Tuple
import numpy as np
import torch


def calculate_metrics(y_true: List[int], y_pred: List[int], num_classes: int) -> Dict[str, float]:
    """
    Computes overall Accuracy, Macro-Precision, Macro-Recall, and Macro-F1.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    acc = float(np.mean(y_true == y_pred))
    
    precisions = []
    recalls = []
    f1s = []
    
    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        
    return {
        "accuracy": acc,
        "precision_macro": float(np.mean(precisions)),
        "recall_macro": float(np.mean(recalls)),
        "f1_macro": float(np.mean(f1s)),
    }


def compute_confusion_matrix(y_true: List[int], y_pred: List[int], num_classes: int) -> np.ndarray:
    """
    Computes confusion matrix where cm[i, j] is count of true class i predicted as j.
    """
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def print_ascii_confusion_matrix(cm: np.ndarray, class_names: List[str]):
    """
    Prints a clean formatted confusion matrix in terminal.
    """
    header = f"{'True \\ Pred':<15}" + "".join([f"{name[:6]:>8}" for name in class_names])
    print(header)
    print("-" * len(header))
    for idx, row in enumerate(cm):
        row_str = f"{class_names[idx][:14]:<15}" + "".join([f"{val:>8}" for val in row])
        print(row_str)
