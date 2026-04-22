from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score


def _filter_valid(probs: np.ndarray, labels: np.ndarray, mask: np.ndarray):
    labels = np.asarray(labels)
    mask = np.asarray(mask)
    valid = (mask > 0) & (labels >= 0)
    probs = np.asarray(probs)[valid]
    labels = labels[valid]
    preds = probs.argmax(axis=-1) if probs.size else np.asarray([], dtype=np.int64)
    return preds, labels


def compute_metrics(probs: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    preds, labels = _filter_valid(probs, labels, mask)
    if labels.size == 0:
        return {'accuracy': 0.0, 'weighted_f1': 0.0, 'macro_f1': 0.0}
    return {
        'accuracy': float(accuracy_score(labels, preds) * 100.0),
        'weighted_f1': float(f1_score(labels, preds, average='weighted', zero_division=0) * 100.0),
        'macro_f1': float(f1_score(labels, preds, average='macro', zero_division=0) * 100.0),
    }


def build_classification_report(probs: np.ndarray, labels: np.ndarray, mask: np.ndarray, class_names: list[str] | None = None) -> str:
    preds, labels = _filter_valid(probs, labels, mask)
    if labels.size == 0:
        return 'No valid labeled samples found.'
    unique_labels = sorted(np.unique(np.concatenate([labels, preds])))
    if class_names is not None and len(class_names) > 0:
        target_names = [class_names[i] if i < len(class_names) else str(i) for i in unique_labels]
    else:
        target_names = [str(i) for i in unique_labels]
    return classification_report(labels, preds, labels=unique_labels, target_names=target_names, digits=4, zero_division=0)
