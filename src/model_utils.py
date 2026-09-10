import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, average_precision_score,
    roc_curve
)

def ensure_dirs():
    from pathlib import Path
    Path("models").mkdir(exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)

def low_fpr_recall(y_true, scores, target_fprs=(0.001, 0.005, 0.01, 0.02)):
    fpr, tpr, _ = roc_curve(y_true, scores)
    out = {}
    for target in target_fprs:
        valid = np.where(fpr <= target)[0]
        out[f"recall_at_fpr_{target:g}"] = float(tpr[valid[-1]]) if len(valid) else 0.0
    return out

def threshold_from_validation(y_true, scores):
    """Choose threshold on validation data only by maximising F1."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    order = np.argsort(scores)
    candidates = scores[order]
    best_t, best_f1 = float(candidates[0]), -1.0
    # Evaluate unique score thresholds. Dataset is small enough for this.
    for t in np.unique(candidates):
        pred = (scores >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = float(f1), float(t)
    return best_t

def evaluate_scores(y_true, scores, threshold):
    pred = (np.asarray(scores) >= threshold).astype(int)
    result = {
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "threshold": float(threshold),
    }
    result.update(low_fpr_recall(y_true, scores))
    return result
