"""
Evaluation metrics for Diabetic Retinopathy Detection.
Includes Quadratic Weighted Kappa (QWK), Sensitivity, Specificity,
Multi-class ROC-AUC, and Referable DR screening evaluation.
"""

from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Quadratic Weighted Kappa (QWK) between ground truth and predicted labels.

    QWK is the official benchmark metric for DR grading on APTOS & EyePACS.
    Penalizes misclassifications quadratically based on grade distance.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def calculate_clinical_metrics(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Calculate comprehensive clinical metrics for multi-class and referable DR screening.

    Args:
        y_true: 1D array of true integer class labels (0 to 4)
        y_pred_probs: 2D array of predicted class probabilities shape [N, 5]
        class_names: optional list of string class names

    Returns:
        dict containing multi-class metrics, confusion matrix, per-class sensitivity/specificity,
        and referable DR screening performance (Grades 0-1 vs Grades 2-4).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred_probs = np.asarray(y_pred_probs, dtype=float)
    y_pred = np.argmax(y_pred_probs, axis=1)

    n_classes = y_pred_probs.shape[1]
    if class_names is None:
        class_names = [f"Grade {i}" for i in range(n_classes)]

    # Standard Multi-class Metrics
    acc = float(accuracy_score(y_true, y_pred))
    qwk = compute_quadratic_weighted_kappa(y_true, y_pred)
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    # Per-class Sensitivity & Specificity (One-vs-Rest)
    per_class = {}
    total_samples = len(y_true)

    for i in range(n_classes):
        tp = float(cm[i, i])
        fn = float(cm[i, :].sum() - tp)
        fp = float(cm[:, i].sum() - tp)
        tn = float(total_samples - (tp + fn + fp))

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * (prec * sens) / (prec + sens) if (prec + sens) > 0 else 0.0

        per_class[class_names[i]] = {
            "class_id": i,
            "sensitivity_recall": round(sens, 4),
            "specificity": round(spec, 4),
            "precision": round(prec, 4),
            "f1_score": round(f1, 4),
            "support": int(cm[i, :].sum()),
        }

    # Macro Sensitivity and Specificity
    macro_sens = float(np.mean([m["sensitivity_recall"] for m in per_class.values()]))
    macro_spec = float(np.mean([m["specificity"] for m in per_class.values()]))

    # Multi-class One-vs-Rest ROC-AUC
    try:
        # Check if all classes present in y_true
        if len(np.unique(y_true)) > 1:
            roc_auc_ovr = float(
                roc_auc_score(y_true, y_pred_probs, multi_class="ovr", average="macro")
            )
        else:
            roc_auc_ovr = 0.0
    except Exception:
        roc_auc_ovr = 0.0

    # -------------------------------------------------------------
    # Clinical Screening: Referable DR Evaluation
    # Non-referable: Grade 0 (No DR) & Grade 1 (Mild)
    # Referable: Grade 2 (Moderate), Grade 3 (Severe), Grade 4 (PDR)
    # -------------------------------------------------------------
    y_true_referable = (y_true >= 2).astype(int)
    y_pred_referable_prob = np.sum(y_pred_probs[:, 2:], axis=1)
    y_pred_referable = (y_pred_referable_prob >= 0.5).astype(int)

    ref_cm = confusion_matrix(y_true_referable, y_pred_referable, labels=[0, 1])
    ref_tp = float(ref_cm[1, 1])
    ref_fn = float(ref_cm[1, 0])
    ref_fp = float(ref_cm[0, 1])
    ref_tn = float(ref_cm[0, 0])

    ref_sensitivity = ref_tp / (ref_tp + ref_fn) if (ref_tp + ref_fn) > 0 else 0.0
    ref_specificity = ref_tn / (ref_tn + ref_fp) if (ref_tn + ref_fp) > 0 else 0.0
    ref_accuracy = (ref_tp + ref_tn) / len(y_true) if len(y_true) > 0 else 0.0

    try:
        ref_auc = float(roc_auc_score(y_true_referable, y_pred_referable_prob))
    except Exception:
        ref_auc = 0.0

    # Uncertainty: Entropy of predicted distributions
    # H = -sum(p * log(p))
    epsilon = 1e-10
    entropy = -np.sum(y_pred_probs * np.log(y_pred_probs + epsilon), axis=1)
    mean_uncertainty = float(np.mean(entropy))

    return {
        "accuracy": round(acc, 4),
        "quadratic_weighted_kappa": round(qwk, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "precision_macro": round(prec_macro, 4),
        "recall_sensitivity_macro": round(rec_macro, 4),
        "specificity_macro": round(macro_spec, 4),
        "roc_auc_ovr": round(roc_auc_ovr, 4),
        "mean_uncertainty_entropy": round(mean_uncertainty, 4),
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": per_class,
        "referable_dr_screening": {
            "referable_definition": "Grades 2, 3, 4 (Moderate, Severe, PDR)",
            "sensitivity": round(ref_sensitivity, 4),
            "specificity": round(ref_specificity, 4),
            "accuracy": round(ref_accuracy, 4),
            "roc_auc": round(ref_auc, 4),
            "confusion_matrix": ref_cm.tolist(),
        },
    }
