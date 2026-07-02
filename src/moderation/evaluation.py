"""Metricas de evaluacion: precision/recall/F1 con intervalo de Wilson y
reponderacion de la precision a la prevalencia real (problema de base rate).
"""
import numpy as np

from .config import REAL_PREVALENCE


def wilson_ci(k, n, z=1.96):
    """Intervalo de confianza de Wilson para una proporcion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (round(float(center - half), 4), round(float(center + half), 4))


def precision_at_prevalence(recall, specificity, prevalence=REAL_PREVALENCE):
    """Precision esperada a una prevalencia dada
    """
    tpr, fpr = recall, 1 - specificity
    den = prevalence * tpr + (1 - prevalence) * fpr
    return round(prevalence * tpr / den, 4) if den else float("nan")


def evaluate(y_true, y_pred):
    """Matriz de confusion + precision/recall/F1 con IC y precision reponderada."""
    yt = np.asarray([bool(x) for x in y_true])
    yp = np.asarray([bool(x) for x in y_pred])
    tp = int((yt & yp).sum())
    fp = int((~yt & yp).sum())
    fn = int((yt & ~yp).sum())
    tn = int((~yt & ~yp).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    f1 = (2 * prec * rec / (prec + rec)) if (tp + fp and tp + fn and prec + rec) else float("nan")
    return {
        "n": len(yt), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(prec, 4), "precision_ci": wilson_ci(tp, tp + fp),
        "recall": round(rec, 4), "recall_ci": wilson_ci(tp, tp + fn),
        "specificity": round(spec, 4), "f1": round(f1, 4),
        "precision_real_prev": precision_at_prevalence(rec, spec),
    }
