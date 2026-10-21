"""
ROC curves and AUC, shared by all models. Sort the pairs by the model's score, lower the 
threshold one pair at a time, and record the true positive (TP) rate and false positive (FP) rate at 
each threshold.

Here "positive" means labels = 1 (comment A won). A score is anything where higher means "more
likely A won": a probability from a model, or log(words in A / words in B) for the length baseline.
"""
import numpy as np

def roc_points(labels, scores):
    """(FP rates, TP rates), from the highest threshold to the lowest.
    Pairs with the same score are passed together, so ties make one diagonal step."""
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="stable")
    labels, scores = labels[order], scores[order]
    # The last index of each run of equal scores: the threshold moves past the whole run at once
    last_of_run = np.r_[np.flatnonzero(np.diff(scores)), len(scores) - 1]
    true_positives = np.cumsum(labels)[last_of_run]
    false_positives = (last_of_run + 1) - true_positives
    true_positive_rate = np.r_[0, true_positives / labels.sum()]
    false_positive_rate = np.r_[0, false_positives / (len(labels) - labels.sum())]
    return false_positive_rate, true_positive_rate

def auc(labels, scores):
    """Area under the ROC curve: 0.5 is random guessing, 1.0 ranks every A-win above every B-win."""
    false_positive_rate, true_positive_rate = roc_points(labels, scores)
    return float(np.trapezoid(true_positive_rate, false_positive_rate))

def plot_roc(axis, curves, title):
    """Draws one ROC curve per (name, labels, scores) in curves on a matplotlib axis."""
    for name, labels, scores in curves:
        false_positive_rate, true_positive_rate = roc_points(labels, scores)
        axis.plot(false_positive_rate, true_positive_rate,
                  label=f"{name} (AUC {auc(labels, scores):.3f})")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guessing (AUC 0.500)")
    axis.set_xlabel("False positive rate (B won, model said A)")
    axis.set_ylabel("True positive rate (A won, model said A)")
    axis.set_title(title)
    axis.legend(loc="lower right", fontsize=8)
    axis.set_aspect("equal")
