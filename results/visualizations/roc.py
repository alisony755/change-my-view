"""
ROC curves and AUC, shared by all models. Sort the pairs by the model's score, lower the 
threshold one pair at a time, and record the true positive (TP) rate and false positive (FP) rate at 
each threshold.

Here "positive" means labels = 1 (comment A won). A score is anything where higher means "more
likely A won": a probability from a model, or log(words in A / words in B) for the length baseline.
"""
import numpy as np

import matplotlib
matplotlib.use("Agg")  # write the image file without opening a window
import matplotlib.pyplot as plt

from data.load_data import balance_length
from results.metrics.accuracy import length_scores

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

"""
Draws and saves a model's ROC chart on the dev set, next to the length baseline. The ROC math
and plot_roc are in roc.py; this file builds the figure and saves it.
"""
def roc_chart(name, dev, probabilities, path):
    """Saves ROC curves of a model and the length baseline to path, on all dev pairs and on the
    length-balanced ones."""
    labels = dev["labels"].to_numpy()
    baseline_scores = np.asarray(length_scores(dev))
    balanced = dev.index.get_indexer(balance_length(dev).index)  # their positions in dev
    figure, axes = plt.subplots(1, 2, figsize=(12, 6))
    plot_roc(axes[0], [(name, labels, probabilities), ("Length baseline", labels, baseline_scores)],
             f"All dev pairs ({len(dev)})")
    plot_roc(axes[1], [(name, labels[balanced], probabilities[balanced]),
                       ("Length baseline", labels[balanced], baseline_scores[balanced])],
             f"Length-balanced dev pairs ({len(balanced)}): length cannot help")
    figure.suptitle(f"{name} on r/ChangeMyView dev: which comment got more upvotes?")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"\nLength-balanced ROC AUC: {auc(labels[balanced], probabilities[balanced]):.3f} "
          f"(length baseline {auc(labels[balanced], baseline_scores[balanced]):.3f})")
    print(f"ROC curves saved to {path}")
