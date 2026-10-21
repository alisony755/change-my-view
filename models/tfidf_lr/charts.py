"""
Evaluation charts for TF-IDF + LR on dev, with the settings in train.py: ROC curves and AUC next
to the length baseline, on all dev pairs and on the length-balanced dev pairs, plus the confusion
matrix, precision, recall and F1. 
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write the image file without opening a window
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO))
sys.path.append(str(Path(__file__).resolve().parent))
from data.load_data import balance_length, load_training_data  # noqa: E402
from results.metrics.accuracy import confusion_report, length_scores  # noqa: E402
from results.visualizations.roc import auc, plot_roc  # noqa: E402
from train import LogisticRegression, Tfidf, pair_features  # noqa: E402

CHART_PATH = REPO / "results" / "visualizations" / "tfidf_lr_dev.png"


if __name__ == "__main__":
    train, dev = load_training_data(columns_only=True)
    tfidf = Tfidf()
    tfidf.fit(list(train["human_ref_A"]) + list(train["human_ref_B"]))
    model = LogisticRegression()
    model.fit(pair_features(tfidf, train), train["labels"])

    labels = dev["labels"].to_numpy()
    model_scores = model.predict_probability(pair_features(tfidf, dev))
    baseline_scores = length_scores(dev)
    balanced = dev.index.get_indexer(balance_length(dev).index)

    # Check the hand-written ROC against scikit-learn's
    from sklearn.metrics import roc_auc_score
    assert abs(auc(labels, model_scores) - roc_auc_score(labels, model_scores)) < 1e-9
    assert abs(auc(labels, baseline_scores) - roc_auc_score(labels, baseline_scores)) < 1e-9

    confusion_report(f"TF-IDF + LR, dev ({len(dev)} pairs)", labels, (model_scores >= 0.5).astype(int))
    confusion_report("Length baseline (pick the longer comment), dev", labels,
                     (baseline_scores > 0).astype(int))
    confusion_report(f"TF-IDF + LR, length-balanced dev ({len(balanced)} pairs)", labels[balanced],
                     (model_scores[balanced] >= 0.5).astype(int))

    figure, axes = plt.subplots(1, 2, figsize=(12, 6))
    plot_roc(axes[0], [("TF-IDF + LR", labels, model_scores),
                       ("Length baseline", labels, baseline_scores)],
             f"All dev pairs ({len(dev)})")
    plot_roc(axes[1], [("TF-IDF + LR", labels[balanced], model_scores[balanced]),
                       ("Length baseline", labels[balanced], baseline_scores[balanced])],
             f"Length-balanced dev pairs ({len(balanced)}): length cannot help")
    figure.suptitle("TF-IDF + LR on r/ChangeMyView dev: which comment got more upvotes?")
    figure.tight_layout()
    figure.savefig(CHART_PATH, dpi=150, bbox_inches="tight")
    print(f"\nAUC: model {auc(labels, model_scores):.3f}, length baseline {auc(labels, baseline_scores):.3f}"
          f" | length-balanced: model {auc(labels[balanced], model_scores[balanced]):.3f}, "
          f"length baseline {auc(labels[balanced], baseline_scores[balanced]):.3f}")
