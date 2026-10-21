"""
Accuracy next to the baselines, shared by all models. The functions take a model's predictions
(1 = comment A won) and the pairs, so every model is measured the same way.
"""
import numpy as np

from data.load_data import balance_length, length_ratio, pick_longer


def accuracy(predictions, labels):
    return float(np.mean(np.asarray(predictions) == np.asarray(labels)))

def margin(accuracy_value, pair_count):
    """Half of the 95% confidence interval for an accuracy measured on pair_count pairs: the true
    accuracy is probably within plus or minus this. Gaps smaller than it are noise."""
    return 1.96 * np.sqrt(accuracy_value * (1 - accuracy_value) / pair_count)

def report(name, predictions, labels, baseline):
    """Prints one accuracy with its margin and number of pairs, next to the length baseline's
    accuracy on the same pairs."""
    if len(labels) == 0:
        print(f"{name}: no pairs")
        return
    model_accuracy = accuracy(predictions, labels)
    print(f"{name}: {model_accuracy:.4f} ± {margin(model_accuracy, len(labels)):.3f} "
          f"on {len(labels)} pairs (length baseline {accuracy(baseline, labels):.4f})")

def evaluate(name, predictions, pairs, majority_label):
    """Prints the model's accuracy on all pairs and on the pairs where length cannot help, each
    next to the length baseline on the same pairs, and the majority baseline: always guessing
    the label that is more common in train (majority_label)."""
    predictions = np.asarray(predictions)
    labels = pairs["labels"].to_numpy()
    longer = pick_longer(pairs).to_numpy()
    # Similar length: the longer comment is less than 1.25x the shorter one
    similar = (length_ratio(pairs) < 1.25).to_numpy()
    # Length-balanced: in every length bucket, the longer comment wins half the time
    balanced = pairs.index.get_indexer(balance_length(pairs).index)

    print(f"\n{name}")
    report("Accuracy", predictions, labels, longer)
    report("Similar-length accuracy", predictions[similar], labels[similar], longer[similar])
    report("Length-balanced accuracy", predictions[balanced], labels[balanced], longer[balanced])
    majority = accuracy(np.full(len(labels), majority_label), labels)
    print(f"Majority baseline (always guess {'A' if majority_label else 'B'}): {majority:.4f}")

def length_scores(pairs):
    """The length baseline as a score, for ROC curves: log(words in A / words in B). Above 0
    means A is longer."""
    lengths_a = pairs["human_ref_A"].str.split().str.len().clip(lower=1)
    lengths_b = pairs["human_ref_B"].str.split().str.len().clip(lower=1)
    return np.log(lengths_a / lengths_b).to_numpy()

def confusion_report(name, labels, predictions):
    """The confusion matrix and precision, recall and F1, with "A won" as the positive class."""
    true_positive = int(np.sum((predictions == 1) & (labels == 1)))
    false_positive = int(np.sum((predictions == 1) & (labels == 0)))
    false_negative = int(np.sum((predictions == 0) & (labels == 1)))
    true_negative = int(np.sum((predictions == 0) & (labels == 0)))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    print(f"\n{name}")
    print("                 A won   B won")
    print(f"    said A won   {true_positive:5d}   {false_positive:5d}")
    print(f"    said B won   {false_negative:5d}   {true_negative:5d}")
    print(f"    Accuracy {np.mean(predictions == labels):.3f}   Precision {precision:.3f}   "
          f"Recall {recall:.3f}   F1 {f1:.3f}")
