"""
Length independence: is a model right because it reads the comments, or only because it picks
the longer one? Its accuracy in each length bucket, next to the length baseline, shared by all
models. A model that only follows length tracks the baseline in every bucket.
"""
import numpy as np
import pandas as pd

from data.load_data import LENGTH_RATIO_BINS, LENGTH_RATIO_LABELS, length_ratio, pick_longer
from results.metrics.accuracy import accuracy


def accuracy_by_length_ratio(predictions, pairs):
    """A table with one row per length bucket (how many times longer the longer comment is):
    the number of pairs, the model's accuracy and the length baseline's accuracy. Pairs of equal
    length are in the first bucket."""
    predictions = np.asarray(predictions)
    labels = pairs["labels"].to_numpy()
    longer = pick_longer(pairs).to_numpy()
    buckets = pd.cut(length_ratio(pairs), LENGTH_RATIO_BINS, right=False,
                     labels=LENGTH_RATIO_LABELS).to_numpy()
    rows = []
    for bucket in LENGTH_RATIO_LABELS:
        here = buckets == bucket
        rows.append({"Longer comment is": bucket, "Pairs": int(here.sum()),
                     "Accuracy": accuracy(predictions[here], labels[here]) if here.any() else np.nan,
                     "Length baseline": accuracy(longer[here], labels[here]) if here.any() else np.nan})
    return pd.DataFrame(rows)
