"""
Loads the SHP data for r/ChangeMyView. We train on r/ChangeMyView only. 
The other 17 subreddits are never trained on and serve as unseen-topic test sets. 
To train on more subreddits, add them to EXTRA_TRAIN_SUBREDDITS and they move from 
the unseen tests to the seen tests.

Each pair is a post (history), two comments on it (human_ref_A, human_ref_B), and
labels = 1 if comment A got more upvotes, 0 if comment B did.
"""
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

DATASET_NAME = "stanfordnlp/SHP"
MAIN_SUBREDDIT = "changemyview"
SUBREDDITS = [
    "askacademia", "askanthropology", "askbaking", "askcarguys", "askculinary", "askdocs",
    "askengineers", "askhistorians", "askhr", "askphilosophy", "askphysics", "askscience",
    "asksciencefiction", "asksocialscience", "askvet", "changemyview", "explainlikeimfive",
    "legaladvice",
]
EXTRA_TRAIN_SUBREDDITS = []
TRAIN_SUBREDDITS = [MAIN_SUBREDDIT] + EXTRA_TRAIN_SUBREDDITS
UNSEEN_SUBREDDITS = [s for s in SUBREDDITS if s not in TRAIN_SUBREDDITS]

MIN_SCORE_RATIO = 2.0   # winner needs 2x the loser's upvotes, otherwise the label is too noisy
MAX_PAIRS_PER_POST = 5  # so a few huge threads don't dominate the data

MODEL_COLUMNS = ["subreddit", "post_id", "history", "human_ref_A", "human_ref_B", "labels"]


def load_subreddit(subreddit):
    """All pairs of one subreddit, unfiltered, as {"train", "validation", "test"} -> DataFrame."""
    dataset = load_dataset(DATASET_NAME, data_dir=subreddit)
    return {split: dataset[split].to_pandas().assign(subreddit=subreddit) for split in dataset}

def filter_pairs(pairs, max_per_post=MAX_PAIRS_PER_POST):
    """Filter 1: score ratio >= MIN_SCORE_RATIO. Filter 2: at most max_per_post random pairs per post."""
    kept = pairs[pairs["score_ratio"] >= MIN_SCORE_RATIO]
    if max_per_post is not None:
        kept = (kept.sample(frac=1.0, random_state=0)
                    .groupby("post_id", sort=False).head(max_per_post)
                    .sort_index())
    return kept.reset_index(drop=True)

def prepare(pairs, filtered, columns_only):
    if filtered:
        pairs = filter_pairs(pairs)
    return pairs[MODEL_COLUMNS] if columns_only else pairs

def load_cmv(filtered=True, columns_only=False):
    """(train, dev, test) for r/ChangeMyView. SHP splits by post, so no post is in two splits."""
    splits = load_subreddit(MAIN_SUBREDDIT)
    return tuple(prepare(splits[s], filtered, columns_only) for s in ("train", "validation", "test"))

def load_training_data(subreddits=TRAIN_SUBREDDITS, filtered=True, columns_only=False):
    """Train and dev pairs of several subreddits, stacked into (train, dev)."""
    splits = [load_subreddit(s) for s in subreddits]
    train = pd.concat([prepare(s["train"], filtered, columns_only) for s in splits], ignore_index=True)
    dev = pd.concat([prepare(s["validation"], filtered, columns_only) for s in splits], ignore_index=True)
    return train, dev

def load_test_sets(subreddits, filtered=True, columns_only=False):
    """Test pairs per subreddit, kept separate so we can report accuracy per subreddit."""
    return {s: prepare(load_subreddit(s)["test"], filtered, columns_only) for s in subreddits}

def load_seen_tests(filtered=True, columns_only=False):
    return load_test_sets(TRAIN_SUBREDDITS, filtered, columns_only)

def load_unseen_tests(filtered=True, columns_only=False):
    return load_test_sets(UNSEEN_SUBREDDITS, filtered, columns_only)


# Data Report

def words(texts):
    return texts.str.split().str.len()

def length_ratio(pairs):
    """How many times longer the longer comment is (1 = same length)."""
    a, b = words(pairs["human_ref_A"]), words(pairs["human_ref_B"])
    return np.maximum(a, b) / np.minimum(a, b).clip(lower=1)

def longer_won(pairs):
    """For each pair of unequal length: True if the longer comment got more upvotes."""
    a, b = words(pairs["human_ref_A"]), words(pairs["human_ref_B"])
    return ((a > b).astype(int) == pairs["labels"])[a != b]

if __name__ == "__main__":
    print(__doc__)
    print(f"Filters: score ratio >= {MIN_SCORE_RATIO:g}, at most {MAX_PAIRS_PER_POST} pairs per post.")
    raw = load_subreddit(MAIN_SUBREDDIT)
    train = filter_pairs(raw["train"])
    test = filter_pairs(raw["test"])

    print("\n1. Example pair (train)")
    for col in MODEL_COLUMNS:
        print(f"  {col}: {' '.join(str(train[col].iloc[0]).split())[:90]}")

    # Sample counts and class distribution, for the check-in Data section
    print("\n2. Pairs per split: raw / after filter 1 / after both filters, posts, A wins")
    for split in ["train", "validation", "test"]:
        pairs = raw[split]
        kept = filter_pairs(pairs)
        print(f"  {split}: {len(pairs)} / {len(filter_pairs(pairs, None))} / {len(kept)}, "
              f"{kept['post_id'].nunique()} posts, A wins {kept['labels'].mean():.1%}")

    print("\n3. Length in words (train)")
    comments = pd.concat([words(train["human_ref_A"]), words(train["human_ref_B"])])
    posts = words(train.drop_duplicates("post_id")["history"])  # count each post once
    for name, n in [("comment", comments), ("post", posts)]:
        print(f"  {name}: mean {n.mean():.0f}, median {n.median():.0f}, "
              f"90th percentile {n.quantile(0.9):.0f}, max {n.max()}")

    # RoBERTa can read 512 tokens, so this tells us how often we need to cut the input
    print("\n4. Length in RoBERTa tokens (train)")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    count = lambda texts: texts.map(lambda t: len(tokenizer(t, add_special_tokens=False)["input_ids"]))
    both_comments = count(train["human_ref_A"]) + count(train["human_ref_B"])
    with_post = both_comments + count(train["history"])
    for name, n in [("comment A + comment B", both_comments), ("post + both comments", with_post)]:
        print(f"  {name}: median {n.median():.0f}, over 512: {(n > 512).mean():.1%}")

    # Any model has to beat these to show it learned more than a shortcut
    print("\n5. Guessing without reading")
    for split in ["train", "validation", "test"]:
        kept = filter_pairs(raw[split])
        majority = max(kept["labels"].mean(), 1 - kept["labels"].mean())
        print(f"  {split}: majority label {majority:.1%}, always pick the longer comment "
              f"{longer_won(kept).mean():.1%}")

    # The accuracy vs. length ratio, and the similar-length pairs used to check length bias
    print("\n6. Longer comment win rate by length ratio (test)")
    ratio = length_ratio(test)[longer_won(test).index]
    groups = pd.cut(ratio, bins=[1, 1.25, 1.5, 2, 3, 5, np.inf], include_lowest=True,
                    labels=["1-1.25x", "1.25-1.5x", "1.5-2x", "2-3x", "3-5x", "5x+"])
    for group, won in longer_won(test).groupby(groups, observed=True):
        print(f"  {group}: {len(won)} pairs, longer wins {won.mean():.1%}")
    print(f"  similar-length pairs (ratio <= 1.25): {(length_ratio(test) <= 1.25).sum()} of {len(test)}")

    # Out-of-domain generalization sets
    print("\n7. Unseen subreddits (test)")
    print(f"  {'subreddit':<20}{'pairs':>8}{'A wins':>10}{'longer wins':>14}")
    total = 0
    for name, unseen in load_unseen_tests().items():
        total += len(unseen)
        print(f"  {name:<20}{len(unseen):>8}{unseen['labels'].mean():>10.1%}"
              f"{longer_won(unseen).mean():>14.1%}")
    print(f"  {'total':<20}{total:>8}")