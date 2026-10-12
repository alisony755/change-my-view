"""
Loads the SHP data. We train on r/ChangeMyView only, and the other 17 subreddits are
unseen-topic test sets.

Each pair is a post (history), two comments (human_ref_A, human_ref_B), and
labels = 1 if comment A got more upvotes, else 0.
"""
from collections import Counter

import nltk
import pandas as pd
from datasets import load_dataset

nltk.download("punkt_tab", quiet=True)  # data files for nltk.word_tokenize

DATASET_NAME = "stanfordnlp/SHP"
MAIN_SUBREDDIT = "changemyview"
SUBREDDITS = [
    "askacademia", "askanthropology", "askbaking", "askcarguys", "askculinary", "askdocs",
    "askengineers", "askhistorians", "askhr", "askphilosophy", "askphysics", "askscience",
    "asksciencefiction", "asksocialscience", "askvet", "changemyview", "explainlikeimfive",
    "legaladvice",
]
EXTRA_TRAIN_SUBREDDITS = []  # subreddits added here are trained on and leave the unseen tests
TRAIN_SUBREDDITS = [MAIN_SUBREDDIT] + EXTRA_TRAIN_SUBREDDITS
UNSEEN_SUBREDDITS = [s for s in SUBREDDITS if s not in TRAIN_SUBREDDITS]

MIN_SCORE_RATIO = 2.0   # winner needs 2x the loser's upvotes, otherwise the label is too noisy
MAX_PAIRS_PER_POST = 5  # so a few huge threads don't dominate the data

# Text that is not part of the discussion: the moderator footer on posts and moderator bot comments
MOD_FOOTERS = ["Hello, users of CMV! This is a footnote from your moderators",
               "This is a footnote from the CMV moderators"]  # how the two footers start
BOT_COMMENT = "your thread has not been removed"  # near the start of every bot note

PAD, UNK = "<PAD>", "<UNK>"  # padding for the BiLSTM, and the stand-in for words not in the vocabulary
TEXT_COLUMNS = ["history", "human_ref_A", "human_ref_B"]

MODEL_COLUMNS = ["subreddit", "post_id", "history", "human_ref_A", "human_ref_B", "labels"]


# Load the data

def load_subreddit(subreddit):
    """All pairs of one subreddit, unfiltered, as {"train", "validation", "test"} -> DataFrame."""
    dataset = load_dataset(DATASET_NAME, data_dir=subreddit)
    return {split: dataset[split].to_pandas().assign(subreddit=subreddit) for split in dataset}

def cut_footer(post):
    """The post without the moderator footer at its end."""
    for footer in MOD_FOOTERS:
        post = post.split(footer)[0]
    return post.rstrip(" _>*")  # the line and quote marks in front of the footer

def is_bot_comment(comment):
    return BOT_COMMENT in comment[:80].lower().replace("*", "")

def clean_pairs(pairs):
    """Cuts the moderator footer off the posts and drops pairs where a comment is a moderator bot note."""
    pairs = pairs.assign(history=pairs["history"].map(cut_footer))
    is_bot = pairs["human_ref_A"].map(is_bot_comment) | pairs["human_ref_B"].map(is_bot_comment)
    return pairs[~is_bot]

def filter_pairs(pairs, max_per_post=MAX_PAIRS_PER_POST):
    """Cleans the pairs, then filter 1: score ratio >= MIN_SCORE_RATIO, and filter 2: at most
    max_per_post random pairs per post."""
    kept = clean_pairs(pairs)
    kept = kept[kept["score_ratio"] >= MIN_SCORE_RATIO]
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


# Shared by the models

def tokenize(text):
    """Lowercased NLTK tokens of a text (words and punctuation marks). TF-IDF and the BiLSTM
    both use this, so they see the same tokens."""
    return nltk.word_tokenize(text.lower())

def build_vocabulary(train, min_count=2, columns=TEXT_COLUMNS):
    """Maps each word seen at least min_count times in the training texts to an index.
    Index 0 is PAD and index 1 is UNK. Build it from the training pairs only."""
    counts = Counter()
    for column in columns:
        for text in train[column].drop_duplicates():  # the same post or comment is in many pairs
            counts.update(tokenize(text))
    kept = sorted(word for word, count in counts.items() if count >= min_count)
    return {word: index for index, word in enumerate([PAD, UNK] + kept)}

def encode(text, vocabulary):
    """The text as a list of word indexes, with UNK for words outside the vocabulary."""
    return [vocabulary.get(word, vocabulary[UNK]) for word in tokenize(text)]

def pick_longer(pairs):
    """The length baseline: predicts 1 if comment A has more words than comment B, else 0."""
    return (words(pairs["human_ref_A"]) > words(pairs["human_ref_B"])).astype(int)

def length_ratio(pairs):
    """How many times longer the longer comment is (1.0 = same length). Use it to report a
    model's accuracy on pairs of similar length, where the length shortcut does not help."""
    lengths = pd.concat([words(pairs["human_ref_A"]), words(pairs["human_ref_B"])], axis=1)
    return lengths.max(axis=1) / lengths.min(axis=1).clip(lower=1)


# Data Report

def words(texts):
    return texts.str.split().str.len()

def longer_won(pairs):
    """For each pair of unequal length: True if the longer comment got more upvotes."""
    a, b = words(pairs["human_ref_A"]), words(pairs["human_ref_B"])
    return ((a > b).astype(int) == pairs["labels"])[a != b]

if __name__ == "__main__":
    from transformers import AutoTokenizer  # slow import, only the report needs it

    raw = load_cmv(filtered=False)
    train, dev, test = load_cmv()

    # Sample counts and class distribution. "Longer wins" is the shortcut any model has to beat.
    print("Pairs per split")
    for name, before, pairs in zip(["train", "dev", "test"], raw, [train, dev, test]):
        print(f"  {name}: {len(before)} raw, {len(pairs)} after filters, "
              f"A wins {pairs['labels'].mean():.1%}, longer comment wins {longer_won(pairs).mean():.1%}")

    print("\nLength in words (train)")
    comments = pd.concat([words(train["human_ref_A"]), words(train["human_ref_B"])])
    posts = words(train.drop_duplicates("post_id")["history"])  # count each post once
    for name, n in [("comment", comments), ("post", posts)]:
        print(f"  {name}: mean {n.mean():.0f}, median {n.median():.0f}, max {n.max()}")

    # RoBERTa can read 512 tokens, so this tells us how often we need to cut the input
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    text = train["history"] + " " + train["human_ref_A"] + " " + train["human_ref_B"]
    tokens = text.map(lambda t: len(tokenizer(t, add_special_tokens=False)["input_ids"]))
    print(f"\nPost + both comments over 512 RoBERTa tokens (train): {(tokens > 512).mean():.1%}")
