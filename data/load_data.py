"""
Loads the SHP data and writes data/dataset_stats.md (see that file for the data format).

Each pair is a post (history), two comments (human_ref_A, human_ref_B), and
labels = 1 if comment A got more upvotes, else 0. The persuasion test set (load_winning_args)
has the same columns, with labels = 1 if comment A earned a delta.

# Note: if running this file prints "You are sending unauthenticated requests to the HF Hub",
# run `hf auth login` once and enter the code given.
"""
from collections import Counter
from functools import cache
import html
import json
from pathlib import Path
import re
import time
import urllib.request
import zipfile

START_TIME = time.perf_counter()  # times how long it takes to load data

import pandas as pd

DATASET_NAME = "stanfordnlp/SHP"
CACHE_DIR = Path(__file__).with_name("cache")  # local copy of each subreddit, not in git
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
# How many times longer the longer comment is, in buckets. balance_length evens out each bucket.
LENGTH_RATIO_BINS = [1, 1.25, 1.5, 2, 3, 5, float("inf")]
LENGTH_RATIO_LABELS = ["1-1.25x", "1.25-1.5x", "1.5-2x", "2-3x", "3-5x", "5x+"]

# The persuasion test set: Winning Arguments (Tan et al., 2016), from ConvoKit
WINNING_ARGS_URL = ("https://zissou.infosci.cornell.edu/convokit/datasets/winning-args-corpus/"
                    "winning-args-corpus.zip")
WINNING_ARGS_PATH = CACHE_DIR / "winning_args.parquet"  # its pairs, extracted once from the zip
DELTA_MENTION = "∆|delta"  # a reply that mentions the delta gives the label away

# Text that is not part of the discussion: the moderator footer on posts and moderator bot comments
MOD_FOOTERS = ["Hello, users of CMV! This is a footnote from your moderators",
               "This is a footnote from the CMV moderators"]  # how the two footers start
# How bot notes start: the moderators' note, and DeltaBot's "/u/name (OP) has awarded 2 deltas"
BOT_COMMENTS = ["your thread has not been removed", "(op) has awarded"]

PAD, UNK = "<PAD>", "<UNK>"  # padding for the BiLSTM, and the stand-in for words not in the vocabulary
TEXT_COLUMNS = ["history", "human_ref_A", "human_ref_B"]

MODEL_COLUMNS = ["subreddit", "post_id", "history", "human_ref_A", "human_ref_B", "labels"]


# Load the data

SPLITS = ["train", "validation", "test"]

def load_subreddit(subreddit):
    """All pairs of one subreddit, unfiltered: {"train": df, "validation": df, "test": df}.
    The first call downloads it from Hugging Face and saves its splits in data/cache/<subreddit>/.
    We read this copy afterwards because load_dataset checks the Hub on every call, which is slow."""
    folder = CACHE_DIR / subreddit
    if not (folder / "test.parquet").exists():
        from datasets import load_dataset  # slow to import, and only needed the first time
        dataset = load_dataset(DATASET_NAME, data_dir=subreddit)
        folder.mkdir(parents=True, exist_ok=True)
        for split in SPLITS:  # test goes last, so if test.parquet exists the download finished
            dataset[split].to_pandas().to_parquet(folder / f"{split}.parquet")

    splits = {}
    for split in SPLITS:
        splits[split] = pd.read_parquet(folder / f"{split}.parquet")
        splits[split]["subreddit"] = subreddit
    return splits

def cut_footer(post):
    """The post without the moderator footer at its end."""
    for footer in MOD_FOOTERS:
        post = post.split(footer)[0]
    post = post.replace("&gt;", ">")  # ">" is sometimes stored as the HTML code "&gt;"
    return post.rstrip(" \n_>*")  # the line breaks, line and quote marks in front of the footer

def is_bot_comment(comment):
    start = comment[:80].lower().replace("*", "")
    return any(note in start for note in BOT_COMMENTS)

def clean_pairs(pairs):
    """Cuts the moderator footer off the posts and drops pairs 
    where a comment is a moderator bot note."""
    footerless = {post: cut_footer(post) for post in pairs["history"].unique()}  # a post is in many pairs
    pairs = pairs.assign(history=pairs["history"].map(footerless))
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

def load_training_data(subreddits=TRAIN_SUBREDDITS, filtered=True, columns_only=False):
    """Train and dev pairs of several subreddits, stacked into (train, dev)."""
    splits = [load_subreddit(s) for s in subreddits]
    train = pd.concat([prepare(s["train"], filtered, columns_only) for s in splits], ignore_index=True)
    dev = pd.concat([prepare(s["validation"], filtered, columns_only) for s in splits], ignore_index=True)
    return train, dev

def load_test_sets(subreddits, filtered=True, columns_only=False):
    """Test pairs per subreddit, kept separate so we can report accuracy per subreddit.
    TRAIN_SUBREDDITS gives the seen-topic tests, UNSEEN_SUBREDDITS the unseen-topic tests."""
    return {s: prepare(load_subreddit(s)["test"], filtered, columns_only) for s in subreddits}

def paired_replies(utterances):
    """The top-level replies that start a paired thread, one row per pair they are in.
    Deeper replies are left out: they include the post author's delta message."""
    for utterance in utterances:
        meta = utterance["meta"]
        if meta["success"] is None or utterance["reply-to"] != utterance["root"]:
            continue  # not in a pair, or not a top-level reply
        for pair_id in meta["pair_ids"]:
            yield {"pair_id": pair_id,
                   "post_id": utterance["root"].removeprefix("t3_"),
                   "delta": meta["success"],  # 1 if this thread earned the delta
                   "id": utterance["id"].removeprefix("t1_"),
                   "text": utterance["text"],
                   "score": meta["score"],
                   "time": utterance["timestamp"]}

def download_winning_args():
    """Downloads Winning Arguments and saves its pairs to WINNING_ARGS_PATH.

    In the corpus, each pair is two discussion threads under the same post, started by two
    different people: in one, the post's author awarded a delta, in the other they did not. We
    keep only the top-level reply that starts each thread, like the comments in SHP."""
    zip_path = CACHE_DIR / "winning-args-corpus.zip"
    CACHE_DIR.mkdir(exist_ok=True)
    urllib.request.urlretrieve(WINNING_ARGS_URL, zip_path)

    with zipfile.ZipFile(zip_path) as archive:
        posts = json.loads(archive.read("winning-args-corpus/conversations.json"))
        with archive.open("winning-args-corpus/utterances.jsonl") as lines:
            replies = pd.DataFrame(paired_replies(json.loads(line) for line in lines))

    # One row per pair: the delta reply's columns end in _delta, the other reply's in _other
    delta = replies[replies["delta"] == 1].set_index("pair_id")
    other = replies[replies["delta"] == 0].set_index("pair_id")
    pairs = delta.join(other, lsuffix="_delta", rsuffix="_other", how="inner").reset_index()
    post = lambda post_id: posts["t3_" + post_id]
    pairs["history"] = pairs["post_id_delta"].map(
        lambda post_id: post(post_id)["op-title"] + " " + post(post_id)["op-text-body"])
    pairs.to_parquet(WINNING_ARGS_PATH)
    zip_path.unlink()

def like_shp(texts):
    """The texts written the way SHP writes them: "&gt;" decoded to ">", and no line breaks."""
    return texts.map(lambda text: html.unescape(text).replace("\n", " "))

def side(pairs, column, delta_here):
    """This column for one comment: the delta reply's value where delta_here is True, else the
    other reply's."""
    return pairs[column + "_delta"].where(delta_here, pairs[column + "_other"])

def as_shp_pairs(pairs):
    """Winning Arguments pairs in SHP's columns. The delta reply is comment A in a random half
    of the pairs (the same half every run), so labels = 1 about half the time."""
    delta_is_a = pd.Series(pairs.index.isin(pairs.sample(frac=0.5, random_state=0).index))
    delta_is_b = ~delta_is_a
    return pd.DataFrame({
        "subreddit": "changemyview_delta",
        "post_id": pairs["post_id_delta"],
        "history": like_shp(pairs["history"]),
        "human_ref_A": like_shp(side(pairs, "text", delta_is_a)),
        "human_ref_B": like_shp(side(pairs, "text", delta_is_b)),
        "labels": delta_is_a.astype(int),
        "c_root_id_A": side(pairs, "id", delta_is_a),
        "c_root_id_B": side(pairs, "id", delta_is_b),
        "score_A": side(pairs, "score", delta_is_a),
        "score_B": side(pairs, "score", delta_is_b),
        "created_at_utc_A": side(pairs, "time", delta_is_a),
        "created_at_utc_B": side(pairs, "time", delta_is_b),
    })

def load_winning_args(filtered=True, columns_only=False):
    """The persuasion test set: two top-level replies to the same r/ChangeMyView post (2013-2015),
    where one reply's author went on to earn a delta. labels = 1 if it was comment A's author.
    We never train on it. The delta can be for a later reply in the same thread.

    filtered drops posts that are also in SHP, so no model has trained on them, and pairs where
    a reply mentions the delta."""
    if not WINNING_ARGS_PATH.exists():
        download_winning_args()
    pairs = as_shp_pairs(pd.read_parquet(WINNING_ARGS_PATH))

    if filtered:
        shp_posts = pd.concat(load_subreddit(MAIN_SUBREDDIT).values())["post_id"]  # every split
        mentions_delta = (pairs["human_ref_A"].str.contains(DELTA_MENTION, case=False)
                          | pairs["human_ref_B"].str.contains(DELTA_MENTION, case=False))
        keep = ~pairs["post_id"].isin(shp_posts) & ~mentions_delta
        pairs = clean_pairs(pairs[keep]).reset_index(drop=True)
    return pairs[MODEL_COLUMNS] if columns_only else pairs


# Shared by the models

@cache
def nltk_functions():
    """nltk's word_tokenize and mark_negation, imported on first use: importing nltk takes about
    1 s, and the data report does not need it. The punkt_tab data files are downloaded only if
    they are missing, because nltk.download checks the server every time it is called."""
    import nltk
    from nltk.sentiment.util import mark_negation
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    return nltk.word_tokenize, mark_negation


STRAIGHT_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})
# Found before tokenizing, because NLTK splits a link into pieces (https : //en.wikipedia.org/...).
# A link ends at a space or a closing bracket, and not on the punctuation of the sentence around it
URL = re.compile(r"(?:https?://|www\.)[^\s)\]]*[^\s)\].,!?;:'\"]")
# Punctuation that is less likely to carry meaning
FORMATTING = {"*", ",", ":", "$", "'", "`", "``", "''", "(", ")", "{", "}", "[", "]", "-", ";", "%"}

def tokenize(text, mark_negation=False):
    """Lowercased NLTK tokens of a text (words and punctuation marks). TF-IDF and the BiLSTM
    both use this, so they see the same tokens.

    Every link becomes <url> and every number <num>: the exact link or number is almost never
    repeated, but whether a comment cites one is a signal (see the README). This also stops the
    "?" in a link from counting as a question mark.

    mark_negation adds _NEG to the words after a negation until the next punctuation mark:
    "i do n't think it works ." -> i do n't think_NEG it_NEG works_NEG . For bag-of-words
    models such as TF-IDF, which cannot see word order; the BiLSTM reads the order itself."""
    word_tokenize, negate = nltk_functions()
    text = URL.sub(" urltoken ", text.translate(STRAIGHT_QUOTES))  # one word NLTK will not split
    tokens = word_tokenize(text.lower())
    # NLTK keeps each number as one token: 5, 1,200, 0.46, and list numbers like 2.
    tokens = ["<url>" if token == "urltoken"
              else "<num>" if token[:1].isdecimal() and token.replace(",", "").replace(".", "").isdecimal()
              else token
              for token in tokens]
    if mark_negation:
        tokens = negate(tokens)  # before punctuation is removed: . ? ! end the scope
    return [token for token in tokens if token.removesuffix("_NEG") not in FORMATTING]

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


# Data report

def words(texts):
    """Word count of each text. Faster than texts.str.split().str.len(), which builds the lists."""
    return texts.map(lambda text: len(text.split()))

def longer_won(pairs):
    """For each pair of unequal length: True if the longer comment got more upvotes."""
    a, b = words(pairs["human_ref_A"]), words(pairs["human_ref_B"])
    return ((a > b).astype(int) == pairs["labels"])[a != b]

def balance_length(pairs, seed=0):
    """Drops pairs until, in every length bucket (LENGTH_RATIO_BINS), the longer comment wins
    exactly half the time. Then picking the longer comment scores 50%, so a model can only do
    better by using what the comments say. Pairs of equal length are all kept. The pairs keep
    their index, so you can find them in the full set."""
    won = longer_won(pairs)  # pairs of unequal length only
    buckets = pd.cut(length_ratio(pairs)[won.index], LENGTH_RATIO_BINS, right=False)
    keep = [pairs.index.difference(won.index)]
    for _, in_bucket in won.groupby(buckets, observed=True):
        n = min(in_bucket.sum(), (~in_bucket).sum())  # as many longer-won as shorter-won pairs
        keep += [in_bucket[in_bucket == outcome].sample(n=n, random_state=seed).index
                 for outcome in (True, False)]
    return pairs.loc[keep[0].append(keep[1:]).sort_values()]

REPORT_PATH = Path(__file__).with_name("dataset_stats.md")
# The report's example post: comments of almost equal length, so the length baseline can't help
EXAMPLE_POST_ID = "zzutiu"

def percent(value):
    return f"{value:.1%}"

def markdown_table(header, rows):
    """A Markdown table with padded column to make the file easier to read as plain text."""
    rows = [[str(cell) for cell in row] for row in [header] + rows]
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    lines = ["| " + " | ".join(cell.ljust(width) for cell, width in zip(row, widths)) + " |"
             for row in rows]
    lines.insert(1, "|" + "|".join("-" * (width + 2) for width in widths) + "|")
    return "\n".join(lines)

def quote(text):
    """The text as a Markdown block quote."""
    return "\n".join("> " + line for line in text.strip().splitlines())

def example_section(pair):
    """Section 1: one pair to read, with the result shown after the texts so the reader can guess."""
    length_a, length_b = len(pair["human_ref_A"].split()), len(pair["human_ref_B"].split())
    winner, loser = ("A", "B") if pair["labels"] == 1 else ("B", "A")
    first = "A" if pair["created_at_utc_A"] <= pair["created_at_utc_B"] else "B"
    posted_a = pd.to_datetime(pair["created_at_utc_A"], unit="s")  # SHP has no post time
    post_url = f"https://www.reddit.com/r/{pair['subreddit']}/comments/{pair['post_id']}/"

    if max(length_a, length_b) <= 1.25 * min(length_a, length_b):
        length_note = ("The two comments are about the same length, so picking the longer one "
                       "tells us nothing here. A model has to read them.")
    else:
        longer = "A" if length_a > length_b else "B"
        length_note = (f"Comment {longer} is longer and "
                       f"{'won' if longer == winner else 'lost'}, "
                       f"so here picking the longer comment is {'right' if longer == winner else 'wrong'}.")

    return "\n\n".join([
        "Each pair is one post and two top-level comments on it. Read them and guess which "
        "comment got more upvotes. That is the task every model in this project does.",
        "**The post** (`history`)",
        quote(pair["history"]),
        f"**Comment A** (`human_ref_A`, {length_a} words, posted {posted_a:%B %-d, %Y})",
        quote(pair["human_ref_A"]),
        f"**Comment B** (`human_ref_B`, {length_b} words, "
        f"posted {pair['seconds_difference'] / 60:.0f} minutes "
        f"{'after' if first == 'A' else 'before'} comment A)",
        quote(pair["human_ref_B"]),
        "**The result.** "
        f"Comment {winner} got {pair['score_' + winner]} upvotes and comment {loser} got "
        f"{pair['score_' + loser]}, so {winner} won and labels = {pair['labels']}. "
        f"The score ratio is {pair['score_' + winner]} / {pair['score_' + loser]} = "
        f"{pair['score_ratio']:.1f}, well above the {MIN_SCORE_RATIO:g} that filter 1 asks for, "
        f"so the crowd clearly preferred {winner}. {length_note} {winner} was also posted later: "
        "SHP only keeps pairs where the winner came second, so a comment can't win just by "
        "being seen first.",
        "**What the model sees.** Only the three texts: `history`, `human_ref_A` and "
        "`human_ref_B`. It predicts `labels`. The upvote scores and comment times make the label "
        "and the filters, and the model never sees them. SHP lists the two comments in random "
        "order, so A wins about half the time and labels = 0 only means the winner is listed second.",
        f"**Check it on Reddit:** [the post]({post_url}), "
        f"[comment A]({post_url}comment/{pair['c_root_id_A']}/), "
        f"[comment B]({post_url}comment/{pair['c_root_id_B']}/). "
        "A comment can show as [deleted] now even though SHP kept its text, and its score may "
        "have changed since SHP collected it.",
    ])

REPORT_INTRO = (
    "# The data: SHP, r/ChangeMyView\n\n"
    "Generated by `python data/load_data.py`. Do not edit it by hand: rerun the script after "
    "changing the data code.\n\n"
    f"We train on r/{MAIN_SUBREDDIT} only. The other {len(UNSEEN_SUBREDDITS)} subreddits are never "
    "trained on and serve as unseen-topic test sets. To train on more subreddits, add them to "
    "EXTRA_TRAIN_SUBREDDITS and they move from the unseen tests to the seen tests.\n\n"
    "Each pair is a post (history), two comments on it (human_ref_A, human_ref_B), and "
    "labels = 1 if comment A got more upvotes, 0 if comment B did.\n\n"
    "Cleaning cuts the moderator footer off each post and drops pairs with a moderator bot "
    "comment. A pair's score ratio is the winner's upvotes divided by the loser's: 1 means a "
    "tie, and 10 means the winner got 10 times as many. When the ratio is close to 1, which "
    "comment won is close to chance, so the label is noisy. "
    f"Filter 1 keeps pairs with a score ratio of at least {MIN_SCORE_RATIO:g}. Filter 2 "
    f"keeps at most {MAX_PAIRS_PER_POST} random pairs per post. Everything below is after cleaning "
    "and both filters unless it says raw.\n\n"
    "Section 9 adds a persuasion test set, where the label is whether a reply changed the "
    "poster's view instead of which one got more upvotes.\n")

def data_report():
    """The data report in Markdown, computed from the current filters."""
    from tokenizers import Tokenizer

    splits = load_subreddit(MAIN_SUBREDDIT)  # load once, then filter the same copy
    raw = {"Train": splits["train"], "Dev": splits["validation"], "Test": splits["test"]}
    filtered = {name: filter_pairs(pairs) for name, pairs in raw.items()}
    train = filtered["Train"]
    sections = {}

    # The post's pair with the clearest winner
    example = train[train["post_id"] == EXAMPLE_POST_ID].nlargest(1, "score_ratio").iloc[0]
    sections["1. What does one pair look like?"] = example_section(example)

    rows = [[name, len(pairs), len(clean_pairs(pairs)), len(filter_pairs(pairs, max_per_post=None)),
             len(filtered[name]), filtered[name]["post_id"].nunique(),
             percent(filtered[name]["labels"].mean())]
            for name, pairs in raw.items()]
    sections["2. How many pairs do we have, and how many do the filters remove?"] = (
        "Each column applies one more step to the one before it. Cleaning drops pairs where a "
        "comment is a moderator bot note.\n\n" + markdown_table(
            ["Split", "Raw pairs", "After cleaning", f"+ score ratio >= {MIN_SCORE_RATIO:g}",
             f"+ at most {MAX_PAIRS_PER_POST} per post", "Posts", "A wins"], rows))

    # The same post or comment is in many pairs, so each text is counted once
    comments = words(pd.concat([train["human_ref_A"], train["human_ref_B"]]).drop_duplicates())
    posts = words(train["history"].drop_duplicates())
    rows = [[name, f"{n.mean():.0f}", f"{n.median():.0f}", f"{n.quantile(0.9):.0f}", n.max()]
            for name, n in [("Comment", comments), ("Post (title + body)", posts)]]
    sections["3. How long are the posts and comments? (train, in words)"] = (
        f"Each text counted once: {len(comments)} comments, {len(posts)} posts.\n\n"
        + markdown_table(["Text", "Mean", "Median", "90th percentile", "Max"], rows))

    # RoBERTa reads at most 512 tokens, so this tells us how often we need to cut the input
    tokenizer = Tokenizer.from_pretrained("roberta-base")
    sample = train.sample(n=min(2000, len(train)), random_state=0)
    inputs = {"Comment A + comment B": sample["human_ref_A"] + " " + sample["human_ref_B"],
              "Post + comment A + comment B":
                  sample["history"] + " " + sample["human_ref_A"] + " " + sample["human_ref_B"]}
    rows = []
    for name, texts in inputs.items():
        # One batch, because tokenizing one by one is slow
        encodings = tokenizer.encode_batch(list(texts), add_special_tokens=False)
        lengths = pd.Series([len(encoding.ids) for encoding in encodings])
        rows.append([name, f"{lengths.median():.0f}", percent((lengths > 512).mean())])
    sections["4. Does our input fit in the 512 tokens RoBERTa can read? (train)"] = (
        f"A random sample of {len(sample)} train pairs.\n\n"
        + markdown_table(["Input", "Median tokens", "Over 512 tokens"], rows))

    majority = int(train["labels"].mean() >= 0.5)  # the label a model that ignores the text would pick
    rows = [[name, percent((pairs["labels"] == majority).mean()),
             percent((pick_longer(pairs) == pairs["labels"]).mean())]
            for name, pairs in filtered.items()]
    sections["5. How often can we guess the winner without reading?"] = (
        "Every model has to beat these. The length baseline picks B when both comments have the "
        "same number of words.\n\n" + markdown_table(
            ["Split", f"Always guess {'A' if majority else 'B'} (train majority)",
             "Always pick the longer comment"], rows))

    won = longer_won(train)
    buckets = pd.cut(length_ratio(train)[won.index], LENGTH_RATIO_BINS, right=False,
                     labels=LENGTH_RATIO_LABELS)
    by_bucket = won.groupby(buckets, observed=False).agg(["size", "mean"])
    rows = [[label, int(row["size"]), percent(row["mean"])] for label, row in by_bucket.iterrows()]
    sections["6. Does picking the longer comment work better when it is a lot longer? (train)"] = (
        "Pairs where both comments have the same number of words are left out.\n\n"
        + markdown_table(["Longer comment is", "Pairs", "Longer comment wins"], rows))

    tests = load_test_sets(UNSEEN_SUBREDDITS)
    rows = [[name, len(pairs), percent(pairs["labels"].mean()),
             percent((pick_longer(pairs) == pairs["labels"]).mean())]
            for name, pairs in tests.items()]
    everything = pd.concat(tests.values())
    rows.append(["**Total**", len(everything), percent(everything["labels"].mean()),
                 percent((pick_longer(everything) == everything["labels"]).mean())])
    sections["7. How big are the unseen test sets?"] = (
        "Test splits of the other subreddits, after the same cleaning and filters. We never train on them.\n\n"
        + markdown_table(["Subreddit", "Pairs", "A wins", "Always pick the longer comment"], rows))

    persuasion = load_winning_args()
    sets = {**filtered, "Winning Arguments": persuasion}
    rows = []
    for name, pairs in sets.items():
        balanced = balance_length(pairs)
        rows.append([name, len(pairs), len(balanced), percent(len(balanced) / len(pairs)),
                     percent((pick_longer(balanced) == balanced["labels"]).mean())])
    sections["8. How many pairs are left when length cannot help?"] = (
        "Picking the longer comment is right about 70% of the time, so a model can score well "
        "without reading. `balance_length` drops pairs until, in every length bucket of section 6, "
        "the longer comment wins exactly half the time. Training on these pairs takes length away "
        "as a signal, and accuracy on them shows how much a model gets from what the comments say. "
        "Pairs of equal length are all kept, and the length baseline picks B for them, so it can be "
        "a little off 50%.\n\n" + markdown_table(
            ["Set", "Pairs", "Length-balanced pairs", "Kept", "Always pick the longer comment"], rows))

    raw_persuasion = load_winning_args(filtered=False)
    shp_posts = set(pd.concat(splits.values())["post_id"])
    new_posts = raw_persuasion[~raw_persuasion["post_id"].isin(shp_posts)]
    delta_more = persuasion["score_A"].where(persuasion["labels"] == 1, persuasion["score_B"])
    other_more = persuasion["score_B"].where(persuasion["labels"] == 1, persuasion["score_A"])
    rows = [["Pairs in the corpus", len(raw_persuasion)],
            ["Without posts that are also in SHP", len(new_posts)],
            ["Without replies that mention the delta, and cleaned (final)", len(persuasion)],
            ["Posts", persuasion["post_id"].nunique()],
            ["A wins (A earned the delta)", percent(persuasion["labels"].mean())],
            ["The delta reply is longer", percent((pick_longer(persuasion) == persuasion["labels"]).mean())],
            ["The delta reply got more upvotes", percent((delta_more > other_more).mean())],
            ["Both got the same upvotes", percent((delta_more == other_more).mean())]]
    sections["9. The persuasion test set: does the model pick the reply that changed a view?"] = (
        "Upvotes show what readers liked, not whether a comment changed the poster's mind. On "
        "r/ChangeMyView, the poster awards a delta (∆) to a comment that did. Winning Arguments "
        "(Tan et al., 2016) pairs two top-level replies to the same post from 2013 to 2015: the "
        "author of one went on to earn a delta, the other did not. labels = 1 if comment A's author "
        "earned it. We never train on it. It tests whether a model trained on upvotes picks the "
        "reply that persuaded. `load_winning_args` downloads it the first time.\n\n"
        "Posts that are also in SHP are dropped, so no model has trained on them. The delta can be "
        "for a later reply in the same thread, not the top-level reply itself.\n\n"
        + markdown_table(["", "Value"], rows))
    return REPORT_INTRO + "".join(f"\n## {title}\n\n{body}\n" for title, body in sections.items())

if __name__ == "__main__":
    REPORT_PATH.write_text(data_report())
    print(f"Report written to {REPORT_PATH}")
    print(f"Done in {time.perf_counter() - START_TIME:.1f} s")

