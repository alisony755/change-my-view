# change-my-view

**What makes a persuasive argument on Reddit?** Given a post on
r/ChangeMyView and two comments on it, we predict which comment got more upvotes. We compare
three kinds of models (TF-IDF + logistic regression, a BiLSTM and fine-tuned RoBERTa) on
accuracy, generalization to other topics, length independence, interpretability, cost and
context sensitivity.

Team: Alison Ye, Kae Strange, Patrick Zhang, Hang Hang

## Data

[Stanford Human Preferences (SHP)](https://huggingface.co/datasets/stanfordnlp/SHP), downloaded
automatically by the code (nothing to download by hand). We train on r/ChangeMyView only; the
other 17 subreddits are unseen-topic test sets. `data/load_data.py` cleans and filters the pairs:
a score ratio of at least 2, and at most 5 pairs per post. SHP's dev split is small, so 10% of the
train posts move to dev.

Upvotes are not persuasion, so we also test on
[Winning Arguments](https://convokit.cornell.edu/documentation/winning.html) (Tan et al., 2016):
pairs of replies to the same r/ChangeMyView post where one earned a delta (it changed the poster's
view). And since the longer comment wins about 70% of the time, `balance_length` keeps only pairs
where it wins half the time, to test on what the comments say. See
`data/dataset_stats.md`.

## Setup

Python 3.12. From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python data/report.py            # data statistics, written to data/dataset_stats.md
python models/tfidf_lr/train.py  # TF-IDF + logistic regression
```

## Layout

```
data/load_data.py    loading, cleaning and filtering; shared tokenizer and baselines
data/report.py       the data statistics in data/dataset_stats.md
models/              one folder per model: tfidf_lr, bilstm, roberta
results/             evaluation code and plots
```


