# How to run

To run the training script, you can run the train.py file in this directory.

There are a few arguments that you can pass to the script:

```
--train: trains the model on the training data
--test: evaluates the model on the dev + test data
--resume-training: resumes training from a checkpoint (taken after each epoch)
--quick: runs a quick training just for testing purposes
```

So the command to train and test the entire model would be:

```
python data/models/roberta/train.py --train --test
```

# Architecture

The model is a comment scorer: it reads a post and one comment and outputs a single number,
how good that comment is as a reply to that post. To compare two comments, we score each one
separately and predict that comment A wins if score(A) > score(B).

```
post + comment A --> RoBERTa --> score(A) -+
                                           +--> score(A) - score(B) --> sigmoid --> P(A wins)
post + comment B --> RoBERTa --> score(B) -+
                  (same weights)
```

## Input

Each (post, comment) pair is encoded as one sequence using the `AutoTokenizer` from roBERTa.

RoBERTa reads at most 512 tokens. When the post and comment together are longer than that, the
tokens are cut from whichever of the two is longer (`truncation="longest_first"`), so a long post
doesn't crowd out the comment. Setting `USE_POST = False` encodes the comment alone, for the
context-sensitivity experiment.

## Training setup

| Setting                | Value                                                             |
| ---------------------- | ----------------------------------------------------------------- |
| Optimizer              | AdamW, learning rate 2e-5                                         |
| Learning rate schedule | linear warmup over the first 10% of steps, then linear decay to 0 |
| Batch size             | 8 pairs (16 sequences through the model)                          |
| Epochs                 | 2                                                                 |
| Gradient clipping      | max norm 1.0                                                      |
| Model selection        | the epoch with the best dev accuracy is kept                      |

After every epoch, the full training state is saved to `resume_checkpoint/`, so an interrupted
run can be continued with `--resume-training`. The best epoch is saved to `checkpoint/` when
training finishes. When running `--test` the model will load from there. 

# Testing

Running with `--test` loads the saved model from `checkpoint/` (or uses the freshly trained one
when combined with `--train`) and evaluates it on every evaluation set. ]

Here is how you would run just the tests:
```
python models/roberta/train.py --test
```

## Evaluation sets

| Set                                 | What it measures                                                                 |
| ----------------------------------- | -------------------------------------------------------------------------------- |
| Dev (changemyview)                  | the set used to pick the best epoch, always printed, even without `--test`       |
| Seen test (changemyview)            | held-out pairs from the subreddit the model was trained on                       |
| Unseen test (one per subreddit)     | the other 17 SHP subreddits, never trained on, to see if the model generalizes   |
| Persuasion test (Winning Arguments) | pairs where `labels = 1` if comment A's author earned a delta, not more upvotes |

The persuasion test asks whether the model learned what makes an argument convincing, or only
what gets upvoted. Posts that also appear in SHP are dropped from it, so the model has never
seen them.

## Metrics

Each set prints three numbers:

- **Accuracy**: how often the model picks the comment that actually won.
- **Similar-length accuracy**: accuracy on only the pairs where the longer comment is less than
  1.25x the length of the shorter one. Longer comments tend to win, so this checks that the model
  is using the content of the comment and not just its length.
- **Length baseline accuracy**: the accuracy of always picking the longer comment. The model
  should beat this to be useful.