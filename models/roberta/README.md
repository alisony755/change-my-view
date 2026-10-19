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
post + comment A ──► RoBERTa ──► score(A) ─┐
                                           ├──► score(A) - score(B) ──► sigmoid ──► P(A wins)
post + comment B ──► RoBERTa ──► score(B) ─┘
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

