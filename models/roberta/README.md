# How to run

To run the training script, you can run the train.py file in this directory.

There are a few arguments that you can pass to the script:

```
--train: trains the model on the training data
--test: evaluates the model on the test data
--resume-training: resumes an interrupted --train from the last completed epoch
--quick: runs a quick training just for testing purposes
```

So to train and test the entire model, you can run the following command:

```
python models/roberta/train.py --train --test
```

# Architecture

The model is a **comment scorer**: it reads a post and one comment and outputs a single number,
how good that comment is as a reply to that post. To compare two comments, we score each one
separately and predict that comment A wins if score(A) > score(B).

```
post + comment A ──► RoBERTa ──► score(A) ─┐
                                           ├──► score(A) - score(B) ──► sigmoid ──► P(A wins)
post + comment B ──► RoBERTa ──► score(B) ─┘
                  (same weights)
```

## Input

Each (post, comment) pair is encoded as one sequence, with the post as context:

```
<s> post </s></s> comment </s>
```

RoBERTa reads at most 512 tokens. When the post and comment together are longer than that, the
tokens are cut from whichever of the two is longer (`truncation="longest_first"`), so a long post
doesn't crowd out the comment. Setting `USE_POST = False` encodes the comment alone, for the
context-sensitivity experiment.

## Model

- **Encoder:** pretrained `roberta-base` (12 transformer layers, hidden size 768, about 125M
  parameters). All of its weights are fine-tuned.
- **Scoring head:** Hugging Face's sequence-classification head with `num_labels=1`. It takes the
  final hidden state of the `<s>` token and passes it through dropout, a 768 → 768 dense layer
  with tanh, dropout again, and a 768 → 1 linear layer. The result is the comment's score.
- **Shared weights:** both comments in a pair go through the same model, in one batch. Swapping
  A and B therefore just swaps the two scores, so the prediction flips with them. The model can't
  learn a bias toward whichever comment comes first.

## Training objective

The probability that A wins is modeled as P(A wins) = sigmoid(score(A) - score(B)), the
Bradley-Terry / RankNet pairwise formulation. Training minimizes binary cross-entropy between
this probability and the label (1 if comment A got more upvotes). This is the neural version of
the TF-IDF model, which classifies on TF-IDF(A) - TF-IDF(B); here the features are learned.

## Training setup

| Setting | Value |
| --- | --- |
| Optimizer | AdamW, learning rate 2e-5 |
| Learning rate schedule | linear warmup over the first 10% of steps, then linear decay to 0 |
| Batch size | 8 pairs (16 sequences through the model) |
| Epochs | 2 |
| Gradient clipping | max norm 1.0 |
| Precision | mixed fp16 on CUDA, fp32 elsewhere |
| Model selection | the epoch with the best dev accuracy is kept |
| Seed | 0 |

After every epoch, the full training state is saved to `resume_checkpoint/`, so an interrupted
run can be continued with `--resume-training`. The best epoch is saved to `checkpoint/` when
training finishes.

