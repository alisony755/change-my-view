"""
Each comment becomes a TF-IDF vector. A pair becomes one vector, TF-IDF(A) - TF-IDF(B), and
logistic regression predicts labels = 1 if comment A got more upvotes. The post is not used.
Both TF-IDF and logistic regression are written with NumPy.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))  
from data.load_data import load_training_data, pick_longer, tokenize

# Settings to tune on the dev set
MIN_DOC_FREQUENCY = 2  # a word must be in at least this many training comments to be a feature
LEARNING_RATE = 1.0
EPOCHS = 20
SEED = 0


# TF-IDF features

class Tfidf:
    """TF-IDF: (count of the word in the comment) * log(N / comments containing the word)."""

    def fit(self, texts):
        """Chooses the vocabulary and computes the IDF values. Use the training comments only."""
        # TODO: tokenize each text, count in how many texts each word appears,
        #       keep the words with at least MIN_DOC_FREQUENCY, store self.vocabulary and self.idf
        raise NotImplementedError

    def transform(self, texts):
        """One row per text, one column per vocabulary word."""
        # TODO: count the words of each text and multiply by self.idf
        raise NotImplementedError

def pair_features(tfidf, pairs):
    """TF-IDF(A) - TF-IDF(B). A positive value means the word is used more in comment A."""
    # TODO: transform pairs["human_ref_A"] and pairs["human_ref_B"], return the difference
    raise NotImplementedError


# Logistic regression

class LogisticRegression:
    """Binary logistic regression trained with stochastic gradient descent."""

    def fit(self, features, labels):
        """Learns self.weights and self.bias."""
        # TODO: start from zeros; for each epoch, shuffle, then update with the gradient
        #       (predicted probability - true label) * feature value
        raise NotImplementedError

    def predict_probability(self, features):
        """P(labels = 1), the probability that comment A wins."""
        # TODO: sigmoid(features @ self.weights + self.bias)
        raise NotImplementedError

    def predict(self, features):
        return (self.predict_probability(features) >= 0.5).astype(int)


# Evaluation

def accuracy(predictions, labels):
    return float(np.mean(np.asarray(predictions) == np.asarray(labels)))


if __name__ == "__main__":
    train, dev = load_training_data(columns_only=True)
    print(f"{len(train)} train pairs, {len(dev)} dev pairs")
    print(f"Baseline, pick the longer comment (dev): {accuracy(pick_longer(dev), dev['labels']):.3f}")

    tfidf = Tfidf()
    tfidf.fit(list(train["human_ref_A"]) + list(train["human_ref_B"]))
    model = LogisticRegression()
    model.fit(pair_features(tfidf, train), train["labels"])
    print(f"TF-IDF + LR (dev): {accuracy(model.predict(pair_features(tfidf, dev)), dev['labels']):.3f}")

    # TODO : compare with scikit-learn, report on the seen and unseen test sets (final run only),
    #             accuracy on similar-length pairs, top weighted words
