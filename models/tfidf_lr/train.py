"""
Each comment becomes a TF-IDF vector. A pair becomes one vector, TF-IDF(A) - TF-IDF(B), and
logistic regression predicts labels = 1 if comment A got more upvotes. The post is not used.
Both TF-IDF and logistic regression are written with NumPy.
"""
from collections import Counter
import random
import sys
from pathlib import Path
import numpy as np
from scipy import sparse

sys.path.append(str(Path(__file__).resolve().parents[2]))  
from data.load_data import (length_ratio, load_seen_tests, load_training_data, load_unseen_tests,
                            pick_longer, tokenize)

# Settings to tune on the dev set
MIN_DOC_FREQUENCY = 2  # a word must be in at least this many training comments to be a feature
LEARNING_RATE = 0.01   # small, because the weights change after every single pair
EPOCHS = 20
SEED = 0


# TF-IDF features

class Tfidf:
    """TF-IDF: (count of the word in the comment) * log(N / comments containing the word)."""

    def fit(self, texts):
        """Chooses the vocabulary and computes the IDF values. Use the training comments only."""
        texts = list(set(texts)) # each comment once
        doc_frequency = Counter() # word -> # of comments that contain it
        for text in texts:
            doc_frequency.update(set(tokenize(text)))
        kept = sorted(word for word, count in doc_frequency.items() if count >= MIN_DOC_FREQUENCY)
        self.vocabulary = {word: index for index, word in enumerate(kept)} # word -> column number
        self.idf = np.array([np.log(len(texts) / doc_frequency[word]) for word in kept])

    def transform(self, texts):
        """One row per text, one column per vocabulary word. Words outside of the vocabulary are ignored."""
        rows, columns, values = [], [], []  # where each non-zero valur goes in the matrix
        for row, text in enumerate(texts):
            weights = {} # column number -> TF-IDF value, for this text
            for word, count in Counter(tokenize(text)).items():
                if word in self.vocabulary: # words outside the vocabulary are ignored
                    column = self.vocabulary[word]
                    weights[column] = count * self.idf[column]
            # Scale the vector to length 1, so a long comment does not get bigger values
            length = np.sqrt(sum(value ** 2 for value in weights.values()))
            for column, value in weights.items():
                rows.append(row)
                columns.append(column)
                values.append(value / length)
        shape = (len(texts), len(self.vocabulary))
        return sparse.csr_matrix((values, (rows, columns)), shape=shape)
        
def pair_features(tfidf, pairs):
    """TF-IDF(A) - TF-IDF(B). A positive value means the word is used more in comment A."""
    return tfidf.transform(list(pairs["human_ref_A"])) - tfidf.transform(list(pairs["human_ref_B"]))


# Logistic Regression (LR)

class LogisticRegression:
    """Binary LR trained with stochastic gradient descent."""

    def fit(self, features, labels):
        """Learns self.weights and self.bias."""
        random.seed(SEED)  # so every run shuffles the same way
        labels = np.asarray(labels)
        self.weights = np.zeros(features.shape[1])  # one weight per vocabulary word
        self.bias = 0.0
        order = list(range(features.shape[0]))  # the row numbers: 0, 1, 2, ...
        for epoch in range(EPOCHS):
            random.shuffle(order)  # visit the pairs in a new random order
            for row in order:
                pair = features[[row]]  # the features of one pair
                # gradient of the cross-entropy loss: (predicted - true) * feature value
                error = self.predict_probability(pair) - labels[row]
                weight_gradient = pair.T @ error  # (predicted - true) * feature value, for every word
                bias_gradient = error[0]          # (predicted - true), as if the bias's feature value were 1
                self.weights = self.weights - LEARNING_RATE * weight_gradient
                self.bias = self.bias - LEARNING_RATE * bias_gradient

    def predict_probability(self, features):
        """P(labels = 1), the probability that comment A wins."""
        scores = features @ self.weights + self.bias
        return 1 / (1 + np.exp(-scores)) 

    def predict(self, features):
        return (self.predict_probability(features) >= 0.5).astype(int)


# Evaluation

def accuracy(predictions, labels):
    return float(np.mean(np.asarray(predictions) == np.asarray(labels)))

def similar_length_accuracy(predictions, pairs, max_ratio=1.25):
    """Accuracy on the pairs where the longer comment is less than 1.25x the shorter one.
    There, picking the longer comment does not help, so the model has to use the words."""
    # TODO: keep the pairs with length_ratio(pairs) < max_ratio, return accuracy on those only
    raise NotImplementedError

def evaluate(name, model, tfidf, pairs):
    """Prints the model's accuracy next to the length baseline on one set of pairs."""
    # TODO: predict with model.predict(pair_features(tfidf, pairs)), then print
    #       accuracy, similar_length_accuracy, and accuracy of pick_longer(pairs)
    raise NotImplementedError

def top_words(model, tfidf, count=15):
    """The words with the largest weights: what the model thinks helps or hurts a comment."""
    # TODO: sort the columns by model.weights; print the words of the `count` largest
    #       (help a comment win) and the `count` smallest (hurt a comment)
    raise NotImplementedError

def compare_with_sklearn(model, train_features, train_labels, dev_features):
    """Trains scikit-learn's version of the same method and checks that it predicts like ours."""
    # TODO: train sklearn's SGDClassifier(loss="log_loss") with the same learning rate and epochs,
    #       print how often its dev predictions match ours
    raise NotImplementedError


if __name__ == "__main__":
    train, dev = load_training_data(columns_only=True)
    print(f"{len(train)} train pairs, {len(dev)} dev pairs")

    tfidf = Tfidf()
    tfidf.fit(list(train["human_ref_A"]) + list(train["human_ref_B"]))
    train_features = pair_features(tfidf, train)
    model = LogisticRegression()
    model.fit(train_features, train["labels"])

    evaluate("Dev (changemyview)", model, tfidf, dev)
    top_words(model, tfidf)
    compare_with_sklearn(model, train_features, train["labels"], pair_features(tfidf, dev))

    # The test sets are for the final numbers only: run with --test once the settings are chosen
    if "--test" in sys.argv:
        for name, pairs in load_seen_tests(columns_only=True).items():
            evaluate(f"Seen test ({name})", model, tfidf, pairs)
        for name, pairs in load_unseen_tests(columns_only=True).items():
            evaluate(f"Unseen test ({name})", model, tfidf, pairs)
        pass
