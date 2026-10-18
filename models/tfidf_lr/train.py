"""
Each comment becomes a TF-IDF vector. A pair becomes one vector, TF-IDF(A) - TF-IDF(B), and
logistic regression (LR) predicts labels = 1 if comment A got more upvotes. The post is not used.
Both TF-IDF and LR are written by hand with NumPy and SciPy.
"""
from collections import Counter
import random
import sys
from pathlib import Path
import numpy as np
from scipy import sparse

sys.path.append(str(Path(__file__).resolve().parents[2]))  # the repo root, so we can import data/
from data.load_data import (TRAIN_SUBREDDITS, UNSEEN_SUBREDDITS, balance_length, length_ratio,
                            load_test_sets, load_training_data, load_winning_args, pick_longer,
                            tokenize)

# Settings to tune on the dev set
MIN_DOC_FREQUENCY = 2  # a word must be in at least this many training comments to be a feature
LEARNING_RATE = 0.01   # small, because the weights change after every single pair
EPOCHS = 20
SEED = 0
REGULARIZATION = 0.001  # L2 strength; larger = smaller weights, less overfitting
# Train only on pairs where the longer comment wins half the time, so the model cannot learn
# "longer wins" and has to use the words. False trains on all pairs.
LENGTH_BALANCED = True
# Mark the words after a negation (think -> think_NEG in "don't think"), since TF-IDF cannot see
# word order. In 5-fold cross-validation it was the only course method that helped (+1.3 points).
MARK_NEGATION = True


# TF-IDF features

class Tfidf:
    """TF-IDF: (count of the word in the comment) * log(N / comments containing the word)."""

    def fit(self, texts):
        """Chooses the vocabulary and computes the IDF values. Use the training comments only."""
        texts = list(set(texts)) # each comment once
        doc_frequency = Counter() # word -> # of comments that contain it
        for text in texts:
            doc_frequency.update(set(tokenize(text, mark_negation=MARK_NEGATION)))
        kept = sorted(word for word, count in doc_frequency.items() if count >= MIN_DOC_FREQUENCY)
        self.vocabulary = {word: index for index, word in enumerate(kept)} # word -> column number
        self.idf = np.array([np.log(len(texts) / doc_frequency[word]) for word in kept])

    def transform(self, texts):
        """One row per text, one column per vocabulary word. Words outside of the vocabulary are ignored."""
        rows, columns, values = [], [], []  # where each non-zero value goes in the matrix
        for row, text in enumerate(texts):
            weights = {} # column number -> TF-IDF value, for this text
            for word, count in Counter(tokenize(text, mark_negation=MARK_NEGATION)).items():
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


# LR

def sigmoid(scores):
    """Turns a score into a probability between 0 and 1."""
    return 1 / (1 + np.exp(-scores))

class LogisticRegression:
    """Binary LR trained with stochastic gradient descent (SGD)."""

    def fit(self, features, labels):
        """Learns self.weights and self.bias with stochastic gradient descent and L2 regularization."""
        rng = random.Random(SEED)                # own random generator, so other code isn't affected
        labels = np.asarray(labels)
        self.weights = np.zeros(features.shape[1])
        self.bias = 0.0
        order = list(range(features.shape[0]))
        for epoch in range(EPOCHS):
            rng.shuffle(order)
            for row in order:
                start, end = features.indptr[row], features.indptr[row + 1]
                words = features.indices[start:end]   # the words in this pair
                values = features.data[start:end]     # their feature values
                probability = sigmoid(values @ self.weights[words] + self.bias)
                error = probability - labels[row]     # predicted - true
                self.weights *= 1 - LEARNING_RATE * REGULARIZATION  # L2: shrink every weight a little
                # gradient = error * feature value; only the words in this pair have a non-zero value
                self.weights[words] -= LEARNING_RATE * error * values
                self.bias -= LEARNING_RATE * error    # the bias is not regularized

    def predict_probability(self, features):
        """P(labels = 1), the probability that comment A wins."""
        return sigmoid(features @ self.weights + self.bias)

    def predict(self, features):
        return (self.predict_probability(features) >= 0.5).astype(int)


# Evaluation

def accuracy(predictions, labels):
    return float(np.mean(np.asarray(predictions) == np.asarray(labels)))

def similar_length_accuracy(predictions, pairs, max_ratio=1.25):
    """Accuracy on the pairs where the longer comment is less than 1.25x the shorter one.
    There, picking the longer comment does not help, so the model has to use the words."""
    # Create mask (True for pairs with length ratio < max ratio)
    mask = length_ratio(pairs) < max_ratio
    
    # Return NaN if there are no pairs that meet the length-ratio requirement
    if not mask.any():
        return float("nan")
    
    # Return accuracy for similar-length pairs
    return accuracy(np.asarray(predictions)[mask.to_numpy()], 
                    pairs["labels"].to_numpy()[mask.to_numpy()])

def evaluate(name, model, tfidf, pairs):
    """Prints the model's accuracy next to the length baseline on one set of pairs."""
    # Convert each pair into TF-IDF(A) - TF-IDF(B), then predict which comment wins
    predictions = model.predict(pair_features(tfidf, pairs))
    
    # Extract the actual labels (1 means A won, 0 means B won)
    labels = pairs["labels"].to_numpy()

    # Calculate accuracy, similar-length accuracy, and length baseline accuracy
    model_accuracy = accuracy(predictions, labels)
    similar_accuracy = similar_length_accuracy(predictions, pairs)
    length_accuracy = accuracy(pick_longer(pairs), labels)

    # Accuracy on the pairs where picking the longer comment scores 50%
    balanced = pairs.index.get_indexer(balance_length(pairs).index)
    balanced_accuracy = accuracy(predictions[balanced], labels[balanced])

    # Print metrics to compare performance
    print(f"{name}")
    print(f"Accuracy: {model_accuracy:.4f}")
    print(f"Similar-length accuracy: {similar_accuracy:.4f}")
    print(f"Length-balanced accuracy: {balanced_accuracy:.4f} ({len(balanced)} pairs)")
    print(f"Length baseline accuracy: {length_accuracy:.4f}")

def top_words(model, tfidf, count=15):
    """The words with the largest weights: what the model thinks helps or hurts a comment."""
    # Get vocabulary words in the same order as the columns of the TF-IDF feature matrix
    words = np.array(list(tfidf.vocabulary.keys()))
    
    # Sort vocabulary indices from the smallest learned weight to the largest
    order = np.argsort(model.weights)

    # Positive weights increase the predicted probability that comment A wins
    print(f"Top {count} words favoring comment A:")
    for index in order[-count:][::-1]:
        print(f"  {words[index]}: {model.weights[index]:.4f}")

    # Negative weights decrease the probability that comment A wins, favoring comment B
    print(f"\nTop {count} words favoring comment B:")
    for index in order[:count]:
        print(f"  {words[index]}: {model.weights[index]:.4f}")

def compare_with_sklearn(model, train_features, train_labels, dev_features):
    """Trains scikit-learn's version of the same method and checks that it predicts like ours."""
    from sklearn.linear_model import SGDClassifier

    sklearn_model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=REGULARIZATION,
        learning_rate="constant",
        eta0=LEARNING_RATE,
        max_iter=EPOCHS,
        tol=None,
        random_state=SEED,
    )
    
    # Train the model using the same features and training labels
    sklearn_model.fit(train_features, np.asarray(train_labels))

    # Get both models' predictions for the same development examples
    our_predictions = model.predict(dev_features)
    sklearn_predictions = sklearn_model.predict(dev_features)

    # Calculate the fraction of dev examples where the models predict the same label
    match_rate = np.mean(our_predictions == sklearn_predictions)
    print(f"\nScikit-learn prediction agreement: {match_rate:.4f} ({match_rate:.1%})")


if __name__ == "__main__":
    train, dev = load_training_data(columns_only=True)
    if LENGTH_BALANCED:
        train = balance_length(train)
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
        for name, pairs in load_test_sets(TRAIN_SUBREDDITS, columns_only=True).items():
            evaluate(f"Seen test ({name})", model, tfidf, pairs)
        for name, pairs in load_test_sets(UNSEEN_SUBREDDITS, columns_only=True).items():
            evaluate(f"Unseen test ({name})", model, tfidf, pairs)
        # labels = 1 if comment A earned a delta: did the model learn persuasion, or only upvotes?
        evaluate("Persuasion test (Winning Arguments)", model, tfidf, load_winning_args(columns_only=True))
