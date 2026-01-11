from __future__ import division
from math import log, exp
import os
import pickle
import matplotlib.pyplot as plt  # Correct library for plotting

# Custom dictionary to handle missing keys gracefully
class MyDict(dict):
    def __getitem__(self, key):
        return super().get(key, 0)

# Global variables
pos = MyDict()
neg = MyDict()
features = set()
totals = [0, 0]
CDATA_FILE = "countdata.pickle"
FDATA_FILE = "reduceddata.pickle"


def negate_sequence(text):
    """
    Detects negations and transforms negated words into "not_" form.
    """
    negation = False
    delims = "?.,!:;"
    result = []
    words = text.split()
    prev = None
    pprev = None
    for word in words:
        stripped = word.strip(delims).lower()
        negated = "not_" + stripped if negation else stripped
        result.append(negated)
        if prev:
            bigram = prev + " " + negated
            result.append(bigram)
            if pprev:
                trigram = pprev + " " + bigram
                result.append(trigram)
            pprev = prev
        prev = negated

        if any(neg in word for neg in ["not", "n't", "no"]):
            negation = not negation

        if any(c in word for c in delims):
            negation = False

    return result


def train():
    global pos, neg, totals
    retrain = False
    pos_path = "./aclImdb/train/pos"
    neg_path = "./aclImdb/train/neg"

    # Check if paths exist
    check_paths(pos_path, neg_path)

    # Load counts if they already exist.
    if not retrain and os.path.isfile(CDATA_FILE):
        with open(CDATA_FILE, 'rb') as f:
            pos, neg, totals = pickle.load(f)
        return

    limit = 12500
    for file in os.listdir(pos_path)[:limit]:
        with open(os.path.join(pos_path, file), encoding="utf-8") as f:
            for word in set(negate_sequence(f.read())):
                pos[word] += 1
                neg['not_' + word] += 1

    for file in os.listdir(neg_path)[:limit]:
        with open(os.path.join(neg_path, file), encoding="utf-8") as f:
            for word in set(negate_sequence(f.read())):
                neg[word] += 1
                pos['not_' + word] += 1

    prune_features()

    totals[0] = sum(pos.values())
    totals[1] = sum(neg.values())

    countdata = (pos, neg, totals)
    with open(CDATA_FILE, 'wb') as f:
        pickle.dump(countdata, f)



def classify(text):
    words = set(word for word in negate_sequence(text) if word in features)
    if not words:
        return True
    pos_prob = sum(log((pos[word] + 1) / (2 * totals[0])) for word in words)
    neg_prob = sum(log((neg[word] + 1) / (2 * totals[1])) for word in words)
    return pos_prob > neg_prob


def prune_features():
    """
    Remove features that appear only once.
    """
    global pos, neg
    keys_to_remove = {k for k in pos if pos[k] <= 1 and neg[k] <= 1}
    for key in keys_to_remove:
        del pos[key]
        del neg[key]


def MI(word):
    """
    Compute the weighted mutual information of a term.
    """
    T = totals[0] + totals[1]
    W = pos[word] + neg[word]
    if W == 0:
        return 0
    I = 0
    if neg[word] > 0:
        I += (totals[1] - neg[word]) / T * log(((totals[1] - neg[word]) * T) / ((T - W) * totals[1]))
        I += neg[word] / T * log((neg[word] * T) / (W * totals[1]))
    if pos[word] > 0:
        I += (totals[0] - pos[word]) / T * log(((totals[0] - pos[word]) * T) / ((T - W) * totals[0]))
        I += pos[word] / T * log((pos[word] * T) / (W * totals[0]))
    return I


def feature_selection_trials():
    """
    Select top k features. Vary k and plot data.
    """
    global pos, neg, totals, features
    retrain = True

    if not retrain and os.path.isfile(FDATA_FILE):
        with open(FDATA_FILE, 'rb') as f:
            pos, neg, totals = pickle.load(f)
        return

    words = list(pos.keys() | neg.keys())
    words.sort(key=lambda w: -MI(w))
    num_features, accuracy = [], []
    best_accuracy = 0.0
    path = "./aclImdb/test/"
    limit = 500
    start = 20000
    step = 500
    bestk = 0

    for w in words[:start]:
        features.add(w)

    for k in range(start, start + 20000, step):
        for w in words[k:k + step]:
            features.add(w)
        correct, size = 0, 0

        for file in os.listdir(os.path.join(path, "pos"))[:limit]:
            with open(os.path.join(path, "pos", file), 'r', encoding='utf-8') as f:
                correct += classify(f.read()) == True
                size += 1

        for file in os.listdir(os.path.join(path, "neg"))[:limit]:
            with open(os.path.join(path, "neg", file), 'r', encoding='utf-8') as f:
                correct += classify(f.read()) == False
                size += 1

        num_features.append(k + step)
        acc = correct / size
        accuracy.append(acc)
        if acc > best_accuracy:
            best_accuracy = acc
            bestk = k
        print(f"Features: {k + step}, Accuracy: {acc:.4f}")

    features = set(words[:bestk])
    with open(FDATA_FILE, 'wb') as f:
        pickle.dump((pos, neg, totals), f)

    plt.plot(num_features, accuracy)
    plt.xlabel("Number of Features")
    plt.ylabel("Accuracy")
    plt.title("Feature Selection")
    plt.show()

from prepare_svm import predict_sentiment

# Example input (this could be from a file or a direct string input)
sentence = "आज वह बहुत हँसा"
sentiment = predict_sentiment(sentence)
print(sentiment)
