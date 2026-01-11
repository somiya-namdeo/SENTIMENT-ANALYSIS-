from pretrained import positive, negative  # Import dictionaries from pretrained.py
from collections import Counter
import os

# Create a mapping of words to unique indices
tmap = dict(zip(list(positive.keys()) + list(negative.keys()), range(len(positive) + len(negative))))

POSITIVE_PATH = "./aclImdb/train/pos/"
NEGATIVE_PATH = "./aclImdb/train/neg/"

POSITIVE_TEST_PATH = "./aclImdb/test/pos/"
NEGATIVE_TEST_PATH = "./aclImdb/test/neg/"

def negate_sequence(text):
    """
    Detects negations and transforms negated words into "not_" form.
    """
    negation = False
    delims = "?.,!:;"
    result = []
    words = text.split()
    for word in words:
        stripped = word.strip(delims).lower()
        result.append("not_" + stripped if negation else stripped)

        if any(neg in word for neg in frozenset(["not", "n't", "no"])): 
            negation = not negation

        if any(c in word for c in delims):
            negation = False
    return result

def transform(path, cls):
    """
    Converts a file's text into a formatted string for SVM input.
    """
    with open(path, encoding="utf-8") as file:
        words = Counter(negate_sequence(file.read()))
    return "%s %s\n" % (cls, ' '.join('%d:%f' % (tmap[k], words[k]) for k in words if k in tmap))

def write_file(ofile, pospath, negpath):
    """
    Writes processed data to the output file in SVM format.
    """
    with open(ofile, "w", encoding="utf-8") as f:
        for fil in os.listdir(pospath):
            f.write(transform(os.path.join(pospath, fil), "+1"))
        for fil in os.listdir(negpath):
            f.write(transform(os.path.join(negpath, fil), "-1"))

def predict_sentiment(text):
    """
    Predicts sentiment based on the `positive` and `negative` dictionaries.
    """
    words = set(negate_sequence(text))
    pos_score = sum(positive.get(word, 0) for word in words)
    neg_score = sum(negative.get(word, 0) for word in words)
    return "+1" if pos_score >= neg_score else "-1"

if __name__ == '__main__':
    write_file('train.svmdata', POSITIVE_PATH, NEGATIVE_PATH)
    write_file('test.svmdata', POSITIVE_TEST_PATH, NEGATIVE_TEST_PATH)
