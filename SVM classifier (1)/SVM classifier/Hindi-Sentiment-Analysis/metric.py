"""
F-Score metrics for testing classifier, also includes functions for data extraction.
Author: Vivek Narayanan
"""
import os

def get_paths():
    """
    Returns supervised paths annotated with their actual labels.
    """
    pos_path = "./aclImdb/test/pos/"
    neg_path = "./aclImdb/test/neg/"

    if not os.path.exists(pos_path) or not os.path.exists(neg_path):
        raise FileNotFoundError("The dataset paths do not exist. Please verify the paths.")

    posfiles = [(os.path.join(pos_path, f), True) for f in os.listdir(pos_path) if os.path.isfile(os.path.join(pos_path, f))]
    negfiles = [(os.path.join(neg_path, f), False) for f in os.listdir(neg_path) if os.path.isfile(os.path.join(neg_path, f))]
    return posfiles + negfiles


def fscore(classifier, file_paths):
    tpos, fpos, fneg, tneg = 0, 0, 0, 0
    for path, label in file_paths:
        try:
            with open(path, encoding="utf-8") as f:
                result = classifier(f.read())
        except UnicodeDecodeError:
            print(f"Error reading file: {path}. Skipping.")
            continue

        if label and result:
            tpos += 1
        elif label and (not result):
            fneg += 1
        elif (not label) and result:
            fpos += 1
        else:
            tneg += 1

    total = tpos + tneg + fpos + fneg
    if tpos + fpos > 0:
        prec = tpos / (tpos + fpos)
    else:
        prec = 0.0
    if tpos + fneg > 0:
        recall = tpos / (tpos + fneg)
    else:
        recall = 0.0
    if prec + recall > 0:
        f1 = 2 * prec * recall / (prec + recall)
    else:
        f1 = 0.0

    accu = 100.0 * (tpos + tneg) / total if total > 0 else 0.0

    print(f"Precision: {prec:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"F1-Score: {f1:.6f}")
    print(f"Accuracy: {accu:.2f}%")
    print(f"True Positives: {tpos}, False Positives: {fpos}, False Negatives: {fneg}, True Negatives: {tneg}")


def main():
    try:
        from altbayes import classify, train  # Ensure altbayes is available
    except ImportError:
        raise ImportError("The module 'altbayes' could not be found. Please ensure it's in the working directory.")

    train()
    file_paths = get_paths()
    fscore(classify, file_paths)