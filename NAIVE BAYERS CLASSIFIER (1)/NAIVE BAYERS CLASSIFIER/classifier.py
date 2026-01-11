import nltk
import codecs
from nltk.util import bigrams
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

# Load stopwords
with codecs.open("stopwords.txt", 'r', 'utf-8') as f:
    stopwords = set(f.read().splitlines())

# Load training datasets
pos_tweets = codecs.open("pos_train.txt", 'r', 'utf-8')
neg_tweets = codecs.open("neg_train.txt", 'r', 'utf-8')
neu_tweets = codecs.open("neu_train.txt", 'r', 'utf-8')

# Combine datasets
tweets = []

def preprocess_and_add_to_tweets(file, sentiment):
    for words in file.readlines():
        words_filtered = [e.lower() for e in words.split() if len(e) >= 3 and e not in stopwords]
        tweets.append((words_filtered, sentiment))

preprocess_and_add_to_tweets(pos_tweets, "positive")
preprocess_and_add_to_tweets(neg_tweets, "negative")
preprocess_and_add_to_tweets(neu_tweets, "neutral")

# Get word features
def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
        all_words.extend(words)
    return all_words

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    return list(wordlist.keys())

word_features = get_word_features(get_words_in_tweets(tweets))

# Feature extractor
def get_bigram_features(words):
    bigram_list = list(bigrams(words))
    return bigram_list

def extract_features(document):
    document_words = set(document)
    features = {}
    
    # Unigrams
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    
    # Bigrams
    bigram_features = get_bigram_features(document)
    for bigram in bigram_features:
        features['contains_bigram(%s_%s)' % bigram] = True

    return features

# Training the classifier
training_set = nltk.classify.apply_features(extract_features, tweets)
classifier = nltk.NaiveBayesClassifier.train(training_set)

def classify_sentiment(text):
    # Preprocess and classify sentiment of the input text
    words = [e.lower() for e in text.split() if len(e) >= 3 and e not in stopwords]
    return classifier.classify(extract_features(words))

# Function to evaluate the classifier
def evaluate_classifier(test_file):
    with codecs.open(test_file, 'r', 'utf-8') as f:
        lines = f.readlines()

    actual_labels = []
    predicted_labels = []

    for line in lines:
        parts = line.strip().split()
        sentence = " ".join(parts[:-1])
        actual_sentiment = parts[-1]

        predicted_sentiment = classify_sentiment(sentence)

        actual_labels.append(actual_sentiment)
        predicted_labels.append(predicted_sentiment)

    # Calculate accuracy
    accuracy = accuracy_score(actual_labels, predicted_labels) * 100

    # Generate confusion matrix
    cm = confusion_matrix(actual_labels, predicted_labels, labels=["positive", "negative", "neutral"])

    return accuracy, cm
