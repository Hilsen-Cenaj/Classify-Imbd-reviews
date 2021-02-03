import json
import tensorflow as tf
import numpy as np
from collections import Counter

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

vocabul_to_remove = [')', '(', "'", "<PAD>", "<START>", "<UNK>", "<UNUSED>", "the",
                     "on", "at", "in", "of", "a", "an", "and", "to", "it", "this", "was", "i", "/"]


# Load IMDB data to x_train, y_train, x_test, y_test
def load(count):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
        path="imdb.npz",
        num_words=1000,
        skip_top=0,
        maxlen=None,
        seed=113,
        start_char=1,
        oov_char=2,
        index_from=3,
    )
    x_train = x_train[:count]
    y_train = y_train[:count]

    return x_train, y_train


# Get word from frequency
def get_word(freq, dictionary):
    for key, value in dictionary.items():
        if freq == value:
            return key
    return "word doesn't exist"


# Get word frequency
def get_freq(word, dictionary):
    for key, value in dictionary.items():
        if word == key:
            return value
    return "frequency doesn't exist"


# Make review phrase
def get_phrase(review, dictionary):
    id_to_word = {value: key for key, value in dictionary.items()}
    splitted_review = ""
    for id in review:
        splitted_review += ''.join(id_to_word[id]) + " "  # connect every word in this review
    return splitted_review


# Count total reviews pos or neg
def count_reviews(score, y_train):
    total = 0
    for i in y_train:
        if i == score:
            total += 1
    return total


# Classify words from positive and negative
def classify_words(x_train, y_train, dictionary):
    i = 0
    co = {}
    co[0] = Counter()
    co[1] = Counter()

    while i < len(x_train):
        if y_train[i] == 1:
            d = get_phrase(x_train[i], dictionary)
            words = Counter(d.split())
            co[1] += words
        elif y_train[i] == 0:
            d = get_phrase(x_train[i], dictionary)
            words = Counter(d.split())
            co[0] += words
        i += 1
    return co[0], co[1]


def write_to_json(ct, f):
    with open(f + ".json", "w") as file_to_write:
        json.dump(ct, file_to_write)


def fill_vocabulary(v, x):
    for i in v:
        v[i] = 0
    for i in x:
        if i in v:
            v[i] += x[i]
    return v


def remove_unnecessary(count_words):
    to_remove = set()  # What is going to be removed
    for k in count_words:
        if k in vocabul_to_remove:
            # Picking what to remove
            to_remove.add(k)
    # Removing
    for i in to_remove:
        del count_words[i]
    to_remove.clear()
    return count_words


def call_train(samples):
    x_train, y_train = load(samples)
    INDEX_FROM = 3
    dictionary = tf.keras.datasets.imdb.get_word_index(path="imdb_word_index.json")
    dictionary = {k: (v + INDEX_FROM) for k, v in dictionary.items()}
    dictionary["<PAD>"] = 0
    dictionary["<START>"] = 1
    dictionary["<UNK>"] = 2
    dictionary["<UNUSED>"] = 3

    nc = [0, 0]
    nc[0] = count_reviews(0, y_train)
    nc[1] = count_reviews(1, y_train)
    print("Total number of reviews = ", samples)
    print("Negative = ", nc[1])
    print("Positive = ", nc[0])

    # Initializing arrays
    pos_words = []
    neg_words = []
    vocabulary = {}  # holds occurrences of each word in negative and positive
    condprob = {}
    number_of_words = {}  # holds number of words, 0 position positive, 1 position negative
    cl = [0, 1]  # Each class 0 => positive ,1 => negative

    # Fill vocabulary of negative reviews and positive reviews , holding the occurrence of each word
    (neg_words, pos_words) = classify_words(x_train, y_train, dictionary)

    pos_words = remove_unnecessary(pos_words)
    neg_words = remove_unnecessary(neg_words)
    # union the words of spams and hams
    voc = set(neg_words).union(set(pos_words))

    for c in cl:
        vocabulary[c] = Counter(voc)
        # Find in which class we are
        if cl == 1:
            w = neg_words
        else:
            w = pos_words
        vocabulary[c] = fill_vocabulary(vocabulary[c], w)
        number_of_words[c] = sum(value for value in vocabulary[c].values())
        condprob[c] = dict.fromkeys(vocabulary[c].elements(), 0)

    # We reset the vocabulary as it is before
    vocabulary[0] = dict(vocabulary[0])
    vocabulary[1] = dict(vocabulary[1])
    write_to_json(vocabulary[0], "vocPos" + str(samples))
    write_to_json(vocabulary[1], "vocNeg" + str(samples))

    for c in cl:
        for t in vocabulary[c].keys():
            condprob[c][t] = (vocabulary[c][t] + 1) / (number_of_words[c] + len(voc))
    # Save probabilities
    write_to_json(condprob[0], "probPos" + str(samples))
    write_to_json(condprob[1], "probNeg" + str(samples))
    print("Complete!")
