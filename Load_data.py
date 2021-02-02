import tensorflow as tf
import numpy as np
from collections import Counter
import collections
from sklearn.model_selection import train_test_split

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

# Load IMDB data to x_train, y_train, x_test, y_test
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
    path="imdb.npz",
    num_words=None,
    skip_top=0,
    maxlen=None,
    seed=113,
    start_char=1,
    oov_char=2,
    index_from=3,
)

# x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)

# print(x_train[1])
# print(y_train[0])
# print(x_test[0])
# print(y_test[0])
# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)

# Put each word in dictionary
INDEX_FROM = 3
dictionary = tf.keras.datasets.imdb.get_word_index(path="imdb_word_index.json")
dictionary = {k: (v + INDEX_FROM) for k, v in dictionary.items()}
dictionary["<PAD>"] = 0
dictionary["<START>"] = 1
dictionary["<UNK>"] = 2
dictionary["<UNUSED>"] = 3


# Get word from frequency
def get_word(freq):
    for key, value in dictionary.items():
        if freq == value:
            return key
    return "word doesn't exist"


# Get word frequency
def get_freq(word):
    for key, value in dictionary.items():
        if word == key:
            return value
    return "frequency doesn't exist"


# Make review phrase
def get_phrase(i):
    id_to_word = {value: key for key, value in dictionary.items()}
    print(' '.join(id_to_word[id] for id in x_train[i]))


# Count total reviews pos or neg
def count_reviews(score):
    total = 0
    for i in y_train:
        if i == score:
            total += 1
    return total

# Classify words from positive and negative
def classify_words(neg, pos):
    i = 0
    while i < len(x_train):
        if y_train[i] == 1:
            for word in x_train[i]:
                neg.append(word)
        elif y_train[i] == 0:
            for word in x_train[i]:
                pos.append(word)
        i += 1
    return list(dict.fromkeys(neg)), list(dict.fromkeys(pos))

# Find probabilities for each word of list given
def probabilities(word_list):
    word_prob = {}
    for value in word_list.items():
        word_prob[value] = value / len(dictionary)
    return word_prob.items()


# (neg_words, pos_words) = classify_words(neg_words, pos_words)
# print(count_reviews(0))
# print(get_phrase(1))
# print(y_train[4])
# print(get_freq("l"))
# print(get_word(6434))
# print(dictionary.items())
# print(most_common_words())
