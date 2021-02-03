from Load_data import count_reviews, get_phrase, classify_words, remove_unnecessary
import math
import json
import csv
import tensorflow as tf
from collections import Counter


Neg = 1
Pos = 0
reviews = 25000

# Load data from Json files
def load_data_json(s):
    with open(s, "r") as fh:
        x = json.loads(fh.read())
    return dict(x)


def write_csv(name, t):
    fields = ['N.of docs', 'Accuracy', 'Precision', 'Recall', 'F1']
    with open(name, "a", newline='') as csv_file:
        csvwriter = csv.writer(csv_file, delimiter=";")
        # csvwriter.writerow(fields)
        csvwriter.writerows(t)


def load_test(count):
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
    x_test = x_test[:count]
    y_test = y_test[:count]

    return x_test, y_test


def call_test(samples):

    x_test, y_test = load_test(samples)

    INDEX_FROM = 3
    dictionary = tf.keras.datasets.imdb.get_word_index(path="imdb_word_index.json")
    dictionary = {k: (v + INDEX_FROM) for k, v in dictionary.items()}
    dictionary["<PAD>"] = 0
    dictionary["<START>"] = 1
    dictionary["<UNK>"] = 2
    dictionary["<UNUSED>"] = 3

    vocabulary = {}
    probabilities = {}

    # tp->True positive,tn->True negative
    # fp->False positive ,fn->False negative
    tp, tn, fp, fn = 0, 0, 0, 0

    # vocabulary[Neg] = load_data_json("vocabulary/vocNeg3600.json")
    # vocabulary[Pos] = load_data_json("vocabulary/vocPos3600.json")
    probabilities[Neg] = load_data_json("probabilities/probNeg3600.json")
    probabilities[Pos] = load_data_json("probabilities/probPos3600.json")

    # Calculating total positive/total negative reviews
    total_pos_reviews = count_reviews(0, y_test)
    total_neg_reviews = count_reviews(1, y_test)
    # The possibility of a positive/negative review occuring
    possibities_of_pos_neg = [total_pos_reviews / len(y_test), total_neg_reviews / len(y_test)]

    # Put each x_test list -> phrase
    # Split each word of the phrase -> word
    i = 0
    countw = Counter()
    while i < len(x_test):
        phrase = get_phrase(x_test[i], dictionary)
        word = Counter(phrase.split())
        countw += word
        i += 1

#############################
# Score calculated with log #
#############################
    score = [math.log(float(possibities_of_pos_neg[0])), math.log(float(possibities_of_pos_neg[1]))]
    j = 0
    while j < len(y_test):

        for word in countw:

            if word in probabilities[Pos]:
                score[Pos] += math.log(float(probabilities[Pos][word])) * countw[word]
            if word in probabilities[Neg]:
                score[Neg] += math.log(float(probabilities[Neg][word])) * countw[word]

        if score[Pos] < score[Neg]:
            if y_test[j] == 1:
                fn += 1
            else:
                tn += 1
        elif score[Pos] > score[Neg]:
            if y_test[j] == 0:
                fp += 1
            else:
                tp += 1
        j += 1
    score.clear()

    print(fn, tn, fp, tp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    print("--Test on algorithm with log--")
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    print("Accuracy = ", accuracy)
    print("Precision = ", precision)
    print("Recall = ", recall)
    print("f1 = ", f1)


for i in range(100, 600, 500):
    call_test(i)