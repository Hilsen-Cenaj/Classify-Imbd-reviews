from Load_data import count_reviews, classify_words, probabilities, dictionary
import math

Neg = 1
Pos = 0
reviews = 25000


def call_test():

    # tp->True positive,tn->True negative
    # fp->False positive ,fn->False negative
    tp, tn, fp, fn = 0, 0, 0, 0

    pos_words = []
    neg_words = []
    # pos_prob = []
    # neg_prob = []

    (neg_words, pos_words) = classify_words(neg_words, pos_words)

    total_words = len(pos_words) + len(neg_words)

    pos_prob = probabilities(pos_words)
    neg_prob = probabilities(neg_words)


# Score calculated with log #
    score = [math.log(float(count_reviews(0) / reviews)), math.log(float(count_reviews(1) / reviews))]

    for i, key, value in enumerate(dictionary.items()):

        if value in pos_words:
            score[Pos] += math.log(float(pos_prob[i])) * total_words

        if value in neg_words:
            score[Pos] += math.log(float(neg_prob[i])) * total_words

    if score[Pos] < score[Neg]:
