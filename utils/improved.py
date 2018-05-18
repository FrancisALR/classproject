from sklearn.ensemble import RandomForestClassifier
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import wordnet

from sklearn.feature_extraction import FeatureHasher
from collections import Counter
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
from sklearn.metrics import explained_variance_score, make_scorer
from sklearn.cross_validation import KFold
import numpy as np
import curses
from curses.ascii import isdigit
import nltk
from nltk.corpus import cmudict
import re
import math
import hashlib

lancaster_stemmer = LancasterStemmer()
d = cmudict.dict()
hasher = FeatureHasher(input_type='string')

pos_dict = {}

letter_scores = {"z":5,"j":4,"x":4,"Q":3,"k":2,"v":1}

def get_frequencies(data):
    freqs = {}
    for sent in data:
        sentence = sent["sentence"]
        target_word = sent["target_word"]
        if target_word in freqs:
            freqs[target_word] += 1
        else:
            freqs[target_word] =1
    return freqs

def score_letters(word):
    total = 0
    for l in word:
        if l in letter_scores:
            total += letter_scores[l]
    return total

def convertToNumber(s):
    return int.from_bytes(s.encode(), 'little')

def number_nyms(word):
    synonyms = []
    antonyms = []
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name())
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())
    return len(synonyms), len(antonyms)

def nsyl(word):
    if word in d:
        return sum([len(list(i for i in j if isdigit(i[-1]))) for j in d[word.lower()]])
    else:
        return sum([0])

# Code from scikit-learn http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

class Improved(object):

    def __init__(self, language):
        self.language = language
        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        if language == 'english':
            self.avg_word_length = 5.3
        else:  # spanish
            self.avg_word_length = 6.2

        self.model = RandomForestClassifier()

    def extract_features(self, word, sent, freqs):

        feature = []

        # Number of characters in word
        len_chars = len(word) / self.avg_word_length

        # Number of tokens in word
        len_tokens = len(word.split(' '))

        # Number of syllables in word
        no_syllables = nsyl(word)

        # Part of speech tag of word
        pos_tag = convertToNumber(nltk.pos_tag([word])[0][1])

        no_synonyms, no_antonyms = number_nyms(word)

        len_lemma = len(lancaster_stemmer.stem(word))

        letter_score = score_letters(word)

        word_freq = freqs[word]

        return [len_chars, len_tokens, len_lemma, word_freq, no_syllables]

    def train(self, trainset):
        X = []
        y = []
        freqs = get_frequencies(trainset)

        for sent in trainset:
            X.append(self.extract_features(sent['target_word'],sent,freqs))
            y.append(sent['gold_label'])

        self.model.fit(X, y)

        plt.show()

    def test(self, testset):
        X = []
        freqs = get_frequencies(testset)

        for sent in testset:
            X.append(self.extract_features(sent['target_word'],sent, freqs))

        return self.model.predict(X)
