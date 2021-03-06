from sklearn.linear_model import LogisticRegression, BayesianRidge, LassoLars
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import wordnet
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from nltk import FreqDist
from sklearn.feature_extraction import FeatureHasher

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
ohe = OneHotEncoder()
d = cmudict.dict()
hasher = FeatureHasher(input_type='string')

pos_dict = {}

scrabble_scores = {"a": 1, "b": 3,"c": 3, "d": 2, "e": 1, "g": 2,
          "f": 4, "i": 1, "h": 4, "k": 5, "j": 8, "m": 3,
          "l": 1, "o": 1, "n": 1, "q": 10, "p": 3, "s": 1,
          "r": 1, "u": 1, "t": 1, "w": 4, "v": 4, "y": 4,
          "x": 8, "z": 10}

def score_scrabble(word):
    total = 0
    for l in word:
        if l in scrabble_scores:
            total += scrabble_scores[l]
    return total

def convertToNumber(s):
    return int.from_bytes(s.encode(), 'little')

def getPreprocessed(trainset):
    words = ""
    for sent in trainset:
        words += sent['sentence']

    return words

def number_synonyms(word):
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
        return sum([len(list(y for y in x if isdigit(y[-1]))) for x in d[word.lower()]])
    else:
        return sum([0])

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

class Baseline(object):

    def __init__(self, language):
        self.language = language
        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        if language == 'english':
            self.avg_word_length = 5.3
        else:  # spanish
            self.avg_word_length = 6.2

        self.model = LogisticRegression()

    def extract_features(self, word, sent):

        feature = []

        # Number of characters in word
        len_chars = len(word) / self.avg_word_length

        # Number of tokens in word
        len_tokens = len(word.split(' '))

        # Number of syllables in word
        no_syllables = nsyl(word)

        # Length of the lemma
        len_lemma = len(lancaster_stemmer.stem(word))

        return [len_chars, len_tokens, len_lemma]

    def train(self, trainset):
        X = []
        y = []
        for sent in trainset:
            X.append(self.extract_features(sent['target_word'],sent))
            y.append(sent['gold_label'])

        self.model.fit(X, y)

        # cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
        # plot_learning_curve(self.model, "LogisticRegression", X, y, ylim=(0, 1), cv=cv, n_jobs=4)

        plt.show()

    def test(self, testset):
        X = []
        for sent in testset:
            X.append(self.extract_features(sent['target_word'],testset))

        return self.model.predict(X)
