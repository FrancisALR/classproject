from sklearn.linear_model import LogisticRegression, BayesianRidge, LassoLars
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import wordnet
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from nltk import FreqDist

import curses
from curses.ascii import isdigit
import nltk
from nltk.corpus import cmudict
import re

import hashlib
lancaster_stemmer = LancasterStemmer()
ohe = OneHotEncoder()
d = cmudict.dict()

def frequencyMetric(trainset):
    words = ""
    for sent in trainset:
        words += sent['sentence']

    tokens = nltk.word_tokenize(words)
    fdist=FreqDist(tokens)
    return fdist

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
        return [len(list(y for y in x if isdigit(y[-1]))) for x in d[word.lower()]]
    else:
        return [0]

class Baseline(object):

    def __init__(self, language):
        self.language = language
        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        if language == 'english':
            self.avg_word_length = 5.3
        else:  # spanish
            self.avg_word_length = 6.2

        # self.model = RandomForestClassifier()
        # self.model = svm.SVC()
        self.model = MLPClassifier()

    def extract_features(self, word, sent, fdist):
        len_chars = len(word) / self.avg_word_length
        len_tokens = len(word.split(' '))
        no_syllables_list = nsyl(word)
        # pos_tag = nltk.pos_tag([word])
        no_syllables = sum(no_syllables_list)
        freq = fdist.freq(word)
        no_synonyms, no_antonyms = number_synonyms(word)
        len_lemma = len(lancaster_stemmer.stem(word))

        return [len_chars, len_tokens, no_antonyms, no_synonyms, len_lemma, no_syllables]

    def train(self, trainset):
        X = []
        y = []
        fdist = frequencyMetric(trainset)
        for sent in trainset:

            X.append(self.extract_features(sent['target_word'],sent, fdist))
            y.append(sent['gold_label'])

        self.model.fit(X, y)





    def test(self, testset):
        fdist = frequencyMetric(testset)
        X = []
        for sent in testset:
            X.append(self.extract_features(sent['target_word'],testset,fdist))

        return self.model.predict(X)
