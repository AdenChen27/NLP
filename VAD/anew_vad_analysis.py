"""
Based on: Doris Zhou, https://github.com/dwzhou/SentimentAnalysis/tree/master/src
Modified to increase efficiency. 
"""


import csv
import os
import statistics
import numpy as np
import pandas as pd

import nltk
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
stops = set(stopwords.words("english"))


def has_capital_letters(index):
    return any(char.isupper() for char in index)

anew_path = "EnglishShortened.csv"
ANEW = pd.read_csv(anew_path, index_col='Word')
ANEW.index = ANEW.index.map(str)
ANEW = ANEW[~ANEW.index.map(has_capital_letters)]
ANEW = ANEW.transpose().to_dict(orient='dict')
# {word: {'valence': ..., 'arousal': ..., 'dominance': ...},}


def analyze_sentence(sentence, agg=statistics.mean):
    sentence = tokenize.word_tokenize(sentence.lower())
    words = nltk.pos_tag(sentence)

    all_words = []
    found_words = []
    total_words = 0
    v_list = []  # holds valence scores
    a_list = []  # holds arousal scores
    d_list = []  # holds dominance scores

    # search for each valid word's sentiment in ANEW
    for index, p in enumerate(words):
        # don't process stops or words w/ punctuation
        word = p[0]
        pos = p[1]
        if word in stops or not word.isalpha():
            continue

        # check for negation in 3 words before current word
        j = index - 1
        neg = False
        while j >= 0 and j >= index - 3:
            if words[j][0] == 'not' or words[j][0] == 'no' or words[j][0] == 'n\'t':
                neg = True
                break
            j -= 1

        # lemmatize word based on pos
        if pos[0] == 'N' or pos[0] == 'V':
            lemma = lemmatizer.lemmatize(word, pos=pos[0].lower())
        else:
            lemma = word

        all_words.append(lemma)

        # search for lemmatized word in ANEW
        lemma = lemma.lower()
        if lemma in ANEW:
            row = ANEW[lemma]
            v = float(row['valence'])
            a = float(row['arousal'])
            d = float(row['dominance'])
            if neg:
                # reverse polarity for this word
                v = 5 - (v - 5)
                a = 5 - (a - 5)
                d = 5 - (d - 5)
            v_list.append(v)
            a_list.append(a)
            d_list.append(d)
    
    if v_list == []:
        return np.nan, np.nan, np.nan

    return agg(v_list), agg(a_list), agg(d_list)


def analyze_text(text, agg=statistics.mean):
    sentences = nltk.sent_tokenize(text)
    v_list, a_list, d_list = [], [], []

    for sentence in sentences:
        v, a, d = analyze_sentence(sentence)
        v_list.append(v)
        a_list.append(a)
        d_list.append(d)

    # print(v_list, a_list, d_list)
    # print(agg(v_list), agg(a_list), agg(d_list))
    return agg(v_list), agg(a_list), agg(d_list)



