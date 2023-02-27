import numpy as np
import nltk

from nltk.stem.porter import PorterStemmer

def tokenize(string):
    return nltk.word_tokenize(string)

def stemming(string):
    ps = PorterStemmer()
    return ps.stem(string.lower())

def bow(tokenized_string,words):
    words_in_string = [stemming(word) for word in tokenized_string]
    bag = np.zeros(len(words),dtype=np.float32)
    for i,w in enumerate(words):
        if w in words_in_string:
            bag[i] = 1
    return bag 


