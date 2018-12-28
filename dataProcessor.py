#!/usr/bin/env python3

import os, nltk, string, re, csv, itertools, numpy
from nltk.tokenize import word_tokenize

unknown_token = "UNKNOWN_TOKEN"
start_token = "START_TOKEN"
end_token = "END_TOKEN"
padding_token = "PADDING_TOKEN"
vocabulary_size = 1000
hidden_layers = 100

def read_from_csv(filename):
    modelList = []
    if os.path.exists(filename):
        with open(filename, encoding="ISO-8859-1") as file:
            reader = csv.reader(file)
            for row in reader:
                if (row[1] != 'text'):
                    sent = row[1][:100]
                    modelList.append(sent.lower())    
    modelList = ["%s %s %s" % (start_token, x, end_token) for x in modelList]
    modelList = modelList[:100]
    return modelList

def tokenize_dataset(dataset):
    tokenized_dataset = [word_tokenize(text) for text in dataset]
    return tokenized_dataset

def create_training_data():
    dataset = read_from_csv('./trump.csv')
    tokenized_dataset = tokenize_dataset(dataset)
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_dataset))
    # (word, frequency) pairs
    vocab = word_freq.most_common(vocabulary_size-1)
    # the index in the vector will give the index (ID) of the word
    index_word = [word for (word, freq) in vocab]
    index_word.append(unknown_token)
    word_index = dict([(word,ind) for ind,word in enumerate(index_word)])

    # Replace all words not in our vocabulary with the unknown token
    for i, text in enumerate(tokenized_dataset):
        tokenized_dataset[i] = [w if (w in word_index and w != 'https') else unknown_token for w in text]

    X_train = numpy.asarray([[word_index[w] for w in sent[:-1]] for sent in tokenized_dataset])
    y_train = numpy.asarray([[word_index[w] for w in sent[1:]] for sent in tokenized_dataset])
    return X_train, y_train, word_index, index_word
