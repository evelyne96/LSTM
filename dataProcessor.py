#!/usr/bin/env python3

import enum
import os, nltk, string, re, csv, itertools, numpy
from nltk.tokenize import word_tokenize
from enum import IntEnum
from keras.preprocessing.sequence import pad_sequences

unknown_token = "UNKNOWN_TOKEN"
start_token = "START_TOKEN"
end_token = "END_TOKEN"
padding_token = "PADDING_TOKEN"
max_len = 50
hidden_layers = 64

class TrainingDataType(IntEnum):
    TRUMP = 0
    EN_DE = 1
    TEST  = 2

def read_from_csv(filename):
    modelList = []
    if os.path.exists(filename):
        with open(filename, encoding="ISO-8859-1") as file:
            reader = csv.reader(file)
            for row in reader:
                if (row[1] != 'text'):
                    sent = row[1][:100]
                    sent = remove_punctuation(sent)
                    modelList.append(sent.lower())    
    modelList = ["%s %s %s" % (start_token, x, end_token) for x in modelList]
    modelList = modelList[:1000]

    return modelList

def remove_punctuation(text):
    text = text.strip(string.punctuation)
    text = re.sub(r'http.*', '', text)
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    text = re.sub(r'#[A-Za-z0-9]*', '', text)
    
    return text

def tokenize_dataset(dataset):
    return [word_tokenize(text) for text in dataset]

def word_index_from_dataset(dataset):
    word_freq = nltk.FreqDist(itertools.chain(*dataset))
    # (word, frequency) pairs
    vocab = word_freq.most_common(vocabulary_size-1)
    # the index in the vector will give the index (ID) of the word
    index_word = [word for (word, freq) in vocab]
    index_word.append(unknown_token)
    index_word.append(padding_token)
    index_word.append(end_token)
    word_index = dict([(word,ind) for ind,word in enumerate(index_word)])

    return word_index, index_word

def replace_with_unknown(tokenized_dataset, word_index):
    # Replace all words not in our vocabulary with the unknown token
    for i, text in enumerate(tokenized_dataset):
        tokenized_dataset[i] = [w if w in word_index else unknown_token for w in text]
    return tokenized_dataset

def create_trump_data():
    dataset = read_from_csv('./data/trump.csv')
    tokenized_dataset = tokenize_dataset(dataset)
    word_index, index_word = word_index_from_dataset(tokenized_dataset)
    tokenized_dataset = replace_with_unknown(tokenized_dataset, word_index)
   
    X_train = numpy.asarray([[word_index[w] for w in sent[:-1]] for sent in tokenized_dataset])
    y_train = numpy.asarray([[word_index[w] for w in sent[1:]] for sent in tokenized_dataset])

    vocabulary_size = len(word_index)

    return X_train, y_train, word_index, index_word, vocabulary_size

def create_en_de_data(limit=5000):
    en_file = './data/DE_EN/train.en'
    de_file = './data/DE_EN/train.de'
    data_en = []
    with open(en_file) as f:
        data_en = f.readlines()[:limit]
        data_en = ["%s %s" % (start_token, remove_punctuation(x.strip().lower())) for x in data_en] 
        data_en = tokenize_dataset(data_en)
    
    data_de = []
    with open(de_file) as f:
        data_de = f.readlines()[:limit]
        data_de = ["%s %s" % (start_token, remove_punctuation(x.strip().lower())) for x in data_de]
        data_de = tokenize_dataset(data_de)

    all_data = data_en + data_de
    word_index, index_word = word_index_from_dataset(all_data)
    data_en = replace_with_unknown(data_en, word_index)
    data_de = replace_with_unknown(data_de, word_index)

    X_train = numpy.asarray([[word_index[w] for w in sent[:-1]] for sent in data_en])
    y_train = numpy.asarray([[word_index[w] for w in sent[1:]] for sent in data_de])

    # pad them to be the same lenght
    X_train = pad_sequences(X_train, maxlen=max_len, padding='post', value=word_index[padding_token])
    y_train = pad_sequences(y_train, maxlen=max_len, padding='post', value=word_index[padding_token])

    X_train = [numpy.append(x, word_index[end_token]) for x in X_train]
    y_train = [numpy.append(y, word_index[end_token]) for y in y_train]

    vocabulary_size = len(word_index)
    return X_train, y_train, word_index, index_word, vocabulary_size

def create_test_data():
    X_train = []
    y_train = []
    index_word = []
    for i in range(100):
        index_word.append(i)
        index_word.append(i+10)
        X_train.append(range(i, i+10))
        y_train.append(range(i+1, i+11))

    index_word.append(99+11)

    word_index = dict([(word,ind) for ind,word in enumerate(index_word)])
    vocabulary_size = len(word_index)

    return X_train, y_train, word_index, index_word, vocabulary_size

def create_training_data(type=TrainingDataType.TRUMP):
    if type == TrainingDataType.TRUMP:
        return create_trump_data()
    if type == TrainingDataType.EN_DE:
        return create_en_de_data()
    if type == TrainingDataType.TEST:
        return create_test_data()


