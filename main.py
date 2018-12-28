#!/usr/bin/env python3
import numpy as np
import dataProcessor, pickle, random
from LSTM import LSTM
from datetime import datetime

def train_model(model, X_train, y_train, learning_rate=0.005, nepoch=1000, evaluate_loss_after=5):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # Optionally evaluate the loss
        print("Epoch: ", epoch)
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss))
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5 
                print("Setting learning rate to %f" % learning_rate)

        # For each training example...
        for i in range(len(y_train)):
            # One SGD step
            model.train_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1

def generate_sentence(model):
    # We start the sentence with the start token
    new_sentence = [word_index[dataProcessor.start_token]]
    # Repeat until we get an end token
    while not new_sentence[-1] == word_index[dataProcessor.end_token]:
        next_word = model.predict(new_sentence)
        # We don't want to sample unknown words
        while next_word == word_index[dataProcessor.unknown_token]:
            next_word = model.predict(new_sentence)
        
        new_sentence.append(next_word)

    sentence_str = [index_word[x] for x in new_sentence]
    return sentence_str
 

def generate_sentences(model, num_sentences=10, min_len=5):
    sentences = []

    for i in range(num_sentences):
        sent = []
        # while len(sent) < min_len:
        sent = generate_sentence(model)
        sentences.append(" ".join(sent))

    print(sentences)


x_train, y_train, word_index, index_word = dataProcessor.create_training_data()

# lstm = LSTM(h_size=dataProcessor.hidden_layers, word_dim=dataProcessor.vocabulary_size)
# train_model(lstm, x_train, y_train, nepoch=100)
# pickle_out = open("LSTM_model.pickle","wb")
# pickle.dump(lstm, pickle_out)
# pickle_out.close()

pickle_in = open("LSTM_model.pickle","rb")
pickled_LSTM = pickle.load(pickle_in)

generate_sentences(pickled_LSTM, num_sentences=1)