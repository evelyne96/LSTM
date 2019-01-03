import numpy as np
from LSTM import LSTM
from datetime import datetime
import pickle, random, dataProcessor

def train_model(model, X_train, y_train, learning_rate=0.001, nepoch=1000, evaluate_loss_after=4):
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
            save_model(model, name='models/LSTM.pickle', losses=losses)
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5 
                print("Setting learning rate to %f" % learning_rate)


        # For each training example...
        for i in range(len(y_train)):
            # One SGD step
            model.train_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1

def generate_sentence(model, word_index= [], index_word=[], max_len=15):
    # We start the sentence with the start token
    new_sentence = [word_index[dataProcessor.start_token]]
    # Repeat until we get an end token
    while not new_sentence[-1] == word_index[dataProcessor.end_token] and len(new_sentence) < max_len:
        next_word = model.predict(new_sentence)
        # We don't want to sample unknown words
        while next_word == word_index[dataProcessor.unknown_token]:
            next_word = model.predict(new_sentence) 

        new_sentence.append(next_word)

    sentence_str = [index_word[x] for x in new_sentence]
    return sentence_str
 

def generate_sentences(model, word_index= [], index_word=[], num_sentences=10, min_len=5):
    for i in range(num_sentences):
        sent = []
        # while len(sent) < min_len:
        sent = generate_sentence(model, word_index=word_index, index_word=index_word)
        print(sent)

def generate_test_data(model):
    # We start the sentence with the start token
    new_example = [random.randint(0, 10)]
    # Repeat until we get an end token
    while len(new_example) < 10:
        next_pred = model.predict(new_example)
        new_example.append(next_pred)

    print(new_example)
    return new_example

def geneate_test(model, n=10):
    for i in range(n):
        generate_test_data(model)

def train_lstm(word_dim=0, x=[], y=[], should_load=False):
    if should_load:
        lstm = load_model()
    else:
        lstm = LSTM(h_size=dataProcessor.hidden_layers, word_dim=word_dim)
    train_model(lstm, x, y, nepoch=2000)

def save_model(model, name="models/LSTM.pickle", losses=[]):
    pickle_out = open(name,"wb")
    pickle.dump(model, pickle_out)
    pickle_out.close()

    if len(losses) != 0:
        with open('models/loss.txt', 'a+') as f:
            for item in losses:  
                f.write(str(item) + '\n')
            f.close()
        

def load_model(name="models/LSTM.pickle"):
    pickle_in = open(name,"rb")
    pickled_model = pickle.load(pickle_in)

    return pickled_model
