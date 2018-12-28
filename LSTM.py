#!/usr/bin/env python3
import numpy as np
from datetime import datetime

# Helpers


def sigmoid(x):
    return 1/(1+np.exp(-x))


def dsigmoid(x):
    return x * (1-x)


def tanh(x):
    return np.tanh(x)


def dtanh(x):
    return 1 - (x * x)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


class LSTM:
    def __init__(self, h_size=100, word_dim=100):
        self.h_size = h_size
        self.word_dim = word_dim
        self.bptt_truncate = 4

        # Parameters
        self.w_fh = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (h_size, h_size))
        self.w_fx = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (h_size, word_dim))
        self.b_f = np.zeros((h_size, 1))

        self.w_ih = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (h_size, h_size))
        self.w_ix = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (h_size, word_dim))
        self.b_i = np.zeros((h_size, 1))

        self.w_ch = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (h_size, h_size))
        self.w_cx = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (h_size, word_dim))
        self.b_c = np.zeros((h_size, 1))

        self.w_oh = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (h_size, h_size))
        self.w_ox = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (h_size, word_dim))
        self.b_o = np.zeros((h_size, 1))

        self.w_v = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (word_dim, h_size)) 
        self.b_v = np.zeros((word_dim, 1))

        self.null_deltas()
    
    def null_deltas(self):
        self.dLdWv = np.zeros(self.w_v.shape)
        self.dLdBv = np.zeros(self.b_v.shape)
        self.dLdWfx = np.zeros(self.w_fx.shape)
        self.dLdWfh = np.zeros(self.w_fh.shape)
        self.dLdBf = np.zeros(self.b_f.shape)
        self.dLdWix = np.zeros(self.w_ix.shape)
        self.dLdWih = np.zeros(self.w_ih.shape)
        self.dLdBi = np.zeros(self.b_i.shape)
        self.dLdWcx = np.zeros(self.w_cx.shape)
        self.dLdWch = np.zeros(self.w_ch.shape)
        self.dLdBc = np.zeros(self.b_c.shape)
        self.dLdWox = np.zeros(self.w_ox.shape)
        self.dLdWoh = np.zeros(self.w_oh.shape)
        self.dLdBo = np.zeros(self.b_o.shape)

    def clip_gradients(self):
        # Wx, Wh
        np.clip(self.dLdWv, -1, 1, out=self.dLdWv)
        np.clip(self.dLdWfx, -1, 1, out=self.dLdWfx)
        np.clip(self.dLdWfh, -1, 1, out=self.dLdWfh)
        np.clip(self.dLdWix, -1, 1, out=self.dLdWix)
        np.clip(self.dLdWih, -1, 1, out=self.dLdWih)
        np.clip(self.dLdWcx, -1, 1, out=self.dLdWcx)
        np.clip(self.dLdWch, -1, 1, out=self.dLdWch)
        np.clip(self.dLdWox, -1, 1, out=self.dLdWox)
        np.clip(self.dLdWoh, -1, 1, out=self.dLdWoh)
        # b
        np.clip(self.dLdBv, -1, 1, out=self.dLdBv)
        np.clip(self.dLdBf, -1, 1, out=self.dLdBf)
        np.clip(self.dLdBi, -1, 1, out=self.dLdBi)
        np.clip(self.dLdBc, -1, 1, out=self.dLdBc)
        np.clip(self.dLdBo, -1, 1, out=self.dLdBo)

    def forward(self, x):
        # The total number of time steps
        t = len(x)
        # During forward propagation we save all hidden states in s because need them later.
        # T+1 since the first one by indexing 0-1 will index the last element, so we don't need to care about that
        f = np.zeros((t + 1, self.h_size, 1))
        i = np.zeros((t, self.h_size, 1))
        o = np.zeros((t, self.h_size, 1))
        c = np.zeros((t + 1, self.h_size, 1))
        c_curr = np.zeros((t, self.h_size, 1))
        h = np.zeros((t + 1, self.h_size, 1))
        y = np.zeros((t, self.word_dim, 1))

        for t in np.arange(t):
            f[t] = sigmoid(np.dot(self.w_fh, h[t-1]) + self.w_fx[:, x[t]].reshape(self.h_size, 1) + self.b_f)
            i[t] = sigmoid(np.dot(self.w_ih, h[t-1]) + self.w_ix[:, x[t]].reshape(self.h_size, 1) + self.b_i)
            o[t] = sigmoid(np.dot(self.w_oh, h[t-1]) + self.w_ox[:, x[t]].reshape(self.h_size, 1) + self.b_o)
            c_curr[t] = sigmoid(np.dot(self.w_ch, h[t-1]) + self.w_cx[:, x[t]].reshape(self.h_size, 1) + self.b_c)
            c[t] = f[t] * c[t-1] + i[t] * c_curr[t]
            h[t] = o[t] * np.tanh(c[t])
            # Soft-max
            y[t] = softmax(np.dot(self.w_v, h[t]) + self.b_v)

        return f, i, o, c, c_curr, h, y

# https://blog.aidangomez.ca/2016/04/17/Backpropogating-an-LSTM-A-Numerical-Example/
    def bptt(self, x, y):
        # The total number of time steps
        t_steps = len(x)
        self.null_deltas()
        f, i, o, c, c_curr, h, y_ = self.forward(x)

        # y_ - 1 since 1 should the probability of choosing the correct word
        delta_y_ = y_
        delta_y_[np.arange(len(y)), y] -= 1.
        delta_h = np.zeros(h.shape)
        delta_c = np.zeros(c.shape)
        delta_f = np.zeros(f.shape)
        delta_i = np.zeros(i.shape)
        delta_o = np.zeros(o.shape)
        delta_c_curr = np.zeros(c_curr.shape)

        # For each output backwards...
        for t in np.arange(t_steps)[::-1]:
            # one hot encoding
            x_t = np.zeros((self.word_dim, 1))
            x_t[x[t]] = 1

            delta_h = np.dot(self.w_v.T, delta_y_[t]) + delta_h[t+1]
            delta_c = delta_c[t+1] * f[t+1] + delta_h * o[t] * dtanh(c[t])
            delta_f += delta_c * c[t-1] * dsigmoid(f[t])
            delta_i += delta_c * c_curr[t] * dsigmoid(i[t])
            delta_o += delta_h * dsigmoid(o[t]) * np.tanh(c[t])
            delta_c_curr += delta_c * i[t] * dtanh(c_curr[t])

            # W_v, b_v
            self.dLdWv += np.outer(delta_y_[t], h[t].T)
            self.dLdBv += delta_y_[t]

            # W_fx, W_fh, b_f
            self.dLdWfx += np.dot(delta_f[t], x_t.T)
            self.dLdWfh += np.dot(delta_f[t], h[t-1].T)
            self.dLdBf  += delta_f[t]

            # W_ix, W_ih, b_i
            self.dLdWix += np.dot(delta_i[t], x_t.T)
            self.dLdWih += np.dot(delta_i[t], h[t-1].T)
            self.dLdBi  += delta_i[t]

            # W_cx, W_ch, b_c
            self.dLdWcx += np.dot(delta_c_curr[t], x_t.T)
            self.dLdWch += np.dot(delta_c_curr[t], h[t-1].T)
            self.dLdBc  += delta_c_curr[t]

            # W_ox, W_oh, b_o
            self.dLdWox += np.dot(delta_o[t], x_t.T)
            self.dLdWoh += np.dot(delta_o[t], h[t-1].T)
            self.dLdBo  += delta_o[t]

        self.clip_gradients()


    def train_step(self, x, y, learning_rate):
        # Calculate the gradients
        self.bptt(x, y)
        # update parameters
        self.w_v    -= learning_rate * self.dLdWv
        self.b_v    -= learning_rate * self.dLdBv
        # f
        self.w_fh   -= learning_rate * self.dLdWfh
        self.w_fx   -= learning_rate * self.dLdWfx
        self.b_f    -= learning_rate * self.dLdBf
        # i
        self.w_ih   -= learning_rate * self.dLdWih
        self.w_ix   -= learning_rate * self.dLdWix
        self.b_i    -= learning_rate * self.dLdBi
        # c
        self.w_ch   -= learning_rate * self.dLdWch
        self.w_cx   -= learning_rate * self.dLdWcx
        self.b_i    -= learning_rate * self.dLdBc
        # o
        self.w_oh   -= learning_rate * self.dLdWoh
        self.w_ox   -= learning_rate * self.dLdWox
        self.b_o    -= learning_rate * self.dLdBo


    def calculate_total_loss(self, x, y):
        L = 0
        # For each sentence...
        for i in np.arange(len(y)):
            _, _, _, _, _, _, y_ = self.forward(x[i])
            # we get the probabilities for the real output words from the predictions that we made
            # since the o-output is the matrix which gives us the probabilities of each word from the vocabulary for time step t
            # and it has the predictions for each word we need the probabilities for the words which should have been predicted which we get from the target y
            correct_word_predictions = y_[np.arange(len(y[i])), y[i]]
            # Add to the loss based on how off we were (1 should be the expected probability)
            L += np.sum(np.log(correct_word_predictions))
        return L

    def calculate_loss(self, x, y):
        # Divide the total loss by the number of training examples
        N = np.sum((len(y_i) for y_i in y))
        # -1/N * sum(y_n * log(o_n))
        return -1 * self.calculate_total_loss(x,y)/N

    def predict(self, x):
        # Perform forward propagation and return index of the highest score
        _, _, _, _, _, _, y = self.forward(x)
        return np.argmax(y[-1])


