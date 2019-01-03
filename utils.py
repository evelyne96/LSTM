import numpy as np

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