import numpy as np

def step(x):
    return np.array(x > 0, dtype=np.int)

def sigmoid(x):
    return  1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    maxx = np.max(x)
    exp = np.exp(x - maxx)
    sum_exp = np.sum(exp)
    y = exp / sum_exp
    
    return y

