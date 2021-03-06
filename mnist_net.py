from os.path import dirname, realpath
import sys
import numpy as np
import pickle
from PIL import Image

oreilly_dataset_path = dirname(realpath(__file__)) + "/submodules/oreilly-dataset"
sys.path.append(oreilly_dataset_path)

from dataset.mnist import load_mnist
from activator import *

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

def get_test_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(flatten=True, normalize=False)
    return x_test, t_test

def get_train_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=False, one_hot_label=True)
    return x_train, t_train

def init_network():
    with open(oreilly_dataset_path + "/ch03/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network
        
def predict(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1)
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3)
    y = softmax(a3)

    return y

def main():
    x, t = get_test_data()
    network = init_network()
    
    batch_size = 100
    accuracy_cnt = 0

    for i in range(0, len(x), batch_size):
        x_batch = x[i:i+batch_size]
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis=1)
        accuracy_cnt += np.sum(p == t[i:i+batch_size])

    print("accuracy" + str(float(accuracy_cnt) / len(x)))

main()
