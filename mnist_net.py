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

def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(flatten=True, normalize=False)
    return x_test, t_test

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
    x, t = get_data()
    network = init_network()
    accuracy_cnt = 0
    for i in range(len(x)):
        y = predict(network, x[i])
        p = np.argmax(y)
        if p == t[i]:
            accuracy_cnt += 1

    print("accuracy" + str(float(accuracy_cnt) / len(x)))

main()
