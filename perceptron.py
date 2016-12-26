import numpy as np

def AND(x1, x2):
    w1, w2, b = 0.5, 0.5, -0.7
    return Activate(x1, x2, w1, w2, b)

def NAND(x1, x2):
    w1, w2, b = -0.5, -0.5, 0.7
    return Activate(x1, x2, w1, w2, b)

def OR(x1, x2):
    w1, w2, b = 1.0, 1.0, 0.0
    return Activate(x1, x2, w1, w2, b)

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    return AND(s1, s2)

def Activate(x1, x2, w1, w2, b):
    x =  np.array([x1, x2])
    w = np.array([w1, w2])
    inputs = np.sum(w * x) + b
    if inputs <= 0:
        return 0
    else:
        return 1

print('AND(0, 0) => %s' % AND(0, 0))
print('AND(1, 0) => %s' % AND(1, 0))
print('AND(0, 1) => %s' % AND(0, 1))
print('AND(1, 1) => %s' % AND(1, 1))

print('NAND(0, 0) => %s' % NAND(0, 0))
print('NAND(1, 0) => %s' % NAND(1, 0))
print('NAND(0, 1) => %s' % NAND(0, 1))
print('NAND(1, 1) => %s' % NAND(1, 1))

print('OR(0, 0) => %s' % OR(0, 0))
print('OR(1, 0) => %s' % OR(1, 0))
print('OR(0, 1) => %s' % OR(0, 1))
print('OR(1, 1) => %s' % OR(1, 1))

print('XOR(0, 0) => %s' % XOR(0, 0))
print('XOR(1, 0) => %s' % XOR(1, 0))
print('XOR(0, 1) => %s' % XOR(0, 1))
print('XOR(1, 1) => %s' % XOR(1, 1))
