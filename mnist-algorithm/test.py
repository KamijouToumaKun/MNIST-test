# -*- coding: UTF-8 -*-

import os
import struct
from BPNeuralNetwork import BPNeuralNetwork
from Perceptron import Perceptron
from KNearestNeighbor import KNearestNeighbor
import numpy as np
import random

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)
    #归一化
    images = [x/255.0 for x in images]
    #这样归一化的效果不够好
    #images = [[1 if y>0 else 0 for y in x] for x in images]
    return images, labels

def test(nn, arg):
    correct_count = 0
    for i in range(10000):
        ans = nn.predict(test_matrix[i])
        if ans == test_labels[i]:
            correct_count += 1
    print '%d:%f' % (arg, correct_count/10000.0)

if __name__ == '__main__':
    # BASE_DIR = os.path.dirname(os.path.dirname(__file__)) 
    # 加载数据集
    data_matrix, data_labels = load_mnist(r'..\mnist', 'train')

    # 数据集一共60000个数据，train_indice存储用来训练的数据的序号
    train_indice = range(60000)
    # 打乱训练顺序
    random.shuffle(train_indice)

    test_matrix, test_labels = load_mnist(r'..\mnist', 't10k')
    
    # TEST
    print 'Perceptron:'
    epoch = 5
    nn = Perceptron(epoch, data_matrix, data_labels, train_indice)
    test(nn, epoch)
    
    print 'BPNeuralNetwork:'
    hidden_node_count = 15
    nn = BPNeuralNetwork(hidden_node_count, data_matrix, data_labels, train_indice)
    test(nn, hidden_node_count)

    print 'KNearestNeighbor:'
    # _k = 10
    # nn = KNearestNeighbor(_k, data_matrix, data_labels, train_indice)
    # test(nn, _k)