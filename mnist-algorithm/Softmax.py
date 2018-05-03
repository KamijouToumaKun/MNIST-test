# -*- coding: UTF-8 -*-

import csv
import numpy as np
from numpy import matrix
from math import pow
from collections import namedtuple
import math
import random
import os
import json

class Softmax:
    LEARNING_RATE = 0.1
    # WIDTH_IN_PIXELS = 28
    # 保存神经网络的文件路径
    NN_FILE_PATH = 'Softmax.json'

    def __init__(self, num_hidden_nodes, data_matrix, data_labels, training_indices, use_file=True):
        # sigmoid函数
        self.sigmoid = np.vectorize(self._sigmoid_scalar)
        # sigmoid求导函数
        self.sigmoid_prime = np.vectorize(self._sigmoid_prime_scalar)
        # 决定了要不要导入json
        self._use_file = use_file
        # 数据集
        self.data_matrix = data_matrix
        self.data_labels = data_labels

        if (not os.path.isfile(Softmax.NN_FILE_PATH) or not use_file):
            # 初始化神经网络
            self.theta1 = self._rand_initialize_weights(28*28, num_hidden_nodes)
            self.theta2 = self._rand_initialize_weights(num_hidden_nodes, 10)
            self.input_layer_bias = self._rand_initialize_weights(1, num_hidden_nodes)
            self.hidden_layer_bias = self._rand_initialize_weights(1, 10)

            # 训练并保存
            TrainData = namedtuple('TrainData', ['y0', 'label'])
            self.train([TrainData(self.data_matrix[i], int(self.data_labels[i])) for i in training_indices])
            self.save()
        else:
            # 如果json存在则加载
            self._load()

    def _rand_initialize_weights(self, size_in, size_out):
        return [((x * 0.12) - 0.06) for x in np.random.rand(size_out, size_in)]

    def _sigmoid_scalar(self, z):
        if -z > 100:
            return 0
        elif -z < -100:
            return 1
        else:
            return 1 / (1 + math.e ** -z)

    def _sigmoid_prime_scalar(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def softmax(self, x):
        return np.exp(x) / float(sum(np.exp(x)))

    def train(self, training_data_array):
        for data in training_data_array:
            # 前向传播得到结果向量
            y1 = np.dot(np.mat(self.theta1), np.mat(data.y0).T)
            sum1 =  y1 + np.mat(self.input_layer_bias)
            y1 = self.sigmoid(sum1)

            y2 = np.dot(np.array(self.theta2), y1)
            y2 = np.add(y2, self.hidden_layer_bias)
            # 这里激活函数不用sigmoid，而用softmax
            # 更不能用sigmoid+softmax：归一化只有一次，否则误差太大
            y2 = self.softmax(y2)

            # 后向传播得到误差向量
            actual_vals = [0] * 10 
            actual_vals[data.label] = 1
            # 第一层是sigmoid，决定了计算hidden_error时的导数；
            # 第二层是softmax，决定了损失函数的最佳选择是对数似然（注意损失为loss，和error不是一个东西）
            # 但是我这里还是用的交叉熵；于是得到output_error的计算公式
            output_errors = np.mat(actual_vals).T - np.mat(y2) 
            # 误差δ[l] = (theta[l].T dot δ[l+1]) multiply g'(z[l])
            # for sigmoid, g'(z[l]) = a[l] * (1-a[l])
            # 用到的sigmoid的导数对应上一层
            # 一般softmax都用在最后一层，不会用到其导数（正好，我还不太懂它的导数怎么算）
            hidden_errors = np.multiply(np.dot(np.mat(self.theta2).T, output_errors), self.sigmoid_prime(sum1))

            # 更新权重矩阵与偏置向量
            self.theta1 += self.LEARNING_RATE * np.dot(np.mat(hidden_errors), np.mat(data.y0))
            self.theta2 += self.LEARNING_RATE * np.dot(np.mat(output_errors), np.mat(y1).T)
            self.hidden_layer_bias += self.LEARNING_RATE * output_errors
            self.input_layer_bias += self.LEARNING_RATE * hidden_errors

    def predict(self, test):
        y1 = np.dot(np.mat(self.theta1), np.mat(test).T)
        y1 = y1 + np.mat(self.input_layer_bias) # Add the bias
        y1 = self.sigmoid(y1)

        y2 = np.dot(np.array(self.theta2), y1)
        y2 = np.add(y2, self.hidden_layer_bias) # Add the bias
        y2 = self.softmax(y2)

        results = y2.T.tolist()[0]
        return results.index(max(results))

    def save(self):
        if not self._use_file:
            return

        json_neural_network = {
            "theta1":[np_mat.tolist()[0] for np_mat in self.theta1],
            "theta2":[np_mat.tolist()[0] for np_mat in self.theta2],
            "b1":[np_mat.tolist()[0] for np_mat in self.input_layer_bias],
            "b2":[np_mat.tolist()[0] for np_mat in self.hidden_layer_bias],
        };
        with open(Softmax.NN_FILE_PATH,'w') as nnFile:
            json.dump(json_neural_network, nnFile)

    def _load(self):
        if not self._use_file:
            return

        with open(Softmax.NN_FILE_PATH) as nnFile:
            nn = json.load(nnFile)
        self.theta1 = [np.array(li) for li in nn['theta1']]
        self.theta2 = [np.array(li) for li in nn['theta2']]
        self.input_layer_bias = [np.array(li) for li in nn['b1']]
        self.hidden_layer_bias = [np.array(li) for li in nn['b2']]
