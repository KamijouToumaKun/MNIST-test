# -*- coding: UTF-8 -*-

import csv
import numpy as np
from numpy import matrix
from collections import namedtuple
import math
import random
import os
import json

class Perceptron:
    LEARNING_RATE = 0.1
    # 保存感知机的文件路径
    NN_FILE_PATH = 'Perceptron.json'

    def __init__(self, max_epoch, data_matrix, data_labels, training_indices, use_file=True):
        # 决定了要不要导入nn.json
        self._use_file = use_file
        # 数据集
        self.data_matrix = data_matrix
        self.data_labels = data_labels

        if (not os.path.isfile(Perceptron.NN_FILE_PATH) or not use_file):
            # 初始化多类感知机参数
            self._w = self._rand_initialize_weights(28*28, 10)
            self._b = 0

            # 训练并保存
            TrainData = namedtuple('TrainData', ['y0', 'label'])
            for t in range(max_epoch):
                self.train([TrainData(self.data_matrix[i], int(self.data_labels[i])) for i in training_indices])
            self.save()
        else:
            # 如果json存在则加载
            self._load()

    def _rand_initialize_weights(self, size_in, size_out):
        return [((x * 0.12) - 0.06) for x in np.random.rand(size_out, size_in)]

    def w_dot_phi(self, y0, label):
        return np.dot(np.mat(self._w[label]), np.mat(y0).T) + self._b

    def train(self, training_data_array):
        # delta_w = [[0] * 28*28 for x in range(10)]
        # delta_b = 0
        for data in training_data_array:
            res = self.predict(data.y0)
            # 对于分类错误的数据，更新权重矩阵与偏置向量
            if res != data.label:
                # 注释掉的部分是平均感知机，诡异的是它的效果很差
                # delta_w[data.label] += self.LEARNING_RATE * data.y0
                # delta_w[res] -= self.LEARNING_RATE * data.y0
                # delta_b += self.LEARNING_RATE
                self._w[data.label] += self.LEARNING_RATE * data.y0
                self._w[res] -= self.LEARNING_RATE * data.y0
                self._b += self.LEARNING_RATE #多类感知机需要这个参数么？？？
        # for i in range(10):
        #     self._w[i] += delta_w[i] / 10000.0
        # self._b += delta_b / 10000.0

    def predict(self, test):
        results = [self.w_dot_phi(test, label) for label in range(10)]
        return results.index(max(results))

    def save(self):
        if not self._use_file:
            return

        json_neural_network = {
            "w":[np_mat.tolist() for np_mat in self._w],
            "b":self._b
        };
        with open(Perceptron.NN_FILE_PATH,'w') as nnFile:
            json.dump(json_neural_network, nnFile)

    def _load(self):
        if not self._use_file:
            return

        with open(Perceptron.NN_FILE_PATH) as nnFile:
            nn = json.load(nnFile)
        self._w = [np.array(li) for li in nn['w']]
        self._b = nn['b']