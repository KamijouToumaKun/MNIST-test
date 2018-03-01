# -*- coding: UTF-8 -*-

import numpy as np
from numpy import matrix
from collections import namedtuple
import math
import os
import operator

class NaiveBayes:

    def __init__(self, data_matrix, data_labels, training_indices):
        self.data_matrix = data_matrix
        self.data_labels = data_labels
        self.dict = [[] for x in range(10)]
        #状态储存，离散化地认为一个像素点只有两种状态：是否>=0.5
        self.count = [[[-1 for y0i in range(2)] for i in range(28*28)] for label in range(10)]

        TrainData = namedtuple('TrainData', ['y0', 'label'])
        self.train([TrainData(self.data_matrix[i], int(self.data_labels[i])) for i in training_indices])

    #其实这不算training，只是把数据按照label归一下类
    def train(self, training_data_array):
        for data in training_data_array:
            self.dict[data.label].append(data.y0)

    #给定y0与label，求P(y0|label) P(label)
    def log_p_y0_label(self, y0, label):
        #由朴素贝叶斯假设
        #即求P(y0[0]|label) P(y0[1]|label) ... P(y0[N-1]|label) P(label)
        #求P(label)，这里默认len(self.dict[label]) > 0
        log_p = math.log(len(self.dict[label])) - math.log(len(self.data_matrix))
        for i in range(28*28):
            #求P(y0[i]|label)
            y0i = 1 if y0[i]>=0.5 else 0
            if self.count[label][i][y0i] != -1: #状态已储存
                c_label_y0i = self.count[label][i][y0i]
            else:
                c_label_y0i = 0
                for dict_y0 in self.dict[label]:
                    dict_state = 1 if dict_y0[i] >= 0.5 else 0
                    if dict_state == y0i:
                        c_label_y0i += 1
                self.count[label][i][y0i] = c_label_y0i

            c_label = len(self.dict[label])
            if c_label_y0i == 0: #拉普拉斯平滑
                c_label_y0i = 1
                c_label += 2 #特征有2个：y0[i] = 1或0
            log_p += math.log(c_label_y0i) - math.log(c_label)
        return log_p

    #给定y0，求argmax P(label|y0)
    #只需求argmax P(y0|label) P(label)
    def predict(self, test):
        results = [self.log_p_y0_label(test, label) for label in range(10)]
        return results.index(max(results))