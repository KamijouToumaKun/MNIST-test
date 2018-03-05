# -*- coding: UTF-8 -*-
# 这不是我的原创，不过我从这份源码中学习到了很多，感觉非常受益。
# 我认真读过以后，又加了一些注释、进行了一点修改。
# import csv
import numpy as np
from numpy import matrix
from math import pow
from collections import namedtuple
import math
import random
import os
import json

import copy
import scipy.signal as signal  

def rand_arr(a, b, *args):   
    np.random.seed(0)   
    return np.random.rand(*args) * (b - a) + a  

class CNN:
    # 保存神经网络的文件路径，我没有加保存功能
    # NN_FILE_PATH = 'CNN.json'
    def __init__(self, data_matrix, data_labels, training_indices):  
        self.C_SIZE = 5  #卷积核/滤波器的长与宽，据说一般要是奇数？28*28的图像，按经验C_SIZE=5    
        self.F_NUM = 12  #全连接层的卷积核/滤波器的长与宽，这里就不是奇数了
        self.P_SIZE = 2  #池化窗口的长与宽
        self.MAX_ITER_NUM = 10 #本来是50，很花时间；但是也不要改成1，准确率太低了

        self.data_matrix = data_matrix #卷积层输入图像
        self.data_labels = data_labels

        cLyNum = 20
        pLyNum = 20
        fLyNum = 100
        oLyNum = 10
        self.cLyNum = cLyNum  #卷积层输出、池化层输入图像的个数 = 卷积核数 = feature map数
        self.pLyNum = pLyNum  #池化层输出、全连接层输入图像的个数, = cLyNum
        self.fLyNum = fLyNum  #全连接层输出图像深度，全连接层输出图像为matrix 1*fLyNum
        self.oLyNum = oLyNum  #输出神经元数目，matrix 1*fLyNum -> 1*oLyNum
        self.pSize = self.P_SIZE 
        self.yita = 0.01  #学习率
        
        self.cLyBias = rand_arr(-0.1, 0.1, 1,cLyNum) #matrix 1*cLyNum
        self.fLyBias = rand_arr(-0.1, 0.1, 1,fLyNum) #matrix 1*fLyNum
        self.kernel_c = np.zeros((self.C_SIZE,self.C_SIZE,cLyNum))  
        self.kernel_f = np.zeros((self.F_NUM,self.F_NUM,fLyNum))  
        for i in range(cLyNum):  
        	#各个卷积核也是参数，它的初始化也是随机的，之后也通过反向传播来更新
            self.kernel_c[:,:,i] = rand_arr(-0.1,0.1,self.C_SIZE,self.C_SIZE)  
        for i in range(fLyNum):  
            #全连接层的卷积核
            self.kernel_f[:,:,i] = rand_arr(-0.1,0.1,self.F_NUM,self.F_NUM)  
        self.pooling_a = np.ones((self.pSize,self.pSize))/(self.pSize**2)    
        self.weight_f = rand_arr(-0.1,0.1, pLyNum, fLyNum)  
        self.weight_output = rand_arr(-0.1,0.1,fLyNum,oLyNum)
        
        # 训练并保存
        TrainData = namedtuple('TrainData', ['y0', 'label'])
        for i in range(self.MAX_ITER_NUM):  
            #注意，用CNN的话不要把图像的灰度值归一处理，效果很差
            self.train([TrainData(self.data_matrix[i], int(self.data_labels[i])) for i in training_indices])
        #self.save()

    def _rand_initialize_weights(self, size_in, size_out):
        return [((x * 0.12) - 0.06) for x in np.random.rand(size_out, size_in)]

    #卷积层  
    def convolution(self, data, kernel):  
        data_row, data_col = np.shape(data)  
        kernel_row, kernel_col = np.shape(kernel)  
        n = data_col - kernel_col  
        m = data_row - kernel_row  
        state = np.zeros((m+1, n+1))  
        for i in range(m+1):  
            for j in range(n+1):  
                temp = np.multiply(data[i:i+kernel_row,j:j+kernel_col], kernel)  
                state[i][j] = temp.sum()  
        return state          
    # 池化层（通过平均池化来压缩，不过一般选用最大池化）
    def pooling(self, data, pooling_a):       
        data_r, data_c = np.shape(data)  
        p_r, p_c = np.shape(pooling_a)  
        r0 = data_r/p_r  
        c0 = data_c/p_c  
        state = np.zeros((r0,c0))  
        for i in range(c0):  
            for j in range(r0):  
                temp = np.multiply(data[p_r*i:p_r*(i+1),p_c*j:p_c*(j+1)],pooling_a)  
                state[i][j] = temp.sum()  
        return state  
    #全连接层，相当于乘以一个weight_f的共用参数，再进行卷积
    def convolution_f1(self, state_p1, kernel_f1, weight_f1):  
        #池化层出来的20个特征矩阵乘以池化层与全连接层的连接权重进行相加  
        #wx(这里的偏置项=0),这个结果然后再和全连接层中的神经元的核  
        #进行卷积,返回值:  
        #1：全连接层卷积前,只和weight_f1相加之后的矩阵  
        #2：和全连接层卷积完之后的矩阵    

        #cLyNum, fLyNum      
        n_p0, n_f = np.shape(weight_f1)#n_p0=20(是Feature Map的个数);n_f是100(全连接层神经元个数)  
        #m_p, n_p, cLyNum
        m_p, n_p, pCnt = np.shape(state_p1)#这个矩阵是三维的  
        #F_NUM, F_NUM, fLyNum
        m_k_f1, n_k_f1,fCnt = np.shape(kernel_f1)#12*12*100  

        state_f1_temp = np.zeros((m_p,n_p,n_f))  
        state_f1 = np.zeros((m_p - m_k_f1 + 1,n_p - n_k_f1 + 1,n_f))     
        # state_p(m_c*n_c*cLyNum) dot weight_f(cLyNum*fLyNum) = state_f1_temp(m_c*n_c*fLyNum)
        # 本来矩阵乘法中，还要在最外层对state_p1的前两维进行循环；
        for n in range(n_f):  
            count = 0  
            for m in range(n_p0):  
                # 不过这里可以直接在循环内部进行向量乘法、求和
                temp = state_p1[:,:,m] * weight_f1[m][n]   
                count = count + temp  
            state_f1_temp[:,:,n] = count  
            # 再对state_f1_temp做卷积
            state_f1[:,:,n] = self.convolution(state_f1_temp[:,:,n], kernel_f1[:,:,n])    
        return state_f1, state_f1_temp    
    # softmax 层  
    def softmax_layer(self,state_f1):  
        # print 'softmax_layer'  
        output = np.zeros((1,self.oLyNum))  
        t1 = (np.exp(np.dot(state_f1,self.weight_output))).sum()  
        for i in range(self.oLyNum):  
            t0 = np.exp(np.dot(state_f1,self.weight_output[:,i]))  
            output[:,i]=t0/t1  
        return output     
    #误差反向传播更新权值，我还不是很懂
    def cnn_upweight(self,err_cost, ylab, train_data,state_c1, state_s1, state_f1, state_f1_temp, output):  
        #print 'cnn_upweight'  
        m_data, n_data = np.shape(train_data)  
        # softmax的资料请查看 (TODO)  
        label = np.zeros((1,self.oLyNum))  
        label[:,ylab] = 1   
        delta_layer_output = output - label  #最终输出层的误差（符号是负的，所以更新时是-=而不是+=）

        weight_output_temp = copy.deepcopy(self.weight_output)  
        delta_weight_output_temp = np.zeros((self.fLyNum, self.oLyNum))  
        #print np.shape(state_f1)  
        #更新weight_output              
        for n in range(self.oLyNum):  
            # state_fo dot weight_output + bias 再激活-> layer_output
            # 故weight_output += yita * (delta_layer_output dot state_fo)
            delta_weight_output_temp[:,n] = delta_layer_output[:,n] * state_f1  
        weight_output_temp = weight_output_temp - self.yita * delta_weight_output_temp  
          
        #更新bias_f和kernel_f (推导公式请查看 TODO)     
        delta_layer_f1 = np.zeros((1, self.fLyNum))  
        delta_bias_f1 = np.zeros((1,self.fLyNum))  
        delta_kernel_f1_temp = np.zeros(np.shape(state_f1_temp))  
        kernel_f_temp = copy.deepcopy(self.kernel_f)              
        for n in range(self.fLyNum):  
            # 这层的误差（即delta_layer_f1）= 
            # 下一层的误差（即delta_layer_output）
            # dot 这层的参数（即weight_output）
            # 整体 multiply 上层的激活函数的导数代入sum（即tanh^-1(state_f_pre)）
            count = 0  
            for m in range(self.oLyNum):  
                count = count + delta_layer_output[:,m] * self.weight_output[n,m]
            #本来是multiply，但是这里展开对每一维计算，每一维都是数*数，dot也可以       
            delta_layer_f1[:,n] = np.dot(count, (1 - np.tanh(state_f1[:,n])**2)) 
            delta_bias_f1[:,n] = delta_layer_f1[:,n]  
            delta_kernel_f1_temp[:,:,n] = delta_layer_f1[:,n] * state_f1_temp[:,:,n]  
        # 1  
        self.fLyBias = self.fLyBias - self.yita * delta_bias_f1  
        kernel_f_temp = kernel_f_temp - self.yita * delta_kernel_f1_temp  
           
        #更新weight_f1  
        delta_layer_f1_temp = np.zeros((self.F_NUM,self.F_NUM,self.fLyNum))  
        delta_weight_f1_temp = np.zeros(np.shape(self.weight_f))  
        weight_f1_temp = copy.deepcopy(self.weight_f)  
        for n in range(self.fLyNum):  
            delta_layer_f1_temp[:,:,n] = delta_layer_f1[:,n] * self.kernel_f[:,:,n]  
        for n in range(self.pLyNum):  
            for m in range(self.fLyNum):  
                temp = delta_layer_f1_temp[:,:,m] * state_s1[:,:,n]  
                delta_weight_f1_temp[n,m] = temp.sum()  
        weight_f1_temp = weight_f1_temp - self.yita * delta_weight_f1_temp  

        # 更新bias_c1  
        n_delta_c = m_data - self.C_SIZE + 1  
        delta_layer_p = np.zeros((self.F_NUM,self.F_NUM,self.pLyNum))  
        delta_layer_c = np.zeros((n_delta_c,n_delta_c,self.pLyNum))  
        delta_bias_c = np.zeros((1,self.cLyNum))  
        for n in range(self.pLyNum):  
            count = 0  
            for m in range(self.fLyNum):  
                count = count + delta_layer_f1_temp[:,:,m] * self.weight_f[n,m]  
            delta_layer_p[:,:,n] = count  
            #print np.shape(np.kron(delta_layer_p[:,:,n], np.ones((2,2))/4))  
            delta_layer_c[:,:,n] = np.kron(delta_layer_p[:,:,n], np.ones((2,2))/4) * (1 - np.tanh(state_c1[:,:,n])**2)  
            delta_bias_c[:,n] = delta_layer_c[:,:,n].sum()  
        # 2  
        self.cLyBias = self.cLyBias - self.yita * delta_bias_c  
        #更新 kernel_c1  
        delta_kernel_c1_temp = np.zeros(np.shape(self.kernel_c))  
        for n in range(self.cLyNum):  
            temp = delta_layer_c[:,:,n]  
            r1 = map(list,zip(*temp[::1]))#逆时针旋转90度           
            r2 = map(list,zip(*r1[::1]))#再逆时针旋转90度  
            temp = signal.convolve2d(train_data, r2,'valid')  
            temp1 = map(list,zip(*temp[::1]))  
            delta_kernel_c1_temp[:,:,n] = map(list,zip(*temp1[::1]))  
        self.kernel_c = self.kernel_c - self.yita * delta_kernel_c1_temp                                                  
        self.weight_f = weight_f1_temp  
        self.kernel_f = kernel_f_temp  
        self.weight_output = weight_output_temp               

    def train(self, training_data_array):
        for data_total in training_data_array:
            data = data_total.y0
            ylab = data_total.label

            self.predict(data, ylab)  #预测并训练
    
    def predict(self, test, ylab=None):
        data = test.reshape(28,28)
        d_m, d_n = np.shape(data)  #卷积层输入图像的长与宽
        # 假设输入图像为w1 * h1 * c1，w1表示宽度，h1表示高度，c1表示通道数（图像的深度），输出图像为w2 * h2 * c2
        # 在卷积层中，n表示卷积核的个数，k*k表示卷积核大小，p表示扩充边缘，s表示卷积核步长，那么有：
        # w2 = (w1 + 2 * p - k) / s + 1
        # h2 = (h1 + 2 * p - k) / s + 1
        # c2 = n
        # 这里做步长stride=1的卷积，扩充边缘p=0
        m_c = d_m - self.C_SIZE + 1  #卷积层输出、池化层输入图像的长与宽
        n_c = d_n - self.C_SIZE + 1  #d_m = d_n = 28, C_SIZE = 5, 28-5+1 = 24
        m_p = m_c/self.pSize #m_c = n_c = 24, pSize = 2, 24/2 = 12
        n_p = n_c/self.pSize  
        state_c = np.zeros((m_c, n_c,self.cLyNum))  #卷积层输出、池化层输入图像的深度
        state_p = np.zeros((m_p, n_p, self.pLyNum))  #池化层输出图像的深度
        for n in range(self.cLyNum):  
            state_c[:,:,n] = self.convolution(data, self.kernel_c[:,:,n])  #卷积
            tmp_bias = np.ones((m_c,n_c)) * self.cLyBias[:,n]  # 图像矩阵的每一位都加上偏置
            state_c[:,:,n] = np.tanh(state_c[:,:,n] + tmp_bias)# 加上偏置项然后过激活函数  
            state_p[:,:,n] = self.pooling(state_c[:,:,n],self.pooling_a) #池化层输出压缩过的图像
        #全连接层 
        #m_p = n_p = 12, F_NUM = 12, 12-12+1 = 1
        #state_f为1*1*fLyNum，state_f_pre为m_c*n_c*fLyNum
        # 变成1维才能作为普通神经网络的输入，进入softmax层
        state_f, state_f_pre = self.convolution_f1(state_p,self.kernel_f, self.weight_f)  
        #进入激活函数  
        state_fo = np.zeros((1,self.fLyNum))#全连接层经过激活函数的结果      
        for n in range(self.fLyNum):  
            state_fo[:,n] = np.tanh(state_f[:,:,n] + self.fLyBias[:,n])  
        #进入softmax层  
        output = self.softmax_layer(state_fo)  
        #计算误差  
        y_pre = output.argmax(axis=1)  

        # 如果预测完了还要进行训练
        if ylab != None:
            err = -output[:,ylab]                 
            #print output     
            #计算误差         
            #print err   
            self.cnn_upweight(err,ylab,data,state_c,state_p, state_fo, state_f_pre, output)  
            # print self.kernel_c  
            # print self.cLyBias  
            # print self.weight_f  
            # print self.kernel_f  
            # print self.fLyBias  
            # print self.weight_output   
        return y_pre