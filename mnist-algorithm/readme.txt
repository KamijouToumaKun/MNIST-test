mac解决six报错的问题：
sudo pip install six --upgrade --ignore-installed six

Decision Tree(2000 train, 10000 test): 0.65
（奇怪的是，训练数据为1000或10000时，准确率都会下降到0.4左右）

NaiveBayes: 0.843000

KNearest(10 neighbor, 1000 train, 1000 test): 0.834000
SKLEARN KNearestNeighbor: 0.9665

Perceptron(5 epoch): 0.887000 单层多类

BPNeuralNetwork(1 epoch, 15 hidden neuron): 
Sigmoid + Sigmoid: 0.915100 多层单类
Sigmoid + Softmax: 0.928200 多层多类
(Softmax + Softmax: 0.413700 很差)
SKLEARN BPNeuralNetwork(ReLU): 0.9745

SVM:TODO
SKLEARN SVM: 0.9446

CNN(1 epoch, 10000 train):
参考http://blog.csdn.net/u010866505/article/details/77857394
0.889800[Finished in 4443.6s]
