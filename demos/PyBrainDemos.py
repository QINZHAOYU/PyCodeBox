# -*- encoding: utf-8 -*-
# -------------------------------------------------------------
'''
@Copyright :  Copyright(C) 2021, Qin ZhaoYu. All rights reserved. 

@Author    :  Qin ZhaoYu   
@See       :  https://github.com/QINZHAOYU

@Desc      :  PyBrain 框架学习案例（三峡梯调系统科学计算平台项目）。 

@Note      :  http://www.pybrain.org/docs/index.html
              https://blog.csdn.net/m0_37887016/article/details/70175493?spm=1001.2014.3001.5502

Change History:
---------------------------------------------------------------
v1.0, 2022/01/20, Qin ZhaoYu, zhaoyu.qin@foxmail.com
Init model.
'''
# -------------------------------------------------------------
import matplotlib
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt

from pybrain.tools.shortcuts import buildNetwork 
from pybrain.structure import TanhLayer, SoftmaxLayer   # 激活函数
from pybrain.datasets import SupervisedDataSet          # 监督学习数据包
from pybrain.supervised.trainers import BackpropTrainer # 反馈训练器


def IntroTopo1():
    ''' pybrain的神经网络构建、初始化、激活过程。
    '''
    # 构建一个具有两个输入，三个隐藏和一个输出神经元的网络。
    # PyBrain中，这些图层是Module对象，它们已经连接到FullConnection对象。
    # 网络已经用随机值初始化。
    net = buildNetwork(2, 3, 1)

    # 查看网络结构。
    print("net['in'] : {}".format(net['in']))            # default is Linear Layer.
    print("net['hidden0'] : {}".format(net['hidden0']))  # default is Sigmoid Layer.
    print("net['hidden0'] : {}".format(net['hidden0']))
    print("net['hidden0'] : {}".format(net['hidden0']))
    print("net['out'] : {}".format(net['out']))          # default is Linear Layer.

    # 激活(预报)。
    pred = net.activate([2, 1])
    print("predication: {}".format(pred))


def IntroTopo2():
    '''pybrain的神经网络激活函数定制，设置偏置神经元。

    偏置神经元（阈值），用以增加拟合的灵活度、控制神经元激活状态。

    (x1)            (a1)             (y1)
    (x2)            (a2)             (y2)
    input(layer0)   hidden(layer1)   out(layer2)

    各层之间为全连接，权重为w，则有:
    a1 = w^1_11 * x1 + w^1_21 * x2
    a2 = w^1_12 * x1 + w^1_22 * x2
    y1 = w^2_11 * a1 + w^2_21 * a2
    y2 = w^2_12 * a1 + w^2_22 * a2

    增加偏置神经元（深度学习中每个信号称为一个 “神经元”）:
    (x1)            (a1)             (y1)
    (x2)            (a2)             (y2)
    (b0)            (b1)
    input(layer0)   hidden(layer1)   out(layer2)  

    各层之间转为非全连接，偏置神经元的值为1，权重为b，则有:
    a1 = w^1_11 * x1 + w^1_21 * x2 + b0_1
    a2 = w^1_12 * x1 + w^1_22 * x2 + b0_2
    y1 = w^2_11 * a1 + w^2_21 * a2 + b1_1
    y2 = w^2_12 * a1 + w^2_22 * a2 + b1_2  
                                     bias
    '''
    # hiddenclass, 指定隐藏层激活函数的类型。
    # outclass，指定输出层激活函数类型。
    # bias，指定是否增加偏置神经元。
    net = buildNetwork(2, 3, 2, hiddenclass=TanhLayer, outclass=SoftmaxLayer, bias=True)

    # 查看网络结构。
    print("net['in'] : {}".format(net['in']))            
    print("net['hidden0'] : {}".format(net['hidden0']))  
    print("net['hidden0'] : {}".format(net['hidden0']))
    print("net['hidden0'] : {}".format(net['hidden0']))
    print("net['out'] : {}".format(net['out']))    

    # 激活
    pred = net.activate([2, 3])
    print("predication: {}".format(pred))      


def BuildDataSet():
    '''构建并查看监督学习的数据集。
    '''
    # 定义一个支持二维输入和一维标注信息的数据集。
    ds = SupervisedDataSet(2, 1)

    # 加入实例（以 XOR函数为例）。
    ds.addSample([0, 0], [0, ])
    ds.addSample((0, 1), (1, )) # two formats are both ok.   
    ds.addSample([1, 0], [1, ])
    ds.addSample([1, 1], [0, ])

    # 查看数据集
    print("========= length of ds: ", len(ds))
    print("========= read ds in for:")
    for inpt, obj in ds:
        print(inpt, obj)
    print("========= ds[input]: ")
    print(ds['input'])
    print("========= ds[target]: ")
    print(ds['target'])

    # 清空数据集
    ds.clear()
    print("========= ds is cleared: ")
    print("========= ds[input]: ")
    print(ds['input'])
    print("========= ds[target]: ")
    print(ds['target'])


def TrainModel():
    '''反馈训练，训练监督学习模型。
    '''
    # 构建神经网络
    net = buildNetwork(2, 3, 1, hiddenclass=TanhLayer, bias=True)

    # 构建简单学习数据集
    ds = SupervisedDataSet(2, 1)
    ds.addSample([0, 0], [0, ])
    ds.addSample([0, 1], [1, ]) 
    ds.addSample([1, 0], [1, ])
    ds.addSample([1, 1], [0, ])

    # 构建训练器(BackpropTrainer, 误差反向传播训练器)
    trainer = BackpropTrainer(net, ds)

    # 训练(训练一个完整时期的网络，并返回一个误差值)
    err = trainer.train()
    print("========= train error: ", err)

    # 训练(训练网络直到网络收敛，并返回每个训练周期的误差元组)
    # 若误差数组的每个元素是逐渐减小的，说明网络逐渐收敛。
    res = trainer.trainUntilConvergence()
    print("========= trainUntilConvergence errors: ", res)

    # 预报
    pred = net.activate([0,1])
    print("========= input [0, 1], predication: ", pred)


from pybrain.tools.shortcuts import FeedForwardNetwork  # 前馈网络
from pybrain.structure import LinearLayer, SigmoidLayer # 激活函数
from pybrain.structure import FullConnection            # 全连接层


def BuildCustomNet():
    '''自定义网络结构的前馈网络。
    '''
    # 声明一个前馈网络对象
    net = FeedForwardNetwork()

    # 自定义输入、隐藏、输出层模块（module）
    inLayer = LinearLayer(2, name="inLayer")       # 接受二维输入数据的线性层
    hiddenLayer = SigmoidLayer(3, name="hiddenLayer")  # 三层Sigmoid隐藏层
    outLayer = LinearLayer(1, name="outLayer")      # 输出一维数据的输出层

    # 将自定义模块注入网络
    # addInputModule()， 加载输入层
    # addModule()， 加载隐藏层
    # addOutputModule()，加载输出层
    net.addInputModule(inLayer)
    net.addModule(hiddenLayer)
    net.addOutputModule(outLayer)

    # 自定义不同模块（层）之间的连接方式(connection)
    in_to_hidden = FullConnection(inLayer, hiddenLayer)
    hidden_to_out = FullConnection(hiddenLayer, outLayer)

    # 将自定义连接注入网络
    net.addConnection(in_to_hidden)
    net.addConnection(hidden_to_out)

    # 激活，使网络可用
    net.sortModules()
    pred = net.activate([2, 3])
    print("======= predication: ", pred)

    # 查看网络
    print(net)
    print(in_to_hidden.params)
    print(hidden_to_out.params)
    print(net.params)


from pybrain.datasets import ClassificationDataSet      # 分类学习数据包
from pybrain.utilities import percentError              # 以列表和数组的形式返回百分比误差的工具包

from pylab import ion, ioff, figure, draw, contourf, clf, show, plot, legend
from scipy import arange, meshgrid, where
from numpy.random import multivariate_normal
from numpy import diag


class ClassificationDemo():
    '''神经网络分类器。
    '''

    def __init__(self):
        # 一个具有二维空间点的列表
        self.means = [(-1, 0), (2, 4), (3, 1)]

        # 一个由对角矩阵形成的列表(数据集的划分为三类)
        self.cov = [diag([1, 1]), diag([0.5, 1]), diag([1.5, 0.7])]
        print("====== cov: ", self.cov)

    def geneDataSet(self):
        '''生成输入数据集。
        '''
        # 定义输入数据集，输入数据是二维，目标是一维，数据可被分为三类
        self.data = ClassificationDataSet(2, 1, nb_classes=3)
        for n in range(400):
            for k in range(3):
                # 把输入数据作正态分布处理，以获得更好的数据集
                ds = multivariate_normal(self.means[k], self.cov[k])
                self.data.addSample(ds, [k])

    def splitDataSet(self):
        '''划分测试集和训练集。
        '''    
        testData_temp, trainData_temp = self.data.splitWithProportion(0.25)
        
        # 划分测试集
        self.testData = ClassificationDataSet(2, 1, nb_classes=3)
        for n in range(0, testData_temp.getLength()):
            self.testData.addSample(testData_temp.getSample(n)[0], 
            testData_temp.getSample(n)[1])

        # 划分训练集
        self.trainData = ClassificationDataSet(2, 1, nb_classes=3)
        for n in range(0, trainData_temp.getLength()):
            self.trainData.addSample(trainData_temp.getSample(n)[0], 
            trainData_temp.getSample(n)[1])

        # 将目标数据降为一维，并转存在‘class’字段中
        self.testData._convertToOneOfMany()
        self.trainData._convertToOneOfMany()
    
    def showDataSet(self):
        '''展示数据集信息。
        '''
        print("Number of training data: ", len(self.trainData))
        print("Input and output dimension: ", self.trainData.indim, 
        self.trainData.outdim)
        print("First sample (input, target, class):")
        print(self.trainData["input"][0], self.trainData["target"][0], 
        self.trainData["class"][0])

    def buildNetwork(self):
        '''构建前馈神经网络。
        '''
        inNum = self.trainData.indim
        outNum = self.trainData.outdim
        self.fnn = buildNetwork(inNum, 5, outNum, outclass=SoftmaxLayer)

    def buildTrainerBP(self):
        '''构建反向误差训练器（反馈训练）训练网络。
        '''
        self.trainer = BackpropTrainer(self.fnn, self.trainData, 
        momentum=0.1, verbose=True, weightdecay=0.01)

    def geneGridData(self):
        '''美化成果展示。 
        '''
        ticks = arange(-3.0, 6.0, 0.2)
        self.X, self.Y = meshgrid(ticks, ticks)

        self.gridData = ClassificationDataSet(2, 1, nb_classes=3)
        for i in range(self.X.size):
            self.gridData.addSample([self.X.ravel()[i], self.Y.ravel()[i]], [0])

    def train(self):
        '''训练网络。
        '''
        for i in range(20):
            # 训练网络一次以获得每次训练结果
            self.trainer.trainEpochs(1)
            trainRes = percentError(self.trainer.testOnClassData(), 
            self.trainData["class"])
            testRes = percentError(self.trainer.testOnClassData(dataset=self.testData), 
            self.testData["class"])

            # 打印每次训练的错误率
            print("epoch : {}".format(self.trainer.totalepochs), end="  ")
            print("train error: {}".format(trainRes), end="  ")
            print("test error: {}".format(testRes))

            # 预测
            out = self.fnn.activateOnDataset(self.gridData)




if __name__ == "__main__":
    # IntroTopo1()
    # IntroTopo2()
    # BuildDataSet()
    # TrainModel()
    # BuildCustomNet()

    demo = ClassificationDemo()

    demo.geneDataSet()
    demo.splitDataSet()
    demo.showDataSet()
    
    demo.buildNetwork()
    demo.buildTrainerBP()
    demo.geneGridData()
    demo.train()


