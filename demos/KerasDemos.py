# -*- encoding: utf-8 -*-
# -------------------------------------------------------------
'''
@Copyright :  Copyright(C) 2021, Qin ZhaoYu. All rights reserved. 

@Author    :  Qin ZhaoYu   
@See       :  https://github.com/QINZHAOYU

@Desc      :  To learn to use keras.  

Change History:
---------------------------------------------------------------
v1.0, 2021/11/30, Qin ZhaoYu, zhaoyu.qin@foxmail.com
Init model.
'''
# -------------------------------------------------------------

import numpy as np 
import pandas as pd 
import pandas_profiling as pdprof 
import matplotlib
matplotlib.use('TkAgg')  # add to show plt picture in vscode.
import matplotlib.pyplot as plt

import shap 
import tensorflow as tf  
from tensorflow import random
from keras import regularizers
from keras.layers import Dense,Dropout,BatchNormalization
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping
from keras.datasets import boston_housing 
from sklearn.metrics import  mean_squared_error



class boston_housing_demo():
    '''using keras to predict boston house price.
    '''    

    def __init__(self):
        self.train_x = None
        self.train_y = None
        self.test_x =  None
        self.test_y =  None
        self.history = None

    def step1_getData(self):
        '''1. 数据选择--导入波士顿房价数据集.
        '''
        (self.train_x, self.train_y), (self.test_x, self.test_y) = boston_housing.load_data()

    def step2_analyseData(self):
        '''2. 特征工程--探索性数据分析 
        '''
        # 特征项
        feature_name = [
            'CRIM|住房所在城镇的人均犯罪率',
            'ZN|住房用地超过 25000 平方尺的比例',
            'INDUS|住房所在城镇非零售商用土地的比例',
            'CHAS|有关查理斯河的虚拟变量（如果住房位于河边则为1,否则为0 ）',
            'NOX|一氧化氮浓度',
            'RM|每处住房的平均房间数',
            'AGE|建于 1940 年之前的业主自住房比例',
            'DIS|住房距离波士顿五大中心区域的加权距离',
            'RAD|距离住房最近的公路入口编号',
            'TAX 每 10000 美元的全额财产税金额',
            'PTRATIO|住房所在城镇的师生比例',
            'B|1000(Bk|0.63)^2,其中 Bk 指代城镇中黑人的比例',
            'LSTAT|弱势群体人口所占比例']

        # 转为df格式并输出数据报告
        train_df = pd.DataFrame(self.train_x, columns=feature_name)  
        report = pdprof.ProfileReport(train_df)  
        report.to_file('./report.html')

    def step3_train(self):
        '''3. 模型训练

        3.1 创建模型结构、批标准化、正则化

        结合当前房价的预测任务是一个经典简单表格数据的回归预测任务，采用基础的全连接神经网络。
        通过 `keras.Sequential` 方法创建一个顺序神经网络模型，依次添加带有批标准化的输入层，
        一层带有relu激活函数的k个神经元的隐藏层，并对这层隐藏层添加 `dropout、L1、L2` 正则功能。
        由于回归预测数值实际范围（5~50+）直接用线性输出层，不需要加激活函数。

        最终：输入层特征维数为 3；1层 k个神经元的 `relu` 隐藏层；线性的输出层。

        3.2 选择学习目标

        设定学习目标为（最小化）回归预测损失 `mse`，优化算法为 `adam`。 

        3.3 模型训练
        
        通过传入训练集x，训练集标签y，使用 `fit`（拟合）方法来训练模型，其中 `epochs` 为迭代次数，  
        并通过 `EarlyStopping` 及时停止在合适的epoch，减少过拟合；
        `batch_size` 为每次epoch随机采样的训练样本数目。
                
        最后，这里简单采用for循环，实现类似网格搜索调整超参数，验证了隐藏层的不同神经元数目（超参数k）的效果。
        由验证结果来看，神经元数目为50时，损失可以达到10的较优效果
        （可以继续尝试模型增加深度、宽度，达到过拟合的边界应该有更好的效果）。
        '''
        # 3.1 创建模型结构、批标准化、正则化
        np.random.seed(1)  # 固定np随机种子，使每次运行结果固定
        random.set_seed(1) # 固定tf随机种子
        
        for k in [5, 20, 100]:  # 网格搜索超参数：神经元数k的三种设置    
            self.model = Sequential()
            self.model.add(BatchNormalization(input_dim=13))  # 输入层 批标准化 
            self.model.add(Dense(k,  
                            kernel_initializer='random_uniform',   # 均匀初始化
                            activation='relu',                     # relu激活函数
                            kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01),  # L1及L2正则项
                            use_bias=True))    # 隐藏层
            self.model.add(Dropout(0.1)) # dropout法
            self.model.add(Dense(1,use_bias=True))  # 输出层

        # 3.2 选择学习目标
        self.model.compile(optimizer='adam', loss='mse') 

        # 3.3 模型训练
        self.history = self.model.fit(self.train_x, 
                            self.train_y, 
                            epochs=500,             # 训练迭代次数
                            batch_size=50,          # 每epoch采样的batch大小
                            validation_split=0.1,   # 从训练集再拆分验证集，作为早停的衡量指标
                            callbacks=[EarlyStopping(monitor='val_loss', patience=20)],    #早停法
                            verbose=False)          # 不输出过程  
        
        # 查看结果
        print("------- 验证集最优结果：",min(self.history.history['val_loss']))
        self.model.summary()   #打印模型概述信息     
        return self.history 

    def step4_evaluate(self):
        '''4. 模型评估
        '''
        plt.plot(self.history.history['loss'],c='blue',label='train_loss')    # 蓝色线训练集损失
        plt.plot(self.history.history['val_loss'],c='red',label='val_loss') # 红色线验证集损失
        plt.xlabel("iter times")
        plt.ylabel("loss val")
        plt.legend()
        plt.show()    

    def step5_predict(self):
        '''5. 模型预测
        '''    
        pred_y = self.model.predict(self.test_x)[:,0]

        print("------- 实际与预测值的差异：",mean_squared_error(self.test_y, pred_y))
        plt.plot(range(len(self.test_y)), self.test_y, ls='-.',lw=2,c='r',label='true_val')
        plt.plot(range(len(pred_y)), pred_y, ls='-',lw=2,c='b',label='pred_val')
        plt.legend()
        plt.show()

    def step6_explain(self):
        '''6. 模型解释

        通过SHAP方法，对模型预测单个样本的结果做出解释，可见在这个样本的预测中，
        CRIM犯罪率为0.006、RM平均房间数为6.575对于房价是负相关的。
        LSTAT弱势群体人口所占比例为4.98对于房价的贡献是正相关的...，
        在综合这些因素后模型给出最终预测值。
        '''        
        # 模型解释性
        background = self.test_x[np.random.choice(self.test_x.shape[0],100, replace=False)]
        explainer = shap.DeepExplainer(self.model, background)
        shap_values = explainer.shap_values(self.test_x)  # 传入特征矩阵X，计算SHAP值

        # 可视化第一个样本预测的解释  
        shap.force_plot(explainer.expected_value, shap_values[0,:], self.test_x.iloc[0,:])



if __name__ == "__main__":
    # run bostun hous price demo.
    boston = boston_housing_demo()
    print("------- boston-housing-demo --------")
    boston.step1_getData()
    print("------- data geted")
    # boston.step2_analyseData()
    # print("------- data analysised")
    boston.step3_train()
    print("------- model traned")
    boston.step4_evaluate()
    print("------- model evaluated")
    boston.step5_predict()
    print("------- predict price")
    # boston.step6_explain()
    # print("------- explain model")


