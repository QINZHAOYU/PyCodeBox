# -*- encoding: utf-8 -*-
# -------------------------------------------------------------
'''
@Copyright :  Copyright(C) 2021, Qin ZhaoYu. All rights reserved. 

@Author    :  Qin ZhaoYu   
@See       :  https://github.com/QINZHAOYU

@Desc      :  None  

Change History:
---------------------------------------------------------------
v1.0, 2021/11/30, Qin ZhaoYu, zhaoyu.qin@foxmail.com
Init model.
'''
# -------------------------------------------------------------

'''============================================================ 1. 数据选择--导入波士顿房价数据集
'''
from keras.datasets import boston_housing 

(train_x, train_y), (test_x, test_y) = boston_housing.load_data()


'''============================================================ 2. 特征工程--探索性数据分析
'''
import pandas as pd 
import pandas_profiling

# 特征名称
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
train_df = pd.DataFrame(train_x, columns=feature_name)  
report = pandas_profiling.ProfileReport(train_df)  
# report.to_file('./report.html')


'''============================================================ 3. 模型训练
'''

# 3.1 创建模型结构、批标准化、正则化
# 
# 结合当前房价的预测任务是一个经典简单表格数据的回归预测任务。
# 我们采用基础的全连接神经网络，隐藏层的深度一两层也就差不多。
# 通过 `keras.Sequential` 方法来创建一个神经网络模型，并在依次添加带有批标准化的输入层，
# 一层带有relu激活函数的k个神经元的隐藏层，并对这层隐藏层添加dropout、L1、L2正则的功能。
# 由于回归预测数值实际范围（5~50+）直接用线性输出层，不需要加激活函数。
#
# 模型结构：输入层的特征维数为 3；1层 k个神经元的 relu隐藏层；线性的输出层。
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import random
from keras import regularizers
from keras.layers import Dense,Dropout,BatchNormalization
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping
from sklearn.metrics import  mean_squared_error

np.random.seed(1) # 固定随机种子，使每次运行结果固定
random.set_seed(1)

for k in [5,20,50]:  # 网格搜索超参数：神经元数k    
    model = Sequential()
    model.add(BatchNormalization(input_dim=13))  # 输入层 批标准化 
    model.add(Dense(k,  
                    kernel_initializer='random_uniform',   # 均匀初始化
                    activation='relu',                     # relu激活函数
                    kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01),  # L1及L2正则项
                    use_bias=True))    # 隐藏层
    model.add(Dropout(0.1)) # dropout法
    model.add(Dense(1,use_bias=True))  # 输出层

# 3.2 选择学习目标
#
# 设定学习目标为（最小化）回归预测损失mse，优化算法为adam
model.compile(optimizer='adam', loss='mse') 


# 3.3 模型训练
# 
# 通过传入训练集x，训练集标签y，使用fit（拟合）方法来训练模型，其中epochs为迭代次数，  
# 并通过EarlyStopping及时停止在合适的epoch，减少过拟合；
# batch_size为每次epoch随机采样的训练样本数目。
#
# 最后，这里简单采用for循环，实现类似网格搜索调整超参数，验证了隐藏层的不同神经元数目（超参数k）的效果。
# 由验证结果来看，神经元数目为50时，损失可以达到10的较优效果
# （可以继续尝试模型增加深度、宽度，达到过拟合的边界应该有更好的效果）。
history = model.fit(train_x, 
                    train_y, 
                    epochs=500,             # 训练迭代次数
                    batch_size=50,          # 每epoch采样的batch大小
                    validation_split=0.1,   # 从训练集再拆分验证集，作为早停的衡量指标
                    callbacks=[EarlyStopping(monitor='val_loss', patience=20)],    #早停法
                    verbose=False)          # 不输出过程  

print("验证集最优结果：",min(history.history['val_loss']))
model.summary()   #打印模型概述信息

plt.plot(history.history['loss'],c='blue')    # 蓝色线训练集损失
plt.plot(history.history['val_loss'],c='red') # 红色线验证集损失
plt.show()


'''============================================================ 4. 模型评估：拟合效果
'''




