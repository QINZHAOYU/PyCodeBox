# -*- encoding: utf-8 -*-
# -------------------------------------------------------------
'''
@Copyright :  Copyright(C) 2021, Qin ZhaoYu. All rights reserved. 

@Author    :  Qin ZhaoYu   
@See       :  https://github.com/QINZHAOYU

@Desc      :  最小二乘法案例。  

Change History:
---------------------------------------------------------------
v1.0, 2022/01/19, Qin ZhaoYu, zhaoyu.qin@foxmail.com
Init model.
'''
# -------------------------------------------------------------
import numpy as np
from numpy.linalg import lstsq  # least square method
import pandas as pd 
from pandas import DataFrame
import os
from datetime import datetime
import json

import matplotlib
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt


def demo1():
    ''' y = k * x + b '''
    x = np.array([0, 1, 2, 3])
    y = np.array([-1, 0.2, 0.9, 2.1])

    '''
    | x1 1 |            | y1 | 
    | x2 1 |    | k |   | y2 |  
    | x3 1 |  * | b | = | y3 | 
    | x4 1 |            | y4 | 
       A          X       B
    '''
    A = np.vstack([x, np.ones(len(x))]).T
    B = y

    coeff, err, rank, singulars = lstsq(A, B, rcond=None)
    print(coeff)
    print("residual sum of squares: {}, rank: {}".format(err, rank))

    k, b = coeff
    y2 = [k*i + b for i in x]
    plt.plot(x, y, ls="-", lw=2, c="red", label="measure")
    plt.plot(x, y2, ls="-", lw=2, c="green", label="simu")
    plt.legend()
    plt.show()


def demo2():
    ''' 
    y1 = k1 * x + b1 
    y2 = k2 * x + b2
    '''
    x = np.array([0, 1, 2, 3])
    y1 = np.array([-1, 0.2, 0.9, 2.1])
    y2 = [y+2 for y in y1]

    '''
    | x1 1 |                | y1_1  y2_1| 
    | x2 1 |    | k1 k2 |   | y1_2  y2_2|  
    | x3 1 |  * | b1 b2 | = | y1_3  y2_3| 
    | x4 1 |                | y1_4  y2_4| 
       A            X             B
    '''
    A = np.vstack([x, np.ones(len(x))]).T
    B = np.vstack([y1, y2]).T  # two columns.

    coeff, err, rank, singulars = lstsq(A, B, rcond=None)
    print(coeff, err, rank, singulars)
    print("residual sum of squares: {}, rank: {}".format(err, rank))

    k, b = coeff[0][0], coeff[1][0]  # first column as coeff.
    y1_2 = [k*i + b for i in x] 
    plt.plot(x, y1, ls=':', lw=2, c='b', marker='o', markerfacecolor='r', label="measure")
    plt.plot(x, y1_2, ls='-', lw=2, c='r',  label="simu1")
    plt.legend()
    plt.show()

    k, b = coeff[0][1], coeff[1][1] # second column as coeff.
    y2_2 = [k*i + b for i in x] 
    plt.plot(x, y2, ls=':', lw=2, c='b', marker='o', markerfacecolor='r', label="measure")
    plt.plot(x, y2_2, ls='-', lw=2, c='r',  label="simu1")
    plt.legend()
    plt.show()


def demo3():
    ''' y = k1 * x^2 + k2 * x + b '''
    k1_true, k2_true, b_true = 3, 2, 1  

    x = np.linspace(-1, 1, 100)
    y_true = k1_true * x**2 + k2_true * x + b_true

    xi = 1 - 2 * np.random.rand(100)
    yi = k1_true * xi**2 + k2_true * xi + b_true + np.random.randn(100)

    '''
    | x1^2    x1    1 |    | k1 |   | y1   | 
    | x2^2    x2    1 |    | k2 |   | y2   |  
    | ............... |  * | b  | = | .... | 
    | x100^2  x100  1 |             | y100 | 
            A                X        B
    '''
    A = np.vstack([xi**2, xi**1, xi**0]).T
    B = yi

    coeff, err, rank, singulars = lstsq(A, B, rcond=None)
    print(coeff)
    print("residual sum of squares: {}, rank: {}".format(err, rank))

    k1, k2, b = coeff
    y2 = [k1 * i**2 + k2 * i + b for i in x]
    plt.plot(xi, yi, 'go', alpha=0.5, label="nosie")
    plt.plot(x, y_true, ls="-", lw=2, c="red", label="math")
    plt.plot(x, y2, ls="-", lw=2, c="green", label="simu")
    plt.legend()
    plt.show()


class RsvHydroModel():
    '''水库出流预报模型（最小二乘法拟合的线性模型）。 
    '''

    def __init__(self):
        self.dataColus = ["date", "outflow", "inflow", "outflow_yd", "rwlevel"] 
        self.resuColus = ["v0", "v1", "v2", "v3", "error", "s0", "s1", "s2", "s3"]        
        self.data = DataFrame(columns=self.dataColus)
        self.resu = DataFrame(columns=self.resuColus)
        self.fileTypes = [".XLS", ".XLSX", ".CSV"]

    def LoadData(self, DataFile:"excel or csv file path"):
        '''从文件加载待拟合数据集。 

        注意: Excel 文件将只读第一个表单。
        '''
        _FileType = self.__checkInputFile(DataFile)
        if _FileType == ".CSV":
            self.data = self.__loadCsvFile(DataFile)
        else:
            self.data = self.__loadExcelFile(DataFile)

        # 清除无效数据(存在NaN的数据行)
        self.data = self.data.dropna(how='any')
       
        # 添加常数项列
        self.data["i"] = np.array([1]*len(self.data))

    def ShowDataSetInfo(self, data:DataFrame):
        '''展示数据集的基本信息。 
        '''
        print("============== head:")        
        print(data.head())
        print("============== index:")
        print(data.index.values)
        print("============== columns:")
        print(data.columns.values)
        print("============== info")
        print(data.info())
        print("==============")

    def __checkInputFile(self, DataFile) -> str:
        '''检查输入数据文件是否合法并返回文件类型。 
        '''
        # 检查文件是否正确
        if not os.path.exists(DataFile):
            raise IOError("File or folder {} not existed.".format(DataFile))
        if os.path.isdir(DataFile):
            raise IOError("{} is not a file.".format(DataFile))
        _path, _file = os.path.split(DataFile)
        _name, _type = os.path.splitext(_file)
        if _type.upper() not in self.fileTypes:
            raise IOError("{} is not a excel or csv file.".format(DataFile))
        
        return _type.upper()

    def __loadExcelFile(self, DataFile) -> DataFrame:
        '''从excel文件加载数据集。
        '''
        _data = pd.read_excel(DataFile, 
        sheet_name=0, 
        skiprows=[1,],        
        header=[0,],
        usecols=[1, 2, 3, 4, 6],        
        names=self.dataColus,
        index_col=[0,],
        # nrows=355
        )

        return _data

    def __loadCsvFile(self, DataFile) -> DataFrame:
        '''从csv文件加载数据集。
        '''
        _data = pd.read_csv(DataFile,
        sep=',|\t',
        skiprows=[1,],        
        header=0,
        usecols=[1, 2, 3, 4, 6],
        names=self.dataColus,
        index_col=[0,],
        # nrows=355
        )
        return _data

    def FitByLstsq(self, IsSave=True, OutFile:"json file"=None) -> tuple:
        '''通过最小二乘法拟合水库模型的系数，拟合结果存入表中。

        Q_out = v0 * Q_out_yd + v1 * Q_in + v2 * Z + V3 

        Returns:
        + tuple of coeffs, err, rank, singulars.
        '''
        A = self.data[["outflow_yd", "inflow", "rwlevel", "i"]]
        B = self.data["outflow"] 
        currResu = lstsq(A.values, B.values, rcond=None)     

        # 存入表中
        if IsSave:       
            dates = B.index.values
            resuIndex = self.__GeneIndexOfDataframe(dates[0], dates[-1])

            coeffs, err, _, singulars = currResu
            currResuData = np.concatenate((coeffs, err, singulars))
            
            self.__SaveResuToDataframe(resuIndex, currResuData)
            self.__SaveResuToFile(resuIndex, currResuData, OutFile)

        return currResu

    def __GeneIndexOfDataframe(self, StartDate, EndDate) -> str:
        '''生成当前拟合结果的索引。
        '''
        dates_0 = str(StartDate).split("T")[0]
        dates_0 = dates_0.replace("-", "")
        dates_1 = str(EndDate).split("T")[0]
        dates_1 = dates_1.replace("-", "") 
        return dates_0 + "_" + dates_1

    def __SaveResuToDataframe(self, index, resu):
        '''将拟合结果保存到表中。 
        '''
        self.resu.loc[index] = resu

    def __SaveResuToFile(self, index, resu, OutFile):
        '''将拟合结果输出到json文件。
        '''
        # 检查文件是否合法
        if not OutFile:
            raise IOError("Empty output file.")
        _path, _file = os.path.split(OutFile)        
        if not os.path.exists(_path):
            raise IOError("{} is not existed.".format(_path))
        if os.path.isdir(OutFile):
            raise IOError("{} is no file.".format(OutFile))
        _name, _type = os.path.splitext(_file)
        if _type.upper() != ".JSON":
            print("output file({}) would be turned to .json file.".format(OutFile))
        OutFile = os.path.join(_path, _name + ".json")

        # 将python对象写入json文件
        _data = dict(zip(self.resuColus, resu))
        _resu = {index:_data}
        with open(OutFile, "w", encoding="utf8") as outer:
            json.dump(_resu, outer)

    def LoadModel(self, ModelFile:"model parameters file") -> tuple:
        '''从文件中加载水库出流预报模型参数。 

        Returns:
        + tuple of model parameters(v0, v1, v2, v3).
        '''
        # 检查文件合法性。
        if not os.path.exists(ModelFile):
            raise IOError("{} is not existed.".format(ModelFile))
        if os.path.isdir(ModelFile):
            raise IOError("{} is not a file.".format(ModelFile))
        _path, _file = os.path.split(ModelFile)
        _name, _type = os.path.splitext(_file)
        if _type.upper() != ".JSON":
            raise IOError("{} is not a json file.".format(modelFile))
        
        # 加载json文件读取模型参数。
        with open(ModelFile, "r", encoding="utf8") as reader:
            model = json.load(reader)        
        model = list(model.values())[0]
        params = (model["v0"], model["v1"], model["v2"], model["v3"])
        return params        

    def Predicate(self, model:"reservoir model parameters", 
    InData:"reservoir dataframe", 
    LastQout:"outflow in last day") -> list:
        '''通过水库出流预报模型预报出流。

        Q_out = v0 * Q_out_yd + v1 * Q_in + v2 * Z + V3 

        Args:
        + model: reservoir hydrology model parameters, formated:
            [v0, v1, v2, v3]
        + InData: reservoir weather forecast data, formated as:
            "    inflow   rwlevel  "
            " 0  q1       z1       "
            " ...                  "
        + LastQout: reservoir outflow in yesterday.

        Returns:
        + list of predication of reservoir outflow.
        '''
        # 基本参数
        lastOutflow = LastQout
        v0, v1, v2, v3 = model

        # 预报水库出流
        res = []
        for index, row in InData.iterrows():
            inflow, z = row
            outflow = v0 * lastOutflow + v1 * inflow + v2 * z + v3
            lastOutflow = outflow
            res.append(outflow)
        return res

    def plot(self, **series):
        '''统一绘图。
        '''
        for key, data in series.items():
            x = range(len(data))
            plt.plot(x, data, ls="-", lw=2, label=str(key))
        plt.legend()
        plt.show()        





if __name__ == "__main__":
    # demo1()
    # demo2()
    # demo3()

    path_xlsx = "C:\\Users\\gr\\Desktop\\长江三峡科学计算系统平台\\lstsq\\某一水库数据.xlsx"
    path_csv = "C:\\Users\\gr\\Desktop\\长江三峡科学计算系统平台\\lstsq\\某一水库数据.csv"
    out_json = r"C:\Users\gr\Desktop\长江三峡科学计算系统平台\lstsq\model.json"
    model = RsvHydroModel()
    model.LoadData(path_xlsx)
    orig_outflow = model.data["outflow"]

    res = model.FitByLstsq(True, out_json)
    print(res)

    param = model.LoadModel(out_json)
    print(param)

    inData = model.data[["inflow", "rwlevel"]]
    LastQout = model.data.iat[0, 2]
    pred_outflow = model.Predicate(param, inData, LastQout)
    model.plot(orig=orig_outflow, pred=pred_outflow)




