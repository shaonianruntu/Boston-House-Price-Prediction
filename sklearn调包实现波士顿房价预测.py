import os
import sys
import numpy as np
import pandas as pd
import sklearn

from sklearn.linear_model import LinearRegression   #线性回归模型
from sklearn.linear_model import Lasso              #带L1正则的线性回归模型
from sklearn.linear_model import Ridge              #带L2正则的线性回归模型
from sklearn.linear_model import ElasticNetCV       #带弹性网络
from sklearn.metrics import mean_squared_error      #mse

import matplotlib as mpl
import matplotlib.pyplot as plt


#对数据集中的样本属性进行分割，制作X和Y矩阵 
def feature_label_split(pd_data):
    row_cnt = pd_data.shape[0]
    column_cnt = len(pd_data.iloc[0,0].split())
    
    X = np.empty((row_cnt, column_cnt-1))
    Y = np.empty((row_cnt, 1))
    for i in range(0, row_cnt):
        row_array = pd_data.iloc[i,0].split()
        X[i] = np.array(row_array[0:-1])
        Y[i] = np.array(row_array[-1])
    
    return X, Y


#把特征数据进行标准化为均匀分布
def uniform_norm(X_in):
    X_max = X_in.max(axis=0)
    X_min = X_in.min(axis=0)
    X = (X_in-X_min)/(X_max-X_min)
    return X, X_max, X_min


#主函数
if __name__ == "__main__":

    #读取训练集和测试集文件
    train_data = pd.read_csv("boston_house.train", header=None)
    test_data = pd.read_csv("boston_house.test", header=None)

    #对训练集和测试集进行X，Y分离
    train_X, train_Y = feature_label_split(train_data)
    test_X, test_Y = feature_label_split(test_data)

    #对X（包括train_X, test_X）进行归一化处理，方便后续操作
    unif_trainX, X_max, X_min = uniform_norm(train_X)
    unif_testX = (test_X-X_min)/(X_max-X_min)

    #模型训练, 带L2正则的线性回归
    model = Ridge()
    model.fit(unif_trainX, train_Y)
    
    #模型评估
    print("训练集上效果评估 >>")
    r2 = model.score(unif_trainX, train_Y)
    print("R^2系数 ", r2)
    train_pred = model.predict(unif_trainX)
    mse = mean_squared_error(train_Y, train_pred) 
    print("均方误差 ", mse)

    print("\n测试集上效果评估 >>")
    r2 = model.score(unif_testX, test_Y)
    print("R^2系数 ", r2)
    test_pred = model.predict(unif_testX)
    mse = mean_squared_error(test_Y, test_pred)        
    #等价于 mse = sum((test_pred-test_Y)**2) / test_Y.shape[0]
    print("均方误差", mse)      


    #对测试集上的标注值与预测值进行可视化呈现   
    t = np.arange(len(test_pred))
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(facecolor='w')
    plt.plot(t, test_Y, 'r-', lw=2, label=u'true value')
    plt.plot(t, test_pred, 'b-', lw=2, label=u'estimated')
    plt.legend(loc = 'best')
    plt.title(u'Boston house price', fontsize=18)
    plt.xlabel(u'case id', fontsize=15)
    plt.ylabel(u'house price', fontsize=15)
    plt.grid()
    plt.show()

    