# -*- coding: utf-8 -*-

# @File    : Q1_SvmModel.py
# @Date    : 2022-10-21 19:46  
# @Author  : 刘德智
# @Describe  :线性和非线性支持向量机
from scipy.io import loadmat
'''
import numpy as np
import matplotlib.pyplot as plt
#读取数据
dataSet = np.genfromtxt('ex2data2.txt',delimiter=',')
# print(dataSet)
# x1 = dataSet[:,2]
x_data = dataSet[:,:-1]
y_data = dataSet[:,-1]
def plot():
    plt.scatter(x_data[y_data==0,0],x_data[y_data==0,1],c='r',marker='*',label='label0')
    plt.scatter(x_data[y_data==1,0],x_data[y_data==1,1],c='b',marker='^',label='label1')
    plt.legend()
plot()
plt.show()
'''
from pandas import Series,DataFrame

import pandas as pd

import numpy as np

import h5py
'''
datapath = 'data1.mat'

file = h5py.File('data1.mat','r')
print(file)
'''
import scipy.io as scio

import pandas as pd


import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from scipy.io import loadmat
import pandas as pd
def plot_predict(model,axes):
    x0s = np.linspace(axes[0],axes[1],100)
    x1s = np.linspace(axes[2],axes[3],100)
    x0, x1 = np.meshgrid(x0s,x1s)
    X = np.c_[x0.ravel(),x1.ravel()]
    y_pred = model.predict(X).reshape(x0.shape)
    y_decision = model.decision_function(X).reshape(x0.shape)
    plt.contour(x0,x1,y_pred,cmap=plt.cm.winter)
    plt.contour(x0,x1,y_decision,cmap=plt.cm.winter,alpha=0.2)
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.title('软惩罚参数C:'+str(c))





data_path='D:\workspace\MachineLearning_HW_CQUT\HW3 SVM\data1.mat'
data = loadmat(data_path)
dfdata = pd.DataFrame(data=data['X'][:],columns=['d1', 'd2']).astype(str)
dfdatay = pd.DataFrame(data=data['y'][:],columns=['d3']).astype(str)
dfdata['d1'] = dfdata['d1'].map(lambda x: x.replace('[', '').replace(']', ''))
dfdata['d2'] = dfdata['d2'].map(lambda x: x.replace('[', '').replace(']', ''))
dfdata['d3'] = dfdatay['d3'].map(lambda x: x.replace('[', '').replace(']', ''))
print(dfdata)

# sklearn 库中导入 svm 模块
from sklearn import svm
import scipy.io as scio
#注意带路劲
data_path='D:\workspace\MachineLearning_HW_CQUT\HW3 SVM\data1.mat'
data= scio.loadmat(data_path)
print(data.keys())
data_x=data.get('X')#取出字典里的data  <class 'numpy.ndarray'>
data_y=data.get('y')#取出字典里的label  <class 'numpy.ndarray'>
data_x = data_x.astype('float32')
data_y = data_y.astype('float32')

plt.scatter(data_x[:,0],data_x[:,1],c=data_y.flatten(),cmap="rainbow")##flatten()默认按行的方向降维
# plt.show()

X_train, X_test, y_train, y_test=train_test_split(data_x,data_y,test_size=0.3, random_state=42)
'''
standardScaler = StandardScaler()
standardScaler.fit(X_train)
X_train = standardScaler.transform(X_train)
standardScaler.fit(X_test)
X_test = standardScaler.transform(X_test)
standardScaler.fit(y_train)
y_train = standardScaler.transform(y_train)
standardScaler.fit(y_test)
y_test = standardScaler.transform(y_test)
'''
# c=100
# clf = LinearSVC(C=c,max_iter=5000)
# clf.fit(X_train,y_train.ravel())
# plt.subplot(211)
cc=[1,100,1000,10000]
for i in range(len(cc)):
    tu=i+1
    plt.subplot(2,2,tu)
    c=cc[i]
    clf = LinearSVC(C=c, dual=False)
    clf.fit(X_train, y_train.ravel())
    plt.scatter(data_x[:, 0], data_x[:, 1], c=data_y.flatten())
    plot_predict(clf, [0, 4, 1, 5])
    from sklearn import svm, metrics

    predict = clf.predict(X_test)
    ac_score = metrics.accuracy_score(y_test, predict)
    cl_report = metrics.classification_report(y_test, predict)
    print(ac_score)
    print(cl_report)
plt.show()

from sklearn import svm, metrics
predict = clf.predict(X_test)
ac_score = metrics.accuracy_score(y_test, predict)
cl_report = metrics.classification_report(y_test, predict)
print(ac_score)
print(cl_report)
plt.scatter(data_x[:,0],data_x[:,1],c=data_y.flatten())
plot_predict(clf,[0,4,1,5])


'''
data_train, data_test = train_test_split(data_x, test_size=0.3, random_state=42)
data_train_y, data_test_y = train_test_split(data_y, test_size=0.3, random_state=42)
print(data_train,data_train_y)
plt.scatter(data_train[:,0],data_train[:,1],c=data_train_y,s=50,cmap="rainbow")
plt.scatter(data_test[:,0],data_test[:,1],c=data_test_y,s=50,cmap="rainbow")
plt.show()
clf = svm.LinearSVC()
clf.fit(data_train, data_train_y)
predict = clf.predict(data_test)
from sklearn import svm, metrics
ac_score = metrics.accuracy_score(data_test_y, predict)
cl_report = metrics.classification_report(data_test_y, predict)
print(ac_score)
print(cl_report)
'''
# from sklearn.preprocessing import StandardScaler
# standardScaler = StandardScaler()
# standardScaler.fit(data_train)
# data_train = standardScaler.transform(data_train)
# standardScaler.fit(data_test)
# data_test = standardScaler.transform(data_test)
# standardScaler.fit(data_train_y)
# data_train_y = standardScaler.transform(data_train_y)
# standardScaler.fit(data_test_y)
# data_test_y = standardScaler.transform(data_test_y)
# print(data_train_y)


# from sklearn.svm import LinearSVC
#
# clf = LinearSVC(loss='squared_hinge', penalty='l1', C=1)
# clf.fit(data_train, data_train_y)
# print("Predicted labels:", clf.predict(data_test))


'''
# 定义三个点和标签
X = [[2, 0], [1, 1], [2,3]]
y = [0, 0, 1]
# 定义分类器，clf 意为 classifier，是分类器的传统命名
clf = svm.SVC(kernel = 'linear')  # .SVC（）就是 SVM 的方程，参数 kernel 为线性核函数
clf = svm.LinearSVC()
# 训练分类器
clf.fit(X, y)  # 调用分类器的 fit 函数建立模型（即计算出划分超平面，且所有相关属性都保存在了分类器 cls 里）

# 打印分类器 clf 的一系列参数
print (clf)

# 支持向量
print (clf.support_vectors_)

# 属于支持向量的点的 index
print (clf.support_)

# 在每一个类中有多少个点属于支持向量
print (clf.n_support_)

# 预测一个新的点
print (clf.predict([[2,0]]))

'''
