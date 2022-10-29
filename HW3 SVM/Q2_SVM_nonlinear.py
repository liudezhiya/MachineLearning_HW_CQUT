# -*- coding: utf-8 -*-

# @File    : Q2_SVM_nonlinear.py
# @Date    : 2022-10-29 9:34  
# @Author  : 刘德智
# @Describe  :

import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
import scipy.io as scio

def plot_predict(model,axes):
    x0s = np.linspace(axes[0],axes[1],100)
    x1s = np.linspace(axes[2],axes[3],100)
    x0, x1 = np.meshgrid(x0s,x1s)
    X = np.c_[x0.ravel(),x1.ravel()]
    y_pred = model.predict(X).reshape(x0.shape)
    y_decision = model.decision_function(X).reshape(x0.shape)
    plt.contour(x0,x1,y_pred,cmap=plt.cm.winter)
    plt.contour(x0,x1,y_decision,cmap=plt.cm.winter,alpha=0.2)
    plt.show()



data_path='D:\workspace\MachineLearning_HW_CQUT\HW3 SVM\data2.mat'
data= scio.loadmat(data_path)
print(data.keys())
data_x=data.get('X')#取出字典里的data  <class 'numpy.ndarray'>
data_y=data.get('y')#取出字典里的label  <class 'numpy.ndarray'>
data_x = data_x.astype('float32')
data_y = data_y.astype('float32')
X_train, X_test, y_train, y_test=train_test_split(data_x,data_y,test_size=0.3, random_state=42)
plt.scatter(data_x[:,0],data_x[:,1],c=data_y.flatten(),cmap="rainbow")##flatten()默认按行的方向降维
# plt.show()

k='rbf'
model_svm=svm.SVC(C=50,kernel=k,gamma=100,decision_function_shape='ovo')
model_svm.fit(X_train,y_train.ravel())



plt.scatter(X_train[:,0],X_train[:,1],c=y_train.flatten(),cmap="rainbow")

plot_predict(model_svm,[X_train[:,0].min(),X_train[:,0].max(),X_train[:,1].min(),X_train[:,0].max()])


from sklearn import  metrics
predict = model_svm.predict(X_test)
ac_score = metrics.accuracy_score(y_test, predict)
print('kernel={}时准确率为：{}'.format(k,ac_score))

