# -*- coding: utf-8 -*-

# @File    : Logitstic Regression.py
# @Date    : 2022-09-28
# @Author  : 刘德智
# @Describe  :逻辑回顾
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# loading data
data = np.loadtxt('../data2.txt',delimiter=',')
print(data.shape)
num_feature = data.shape[1] - 1
data = data.astype('float32')

# data normalization
data_ori = data.copy()
maximum = np.max(data[:, :num_feature],axis=0,keepdims=True)
minimun = np.min(data[:, :num_feature],axis=0,keepdims=True)
data[:, :num_feature] = (data[:, :num_feature] - minimun)/(maximum - minimun)

# train val split
data_train, data_test = train_test_split(data, test_size=0.3, random_state=42)
X_train = data_train[:, :2]
X_train = np.concatenate((X_train, np.ones((X_train.shape[0],1))), axis=1)
y_train = data_train[:, 2].reshape(-1,1)
X_test = data_test[:, :2]
X_test = np.concatenate((X_test, np.ones((X_test.shape[0],1))), axis=1)
y_test = data_test[:, 2].reshape(-1,1)

# model init
w = np.zeros((num_feature+1,1))

def cross_entropy_loss(y_pred,y):
    return -np.mean(y*np.log(y_pred)+(1-y)*np.log(1-y_pred))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

iterations = 10000
lr = 0.1

log = []
# gradient descent
for i in range(iterations):
    y_pred = sigmoid(np.matmul(X_train, w))
    g = lr*np.mean((y_pred-y_train)*X_train, axis=0).reshape(-1,1)
    w -= g
    loss = cross_entropy_loss(y_pred,y_train)
    print('iter:{},loss:{}'.format(i,loss))
    log.append([i,loss])
    y_pred_test = sigmoid(np.matmul(X_test, w))
    loss = cross_entropy_loss(y_pred_test,y_test)
#     print('iter:{},val_loss:{}'.format(i,loss))

# loss curve visualization
log = np.array(log)
plt.figure('1')
plt.plot(log[:,0],log[:,1])

# visualization
plt.figure('2')
plt.scatter(X_train[:,0],X_train[:,1],c=y_train.flatten())
x = np.linspace(0,1,10)
y = (- w[0]*x - w[2])/w[1]
plt.plot(x, y)

plt.figure('3')
plt.scatter(X_test[:,0],X_test[:,1],c=y_test.flatten())
x = np.linspace(0,1,10)
y = (- w[0]*x - w[2])/w[1]
plt.plot(x, y)

plt.show()