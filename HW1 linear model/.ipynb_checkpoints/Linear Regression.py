# -*- coding: utf-8 -*-

# @File    : Linear Regression.py
# @Date    : 2022-09-27
# @Author  : 刘德智
# @Describe  :线性回归

import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# loading data
# data = np.loadtxt('data1.txt',delimiter=',')
data = np.loadtxt('../data1.txt',delimiter=',')
print(data.shape)
num_feature = data.shape[1] - 1
data = data.astype('float32')

# data normalization
data_norm = data.copy()
maximum = np.max(data_norm,axis=0,keepdims=True)
print(maximum)
minimun = np.min(data_norm,axis=0,keepdims=True)
data_norm = (data_norm - minimun)/(maximum - minimun)
print(data_norm)

# train val split
data_train, data_test = train_test_split(data_norm, test_size=0.3, random_state=42)
X_train = data_train[:, :2]
X_train = np.concatenate((X_train, np.ones((X_train.shape[0],1))), axis=1)
y_train = data_train[:, 2]
X_test = data_test[:, :2]
X_test = np.concatenate((X_test, np.ones((X_test.shape[0],1))), axis=1)
y_test = data_test[:, 2]

# model init
w = np.random.rand(num_feature+1,1)

# gradient descent
def L2_loss(y_pred,y):
    return np.mean(np.square(y_pred-y))

iterations = 10000
lr = 0.1

log = []
for i in range(iterations):
    y_pred = np.matmul(X_train, w)
    term = lr*np.mean((y_pred-y_train.reshape(-1,1))*X_train, axis=0).reshape(-1,1)
    w -= term
    loss = L2_loss(y_pred,y_train)
    print('iter:{},loss:{}'.format(i,loss))
    log.append([i,loss])

# normal eqution
term = np.matmul(X_train.T, X_train)
term_inv = np.linalg.inv(term)
w = np.matmul(np.matmul(term_inv,X_train.T),y_train.reshape(-1,1))
print(w)

# loss curve visualization
log = np.array(log)
plt.plot(log[:,0],log[:,1])
plt.show()

# visualization
y_pred = np.matmul(X_test, w)
plt.scatter(X_test[:,0],y_pred,c='r')
plt.scatter(X_test[:,0],y_test,c='b')
plt.show()


plt.scatter(X_test[:,1],y_pred,c='r')
plt.scatter(X_test[:,1],y_test,c='b')
plt.show()
