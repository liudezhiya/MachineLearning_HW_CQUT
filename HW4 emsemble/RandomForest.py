# -*- coding: utf-8 -*-

# @File    : RandomForest.py
# @Date    : 2022-10-31 16:24  
# @Author  : 刘德智
# @Describe  :#随机森林

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
#输入数据集
# data1.mat为分类数据集，每一行为一个样本，前两列为特征，最后一列为目标值。按照7:3的比率划分训练集和验证集。
data = loadmat('D:\workspace\MachineLearning_HW_CQUT\HW4 emsemble\data1.mat')
# print(data.items())
X = data['X']
y = data['y'].flatten()
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.7,test_size=0.3,random_state=42)
# 1.2 模型训练（10）
# 使用sklearn工具包，调用ensemble.RandomForestClassifier接口对模型进行训练。
# 1.3 分析（30)
# 换用不同的                             ，分析其对于验证集准确率的影响。
clf = RandomForestClassifier(max_depth=2, random_state=42)
clf.fit(X, y)
acc = clf.score(X_test, y_test)
print('test_acc:',acc)


clf = RandomForestClassifier(max_depth=2, random_state=42)
clf.fit(X, y)
acc = clf.score(X_test, y_test)
print('test_acc:',acc)

fig=plt.figure()
ax=fig.add_subplot(2,2,1)

num=[]
acclist=[]
for nest in range(10,200,10):
    clf = RandomForestClassifier(n_estimators=nest,max_depth=5,min_samples_split=2, random_state=42)
    clf.fit(X, y)
    acc = clf.score(X_test, y_test)
    num.append(nest)
    acclist.append(acc)
plt.rcParams['font.sans-serif'] = 'SimHei'
ax.plot(num,acclist,'r-o')
plt.xlabel('n_estimators num')
plt.ylabel('acc')
plt.title('n_estimators对准确率的影响')



ax=fig.add_subplot(2,2,2)
num=[]
acclist=[]
for nest in range(1,20,1):
    clf = RandomForestClassifier(max_depth=nest)
    clf.fit(X, y)
    acc = clf.score(X_test, y_test)
    num.append(nest)
    acclist.append(acc)
plt.rcParams['font.sans-serif'] = 'SimHei'
ax.plot(num,acclist,'r-o')
plt.xlabel('max_depth num')
plt.ylabel('acc')
plt.title('max_depth对准确率的影响')


ax=fig.add_subplot(2,2,3)
num=['gini','entropy']
acclist=[]
clf = RandomForestClassifier(criterion='gini')
clf.fit(X, y)
acc = clf.score(X_test, y_test)
acclist.append(acc)
clf = RandomForestClassifier(criterion='entropy')
clf.fit(X, y)
acc = clf.score(X_test, y_test)
acclist.append(acc)
plt.rcParams['font.sans-serif'] = 'SimHei'
ax.plot(num,acclist,'r-o')
plt.xlabel('Criterion')
plt.ylabel('acc')
plt.title('Criterion对准确率的影响')



ax=fig.add_subplot(2,2,4)
num=[]
acclist=[]
for nest in range(2,20,1):
    clf = RandomForestClassifier(min_samples_split=nest)
    clf.fit(X, y)
    acc = clf.score(X_test, y_test)
    num.append(nest)
    acclist.append(acc)
plt.rcParams['font.sans-serif'] = 'SimHei'
ax.plot(num,acclist,'r-o')
plt.xlabel('min_samples_split num')
plt.ylabel('acc')
plt.title('min_samples_split对准确率的影响')

plt.tight_layout()
plt.show()











# 3 Bonus（20）
# 3.1 使用Iris数据集分别对adaboost和随机森林进行训练。
# Iris也称鸢尾花卉数据集，是一类多重变量分析的数据集。
# 数据集包含150个数据样本，分为3类，每类50个数据，每个数据包含4个属性。
# 可通过花萼长度，花萼宽度，花瓣长度，花瓣宽度4个属性预测鸢尾花卉属于（Setosa，Versicolour，Virginica）三个种类中的哪一类。
# Iris数据集的调用