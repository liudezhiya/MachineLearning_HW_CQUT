# -*- coding: utf-8 -*-

# @File    : test.py
# @Date    : 2022-11-01 14:32  
# @Author  : 刘德智
# @Describe  :
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from scipy.io import loadmat
from sklearn. model_selection import train_test_split
import numpy as np

iris=load_iris()
X=iris.data
y=iris.target

"*****************************bagging和随机森林***********************************"
bag_clf = BaggingClassifier(
                            SVC(),              #基分类器
                            n_estimators=500,   #基分类器有多少个 抽了500个每次用svm
                            bootstrap=True,     #有放回抽样
                            max_samples=1.0     #比例
                            )
bag_clf.fit(X,y)
y_hat = bag_clf.predict(X)
print(bag_clf.__class__.__name__,'==',accuracy_score(y,y_hat))


bag_clf = BaggingClassifier(
                            DecisionTreeClassifier(),              #基分类器
                            n_estimators=500,   #基分类器有多少个 抽了500个每次用svm
                            bootstrap=True,     #有放回抽样
                            max_samples=1.0     #比例
                            )
bag_clf.fit(X,y)
y_hat = bag_clf.predict(X)
print(bag_clf.__class__.__name__,'==',accuracy_score(y,y_hat))


rnd_clf = RandomForestClassifier()
rnd_clf.fit(X,y)
y_hat = rnd_clf.predict(X)
print(rnd_clf.__class__.__name__,'==',accuracy_score(y,y_hat))
"****************************************************************"
"*****************************adaboost***********************************"
data = loadmat('D:\workspace\MachineLearning_HW_CQUT\HW4 emsemble\data1.mat')
# print(data.items())
X = data['X']
y = data['y'].flatten()
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.7,test_size=0.3,random_state=42)
# 模型训练
# 使用sklearn工具包，调用ensemble.AdaBoostClassifier接口对模型进行训练。
rf = DecisionTreeClassifier()#决策树
model = AdaBoostClassifier(
                            base_estimator=rf,
                            n_estimators=100,
                            learning_rate=0.5)
model.fit(X_train, y_train)
y_predect = model.predict(X_test)
print("train accuarcy:",accuracy_score(y_test,y_predect))





#如果数据大可以归一化，决策树不归一化也能
from sklearn.preprocessing import StandardScaler
X= StandardScaler().fit_transform(X)
# print(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.7,test_size=0.3,random_state=42)
# 模型训练
# 使用sklearn工具包，调用ensemble.AdaBoostClassifier接口对模型进行训练。
rf = DecisionTreeClassifier()#决策树
model = AdaBoostClassifier(
                            base_estimator=rf,
                            n_estimators=100,
                            learning_rate=0.5)
model.fit(X_train, y_train)
y_predect = model.predict(X_test)
print("train accuarcy:",accuracy_score(y_test,y_predect))