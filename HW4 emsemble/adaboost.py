# -*- coding: utf-8 -*-

# @File    : adaboost.py
# @Date    : 2022-10-31 15:50  
# @Author  : 刘德智
# @Describe  :
import numpy as np
from scipy.io import loadmat
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
# #输入数据集
data = loadmat('D:\workspace\MachineLearning_HW_CQUT\HW4 emsemble\data1.mat')
# print(data.items())
X = data['X']
y = data['y'].flatten()
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.7,test_size=0.3,random_state=42)
from sklearn.preprocessing import StandardScaler

# 模型训练
# 使用sklearn工具包，调用ensemble.AdaBoostClassifier接口对模型进行训练。
rf = DecisionTreeClassifier()#决策树
svc = SVC(probability=True, kernel='linear')
lg = LogisticRegression()
# clf = AdaBoostClassifier(base_estimator=svc,n_estimators=100)
clf = AdaBoostClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# 可视化决策边界，并输出验证集准确率
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

plt.scatter(X_train[:,0],X_train[:,1],c=y_train.flatten())
plot_predict(clf,[X_train[:,0].min(),X_train[:,0].max(),X_train[:,1].min(),X_train[:,0].max()])

acc = clf.score(X_test, y_test)
print('验证集准确率:',acc)

# 基于实验，分析不同的基分类器和基分类器数量对于模型性能的影响


#随机森林






