# -*- coding: utf-8 -*-

# @File    : adaboot_2.py
# @Date    : 2022-11-05 15:19  
# @Author  : 刘德智
# @Describe  :
import numpy as np
from scipy.io import loadmat
from sklearn import ensemble
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

def test_AdaBoostClassifier(*data):
    '''
    测试 AdaBoostClassifier 的用法，绘制 AdaBoostClassifier 的预测性能随基础分类器数量的影响
    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return: None
    '''
    clf=ensemble.AdaBoostClassifier(learning_rate=0.1)
    clf.fit(X_train,y_train)
    ## 绘图
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    estimators_num=len(clf.estimators_)
    X=range(1,estimators_num+1)#生成最大迭代次数的数组
    ax.plot(list(X),list(clf.staged_score(X_train,y_train)),label="Traing score")
    ax.plot(list(X),list(clf.staged_score(X_test,y_test)),label="Testing score")
    ax.set_xlabel("estimator num")
    ax.set_ylabel("score")
    ax.legend(loc="best")
    ax.set_title("AdaBoostClassifier")
    plt.show()


def test_AdaBoostClassifier_base_classifier(*data):
    '''
    测试  AdaBoostClassifier 的预测性能随基础分类器数量和基础分类器的类型的影响
    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return:  None
    '''
    from sklearn.naive_bayes import GaussianNB
    X_train,X_test,y_train,y_test=data
    fig=plt.figure()
    ax=fig.add_subplot(2,2,1)
    ########### 默认的个体分类器 #############
    clf=ensemble.AdaBoostClassifier(learning_rate=0.1)
    clf.fit(X_train,y_train)
    ## 绘图
    estimators_num=len(clf.estimators_)
    X=range(1,estimators_num+1)
    ax.plot(list(X),list(clf.staged_score(X_train,y_train)),label="Traing score")
    ax.plot(list(X),list(clf.staged_score(X_test,y_test)),label="Testing score")
    ax.set_xlabel("estimator num")
    ax.set_ylabel("score")
    ax.legend(loc="lower right")
    ax.set_ylim(0,1)
    ax.set_title("AdaBoostClassifier with Decision Tree")
    ####### Gaussian Naive Bayes 个体分类器 ########
    ax=fig.add_subplot(2,2,2)
    clf=ensemble.AdaBoostClassifier(learning_rate=0.1,base_estimator=GaussianNB())
    clf.fit(X_train,y_train)
    ## 绘图
    estimators_num=len(clf.estimators_)
    X=range(1,estimators_num+1)
    ax.plot(list(X),list(clf.staged_score(X_train,y_train)),label="Traing score")
    ax.plot(list(X),list(clf.staged_score(X_test,y_test)),label="Testing score")
    ax.set_xlabel("estimator num")
    ax.set_ylabel("score")
    ax.legend(loc="lower right")
    ax.set_ylim(0,1)
    ax.set_title("AdaBoostClassifier with Gaussian Naive Bayes")
    ####### SVC 个体分类器 ########
    ax = fig.add_subplot(2, 2, 3)
    clf = ensemble.AdaBoostClassifier(learning_rate=0.1, base_estimator=SVC(probability=True, kernel='linear'))
    clf.fit(X_train, y_train)
    ## 绘图
    estimators_num = len(clf.estimators_)
    X = range(1, estimators_num + 1)
    ax.plot(list(X), list(clf.staged_score(X_train, y_train)), label="Traing score")
    ax.plot(list(X), list(clf.staged_score(X_test, y_test)), label="Testing score")
    ax.set_xlabel("estimator num")
    ax.set_ylabel("score")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1)
    ax.set_title("AdaBoostClassifier with SVC")
    ####### LogisticRegression() 个体分类器 ########
    ax = fig.add_subplot(2, 2, 4)
    clf = ensemble.AdaBoostClassifier(learning_rate=0.1, base_estimator=LogisticRegression())
    clf.fit(X_train, y_train)
    ## 绘图
    estimators_num = len(clf.estimators_)
    X = range(1, estimators_num + 1)
    ax.plot(list(X), list(clf.staged_score(X_train, y_train)), label="Traing score")
    ax.plot(list(X), list(clf.staged_score(X_test, y_test)), label="Testing score")
    ax.set_xlabel("estimator num")
    ax.set_ylabel("score")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1)
    ax.set_title("AdaBoostClassifier with LogisticRegression()")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    test_AdaBoostClassifier(X_train,X_test,y_train,y_test)
    test_AdaBoostClassifier_base_classifier(X_train,X_test,y_train,y_test)