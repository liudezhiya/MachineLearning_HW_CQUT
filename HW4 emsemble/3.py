import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # 随机森林
from sklearn.ensemble import AdaBoostClassifier  # Adaboost
from sklearn.svm import SVC
from sklearn.datasets import load_iris  # 鸾尾花数据
import time

RF = RandomForestClassifier(n_estimators=100,max_depth=8, n_jobs=4, oob_score=True)
Ad = AdaBoostClassifier(n_estimators=100, learning_rate=0.1, algorithm='SAMME.R')
svm = SVC()
iris = load_iris()
x = iris.data[:, :2]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.7,test_size=0.3,random_state=42)
print("========")
# 随机森林
starttime = time.time()
RF.fit(X_train, y_train)
print('RandomForestClassifier:', RF.score(X_test, y_test))
# print('RandomForestClassifier:', RF.score(X_train, y_train))
endtime = time.time()
print('所耗时间',endtime - starttime)
print("========")
# Adaboost
starttime = time.time()
Ad.fit(X_test, y_test)
# print('AdaboostClassifier:', Ad.score(X_train, y_train))
print('AdaboostClassifier:', Ad.score(X_test, y_test))
endtime = time.time()
print('所耗时间',endtime - starttime)