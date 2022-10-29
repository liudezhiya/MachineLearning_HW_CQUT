# -*- coding: utf-8 -*-

# @File    : q3.py
# @Date    : 2022-10-29 15:38  
# @Author  : 刘德智
# @Describe  :
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import scipy.io as scio
data_path='D:\workspace\MachineLearning_HW_CQUT\HW3 SVM\data3.mat'
mat= scio.loadmat(data_path)
print(mat.keys())
X, y = mat['X'], mat['y']
Xval, yval = mat['Xval'], mat['yval']


# 数据可视化
def plot_data():
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap='rainbow')
    plt.xlabel('x1')
    plt.ylabel('y1')



plot_data()
plt.show()
# 设定C和gammma的候选值
Cvalues = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
gammas = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]

# 存放最高的分数
best_score = 0
# 存放最好的参数
best_params = (0, 0)
for c in Cvalues:
    for gamma in gammas:
        svc = SVC(C=c, kernel='rbf', gamma=gamma)
        svc.fit(X, y.flatten())
        score = svc.score(Xval, yval.flatten())
        if score > best_score:
            best_score = score
            best_params = (c, gamma)
print(best_score)
print(best_params)

svc2 = SVC(C=0.3, kernel='rbf', gamma=100)
svc2.fit(X, y.flatten())
# 绘制决策边界
def plot_boundary(model):
    x_min, x_max = -0.6, 0.4
    y_min, y_max = -0.7, 0.6
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    # np.c_是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等。
    z = model.predict(np.c_[xx.flatten(), yy.flatten()])
    zz = z.reshape(xx.shape)
    plt.contour(xx, yy, zz)
plot_boundary(svc2)
plot_data()
plt.show()