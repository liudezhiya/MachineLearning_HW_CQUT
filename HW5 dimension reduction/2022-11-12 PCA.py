# -*- coding: utf-8 -*-

# @File    : 2022-11-12 PCA.py
# @Date    : 2022-11-12 21:53  
# @Author  : 刘德智
# @Describe  :
import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt
from numpy import *

def pca_np(x):
    '''
    对x执行主成分分析，该矩阵在行中包含观察值。
    :param x:
    :return: v, s 返回投影矩阵（x^Tx的特征向量，先以最大特征向量排序）和特征值（从最大到最小排序）
    '''
    x = (x - x.mean(axis=0)) #中心化
    num_observations, num_dimensions = x.shape
    if num_dimensions > 2000:
        eigenvalues, eigenvectors = linalg.eigh(dot(x, x.T))#求方阵的特征值与右特征向量
        v = (dot(x.T, eigenvectors).T)[::-1]
        s = sqrt(eigenvalues)[::-1]
    else:
        u, s, v = linalg.svd(x, full_matrices=False)#奇异值分解SVD
    return v, s



#### 1.1 输入数据集 （30）
# data1.mat中，每一行为一个样本，特征维度为2。'X'
data1 = loadmat('D:\workspace\MachineLearning_HW_CQUT\HW5 dimension reduction\data1.mat')
# data1.mat中，每一行为一个样本，特征维度为2。'X'
data2 = loadmat('D:\workspace\MachineLearning_HW_CQUT\HW5 dimension reduction\data2.mat')
# data3.mat为人脸特征数据集，每一行为一个样本，特征维度为1024。'X'
data3 = loadmat('D:\workspace\MachineLearning_HW_CQUT\HW5 dimension reduction\data3.mat')
X1 = data1['X']
X2 = data2['X']
X3 = data3['X']
X1 = np.array(X1)


#可视化data1和data2进行PCA之后的投影直线
print("="*15,'data1','='*15)
vectors,values = pca_np(X1)
print('data1降维特征值[O:5]\n',np.matmul(X1 - X1.mean(axis=0), vectors)[0:5])
W = vectors[:, :1]  # 选取要降维的特征向量数据
X_d = np.matmul(np.matmul(X1, W), W.T)#单维线
plt.figure(1)
plt.scatter(X1[:, 0], X1[:, 1], c='b')
plt.plot(X_d[:,0],X_d[:,1],c='r')
vectors,values = pca_np(X2)
W = vectors[:, :1]  # 选取要降维的特征向量数据

from sklearn.decomposition import PCA
pca = PCA()
T = pca.fit_transform(X1)
print('sklearn PCA 降维[O:5]\n',T[0:5])


T = np.matmul(X2, W)
X_d = np.matmul(np.matmul(X2, W), W.T)#单维线
plt.figure(2)
plt.scatter(X2[:, 0], X2[:, 1], c='b')
plt.plot(X_d[:,0],X_d[:,1],c='r')




#基于实验，分析data3中根据降维程度的不同，信息损失的差异，并选取最优的降维比率。
from sklearn.decomposition import PCA
pca = PCA()
T = pca.fit_transform(X3)
ratio = pca.explained_variance_ratio_# 获得每个主成分解释的比例
#绘制信息保存度于维数的关系图
plt.figure(3)
plt.plot([i for i in range(X3.shape[1])],
         [np.sum(ratio[:i + 1]) for i in range(X3.shape[1])])
plt.xticks(np.arange(X3.shape[1], step=30))
plt.yticks(np.arange(0, 1.01, 0.05))
plt.grid()#显示网格线
# plt.show()




#使用sklearn工具包对data1进行降维，并进行可视化。
from sklearn.decomposition import PCA
pca = PCA(n_components=1)
T2 = pca.fit_transform(X1)
fig=plt.figure(4)
ax=fig.add_subplot(1,2,1)
plt.plot(T2 ,c='g')
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.title('sklearn工具包实现')

vectors,values = pca_np(X1)#特征向量，特征值
W = vectors[:, :1]  # 选取要降维的特征向量数据
T = np.matmul(X1, W)
ax=fig.add_subplot(1,2,2)
plt.plot(T ,c='r')
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus']=False
plt.title('numpy实现')
plt.show()

