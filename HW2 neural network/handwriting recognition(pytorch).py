# -*- coding: utf-8 -*-

# @File    : handwriting recognition(pytorch).py
# @Date    : 2022-10-10
# @Author  : 刘德智
# @Describe  : MNIST手写体数字识别
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

import torchvision
from matplotlib import pyplot as plt
# from utils import plot_image,plot_curve,one_hot
# from torch_study.lesson5_minist_train.utils import plot_curve, plot_image, plt, one_hot
def plot_curve(data):
    fig = plt.figure()
    plt.plot(range(len(data)),data,color = 'blue')
    plt.legend(['value'],loc = 'upper right')
    plt.xlabel('step')
    plt.ylabel('value')
    plt.show()
def plot_image(img,label,name):
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(img[i][0]*0.3081+0.1307,cmap='gray',interpolation='none')
        plt.title("{}:{}".format(name,label[i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()
def one_hot(label,depth=10):
    out = torch.zeros(label.size(0),depth)
    idx = torch.LongTensor(label).view(-1,1)
    out.scatter_(dim = 1,index = idx,value = 1)
    return out



batch_size = 512

# step1. load dataset
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=True)

#batch_size为一次训练多少，shuffle是否打散

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=False)

#查看数据维度
x,y = next(iter(train_loader))
print(x.shape,y.shape,x.min(),x.max())
plot_image(x,y,'image sample')


class Net(nn.Module):

    def __init__(self):
        super(Net,self).__init__()

        #wx+b
        '''
        in_channels：输入channel数
        out_channels：输出channel数
        kernel_size：卷积核的大小，一般使用3比较多，配合上stride=1和padding=1可以保证卷积前后尺寸不变
        '''
        self.fc1 = nn.Linear(28*28,256)
        self.fc2 = nn.Linear(256,64)
        self.fc3 = nn.Linear(64,10)

    def forward(self,x):
        # x:[b,1,28,28]
        # h1=relu(xw1+b1)
        x=F.relu(self.fc1(x))
        # h2=relu(h1*w2+b2)
        x=F.relu(self.fc2(x))
        # h3=h2*w3+b3
        x=self.fc3(x)

        return x


net = Net()
# [w1,b1,w2,b2,w3,b3]  momentum动量

'''
随机梯度下降优化器
params：模型参数
lr：学习率
momentum：动量
weight_decay：正则化参数
'''

optimizer = optim.SGD(net.parameters(),lr=0.05,momentum=0.9)

train_loss = []

#对数据集迭代3次
for epoch in range(3):
    #从数据集中sample出一个batch_size图片
    for batch_idx ,(x,y) in enumerate(train_loader):

        #x:[b,1,28,28] ,y[512]
        #[b,1,28,28] => [b,feature]
        x=x.view(x.size(0),28*28)
        # => [b,10]
        out = net(x)
        #[b,10]
        y_onehot = one_hot(y)
        #loss = mse(out,y_onehot)
        loss = F.cross_entropy(out,y_onehot)
        #清零梯度
        optimizer.zero_grad()
        #计算梯度
        loss.backward()
        #w'=w-lr*grad，更新梯度
        optimizer.step()

        train_loss.append(loss.item())

        if batch_idx %10 ==0:
            print(epoch,batch_idx,loss.item())

#绘制损失曲线
plot_curve(train_loss)
# we get optimal [w1,b1,w2,b2,w3,b3]

#对测试集进行判断
total_corrrect=0
for x,y in test_loader:
    x=x.view(x.size(0),28*28)
    out=net(x)
    # out:[b,10] => pred: [b]
    pred = out.argmax(dim=1)
    correct = pred.eq(y).sum().float().item()
    total_corrrect+=correct

total_num = len(test_loader.dataset)
acc = total_corrrect / total_num
print('test acc:',acc)

x,y=next(iter(test_loader))

out = net(x.view(x.size(0),28*28))
pred = out.argmax(dim=1)
plot_image(x,pred,'test')