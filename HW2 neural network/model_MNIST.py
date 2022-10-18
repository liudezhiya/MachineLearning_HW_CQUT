# -*- coding: utf-8 -*-

# @File    : model_MNIST.py
# @Date    : 2022-10-16-20-18 
# @Author  : 刘德智
# @Describe  :HW2 MNIST手写体数字识别
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

def plot_image(img,label,name):
    '''
    绘制训练集
    :param img:
    :param label:
    :param name:
    :return:
    '''
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(img[i][0]*0.3081+0.1307,cmap='gray',interpolation='none')
        plt.title("{}:{}".format(name,label[i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()


def loadData(batch_size):
    '''
    加载手写体数据集
    :param batch_size:为一次训练多少
    :return:train_loader,test_loader
    '''
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('mnist_data/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size, shuffle=False)
    return train_loader,test_loader



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)#两次
        self.fc = nn.Linear(320, 50)


    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x

# optimizer = optim.SGD(net.parameters(),lr=0.05,momentum=0.9)

def train(epoch):
    '''
    训练
    :param epoch: epoch次数
    :return:
    '''
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        # 清零梯度
        optimizer.zero_grad()

        outputs = model(inputs)#forward+backward+update
        loss = criterion(outputs, target)
        # 计算梯度
        loss.backward()
        # w'=w-lr*grad，更新梯度
        optimizer.step()
        train_loss.append(loss.item())
        running_loss += loss.item()
        if batch_idx % 200 == 199:
            print('[%d, %5d] loss: %.6f' % (epoch + 1, batch_idx + 1, running_loss / 2000))
            running_loss = 0.0

def test():
    '''
    测试
    :return:
    '''
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            acc=correct / total
        print('Accuracy on test set:{:.4%}'.format(acc))


def plot_curve(data):
    '''
    绘制损失曲线
    :param data:
    :return:
    '''
    fig = plt.figure()
    plt.plot(range(len(data)),data,color = 'blue')
    plt.legend(['value'],loc = 'upper right')
    plt.xlabel('step')
    plt.ylabel('value')
    plt.show()
if __name__ == '__main__':
    train_loader, test_loader=loadData(128)
    # 查看数据，example_data为图片数据，
    # example_targets为图片标签,图片的shape为32, 1, 28, 28，单通道，28*28的图片
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    # print(example_targets)
    # print(example_data.shape)
    # 查看数据维度
    x, y = next(iter(train_loader))
    print(x.shape, y.shape, x.min(), x.max())
    plot_image(x, y, 'sample')

    model = Net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # 交叉熵损失函数，返回值为一个函数
    criterion = torch.nn.CrossEntropyLoss()
    '''
    随机梯度下降优化器
        params：模型参数
        lr：学习率
        momentum：动量
        weight_decay：正则化参数 
    '''
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

    train_loss = []#每次loss
    for epoch in range(10):#10次
        train(epoch)
        test()
        # 绘制损失曲线
        if epoch < 3:
            plot_curve(train_loss)

        # 保存模型
    torch.save(model.state_dict(), 'model.pt')

