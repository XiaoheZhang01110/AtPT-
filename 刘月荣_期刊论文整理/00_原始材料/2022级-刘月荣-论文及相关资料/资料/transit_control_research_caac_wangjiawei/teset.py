'''
块：
描述单个层、由多个层组成的组件或整个模型本身

从编程的角度来看,块由类（class)表示，它的任意子类都必须定义一个将其输入转换未输出的前向传播函数，
并且必须存储任何必须的参数。有些块不需要参数。
为了计算梯度，块必须具有反向传播函数。在定义自己的块时，由于自动微分提供了一些后端实现，
我们只需要考虑前向传播的函数和必须的参数
'''
'''
自定义块
    包含一个多层感知机，其具有256个隐藏单元的隐藏层和一个10为输出层
    前向传播函数：
        以x作为输入，计算带有激活函数的隐藏表示，并输出其未规范化的输出值
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
# MLP类继承了表示块的类，实现只需要提供自己的构造函数（python中的__init__函数）和前向传播函数
class MLP(nn.Module):
    # 用模型参数声明层，声明两个全连接的层
    def __init__(self):
        # 调用MLP的父类Module的构造函数来执行必要的初始化
        super().__init__()
        self.hidden = nn.Linear(20,256)  # 隐藏层
        self.out = nn.Linear(256,10)     # 输出层

    # 定义模型的前向传播，即如何根据输入X返回所需要的模型输出
    def forward(self,X):
        # 使用ReLU的函数版本，在nn.functional模块中定义
        return self.out(F.relu(self.hidden(X)))
#
# X = torch.rand(2,20)
# print(X)
# net = MLP()
# print(net(X))
# print(net(X).size())

'''
在前向传播函数中执行代码

'''

class FixedHiddenMLP(nn.Module):
    def __init__(self,*arg):
        super.__init__()
        # 不计算梯度的随机权重参数。因此其在训练期间保持不变
        self.rand_weight = torch.rand((20, 20), requires_grad=False)  # 这个权重不是一个模型参数，永远不会被反向传播更新
        self.linear = nn.Linear(20, 20)

    def forward(self,X):
        X = self.linear(X)
        # 使用创建的常量参数以及relu和mm函数
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复用全连接层。这相当于两个全连接层共享参数
        X = self.linear(X)
        # 控制流
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()

# net = FixedHiddenMLP()
# net(X)

# 指定泊松分布的参数 lambda（平均每个时间段发生的事件数）
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# 设置泊松分布的参数 lambda
lambda_param = 10.0

# 生成可能的事件数量
x = np.arange(0, 50)  # 调整范围以包含50个可能的值
pmf_values = poisson.pmf(x, lambda_param)

# 绘制柱状图
plt.bar(x, pmf_values, color='lightblue', edgecolor='black', alpha=0.7)
plt.xlabel('事件数量')
plt.ylabel('概率')
plt.title('Poisson Distribution PMF')

# 保存图表为 JPG 文件
plt.savefig('poisson_pmf.jpg')

# 显示图表
plt.show()