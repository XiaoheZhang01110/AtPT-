import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903 简单的GAT
    在图注意力层中，权重矩阵 self.W 和注意力系数  self.a  是必需的，它们是图注意力层的核心组成部分。

     self.W 是一个可训练的参数，表示注意力权重。它的维度是 (in_features, out_features)。
     通过学习权重矩阵  self.W ，图注意力层能够根据输入特征对不同节点进行加权聚合。

     self.a  也是一个可训练的参数，表示注意力系数。它的维度是  (2*out_features, 1) 。
     注意力系数用于计算节点间的注意力权重，通过对节点特征进行线性变换和非线性激活，从而确定节点与其邻居节点之间的重要性。
    """
    # 构造函数，初始化图注意力层参数
    def __init__(self, in_features, out_features, dropout, alpha, concat=True,device=None):  # dropout的概率，用于控制特征的随机失活
        super(GraphAttentionLayer, self).__init__()
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.dropout = dropout
        self.in_features = in_features    # 6
        self.out_features = out_features  # 400
        self.alpha = alpha    # LeakyReLU激活函数的负斜率
        self.concat = concat  # 布尔值，表示是否将多头注意力中的结果进行拼接，默认为True

        # nn.Parameter 是 PyTorch 中用于表示可学习参数的类
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))   # self.W权重矩阵——可训练的参数
        nn.init.xavier_uniform_(self.W.data, gain=1.)   # 对参数矩阵self.w进行xavier均匀初始化，使权重在前向传播过程中保持相对一致的方差
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))   # self.a注意力系数
        nn.init.xavier_uniform_(self.a.data, gain=1.)   # 将 self.a 的值初始化为服从均匀分布的随机数，该分布的方差与输入输出维度有关
        self.leakyrelu = nn.LeakyReLU(self.alpha)   # self.alpha用于指定Leaky ReLU在输入为负时的斜率
        self.to(device)

    def forward(self, h, adj):   # h是节点特征矩阵，adj是图的邻接矩阵，表示节点之间的连接关系
        Wh = torch.mm(h, self.W) # self.W可训练的共享权重矩阵   作用：将节点特征h进行了增维  # 将输入特征h与参数矩阵self.w进行矩阵相乘，得到一个新的特征表示    执行时必须满足矩阵乘法的规则
        a_input = self._prepare_attentional_mechanism_input(Wh)   # 对Wh进行扩维操作  torch.Size([1, 1, 800])  为了准备输入到注意力机制的张量
        e = F.tanh(torch.matmul(a_input, self.a).squeeze(2))    # 计算注意力系数，并使用激活函数进行非线性处理  应该就算是σ(f()) squeeze(2)去除第二维度    self.a的size(800,1)
        zero_vec = -9e15*torch.ones_like(e)   # 创建一个与e维度相同的张量，并填充一个非常小的负值  被用作无效边的权重
        attention = torch.where(adj > 0, e, zero_vec)     # 根据邻接矩阵adj的值，确定哪些边是有效的，通过使用torch.where函数，将e中对应有效边的权重保留，无效边的权重用zero_vec中的值代替，将注意力系数限制在有效边上
        attention = F.softmax(attention, dim=1)      # 对注意力系数进行归一化，得到每个节点的注意力权重
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)   # 利用注意力权重 attention 对特征表示 Wh 进行加权聚合，得到最聚合后的特征矩阵。使用 torch.matmul 函数进行矩阵乘法操作。这一步实现了注意力机制中的加权聚合
        return F.tanh(h_prime)   # 对加权后的特征表示h_prime应用指数线性激活函数，并作为输出返回


    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # number of nodes  获取特征wh维度的大小，并将其第一个维度的值赋给变量N

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks):
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        #
        # These are the rows of the second matrix (Wh_repeated_alternating):
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN
        # '----------------------------------------------------' -> N times
        #
        # ___下面两者都是在第一维度上重复，第一个是在维度上插值重复，第二个是在维度上整体重读
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)   # 将wh沿着维度0重复复制N次   dim=0表示在第一维度上
        Wh_repeated_alternating = Wh.repeat(N, 1)  # 在张量的第一维度上重复N次，第一维度为最外层，即行
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)   # 将两个张量沿着指定的维度进行拼接
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)   # 将张量进行维度变换

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'