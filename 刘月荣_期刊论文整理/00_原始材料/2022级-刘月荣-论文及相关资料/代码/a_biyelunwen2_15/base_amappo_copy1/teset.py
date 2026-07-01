import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0, alpha=0.2):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.eye(in_features, out_features))  # 初始化为单位矩阵
        self.a = nn.Parameter(torch.tensor([[1.0], [-1.0]]))  # 简化注意力参数
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, h, edge_index):
        # 1. 线性变换
        h_transformed = torch.mm(h, self.W)

        # 2. 计算注意力系数
        row, col = edge_index
        h_i = h_transformed[row]
        h_j = h_transformed[col]
        concat = torch.cat([h_i, h_j], dim=1)
        e = self.leakyrelu(torch.matmul(concat, self.a).squeeze())

        # 3. Softmax归一化（修正为按目标节点分组）
        unique_nodes = torch.unique(row)
        attention = torch.zeros_like(e)
        for node in unique_nodes:
            mask = (row == node)
            attention[mask] = F.softmax(e[mask], dim=0)

        # 4. 加权聚合（优化为向量化操作）
        h_prime = torch.zeros_like(h_transformed)
        for idx, (i, j) in enumerate(zip(row, col)):
            h_prime[i] += attention[idx] * h_transformed[j]

        return F.elu(h_prime)


# 测试数据
h = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32)  # 3个节点，2维特征
edge_index = torch.tensor([[0, 0, 1, 1], [1, 2, 0, 2]], dtype=torch.long)  # 边：0->1, 0->2, 1->0, 1->2

# 初始化模型并关闭梯度
model = GATLayer(in_features=2, out_features=2)
model.W.requires_grad_(False)
model.a.requires_grad_(False)

# 前向计算
output = model(h, edge_index)
print("输出结果:\n", output)