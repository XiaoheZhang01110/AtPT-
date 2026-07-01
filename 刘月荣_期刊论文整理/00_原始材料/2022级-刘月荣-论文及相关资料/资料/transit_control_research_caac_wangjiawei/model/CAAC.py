import torch
import torch.nn as nn
import numpy as np
import random
import os
from model import layers
import scipy.sparse as sp
import copy
'''
COO 格式是一种用于表示稀疏矩阵的数据结构。它通过三个数组来存储矩阵中的非零元素的坐标和对应的值。这三个数组分别是行索引数组、列索引数组和数值数组
参数说明：
data：一个一维数组，表示矩阵中非零元素的值。
row：一个一维数组，表示非零元素的行索引。
col：一个一维数组，表示非零元素的列索引。
shape：一个元组，表示矩阵的形状。
'''

def prepare_eg(fp):
    u_features = []
    d_features = []
    u_adjs = []
    d_adjs = []
    for i in range(len(fp)):                # 分为上下游车辆，上游车辆，晚于当前车辆发车
        fp_ = fp[i][(fp[i][:, -3] <= 0)]    # 将满足(fp[i][:, -3] <= 0)条件的张量选择出来赋值给fp_,[:,-3]是选择张量中倒数第三列的所有行    这儿的意思是寻找倒数第三列（两车之间的距离=当前车辆-相邻车辆） 等于0表示自身节点特征，小于0时，表示相邻车辆比当前车辆后发车，属于上游事件
        edges = np.zeros([fp_.size(0), fp_.size(0)], dtype=np.int32)
        edges[0, :] = 1    # 指的是第一行全部的列(第一行全部为1)
        '''
        sp.coo_matrix((data, (row, col)), shape=None, dtype=None)用来创建稀疏矩阵的
            data：一个一维数组，表示非零元素的值。
            row：一个一维数组，表示每个非零元素所在的行索引。
            col：一个一维数组，表示每个非零元素所在的列索引。
            shape：可选参数，表示矩阵的形状（行数和列数）。如果未提供该参数，则根据row和col中的最大索引确定形状。
            dtype：可选参数，表示矩阵的数据类型。如果未提供该参数，则根据输入的data参数推断数据类型
        
        np.where()是NumPy库中的一个函数，用于根据给定的条件返回满足条件的元素的索引或值。
        np.where(condition[, x, y])
            condition：一个布尔数组或条件表达式，用于指定筛选条件。
            x：可选参数，表示满足条件的元素的替代值。
            y：可选参数，表示不满足条件的元素的替代值。
            
            np.where()函数可以有两种用法：
            传入单个参数condition，返回满足条件的元素的索引。
            传入三个参数condition、x和y，返回根据条件选择的元素的值。
        '''
        adj = sp.coo_matrix((np.ones(np.sum(edges)), (np.where(edges == 1)[0], np.where(edges == 1)[1])),     # (np.where(edges == 1)[0], np.where(edges == 1)[1]))用来决定生成的矩阵中的非零元素的位置的
                            shape=(edges.shape[0], edges.shape[0]))     # np.ones()创建一个由1组成的数组,np.where()返回满足条件的之的索引
        # Do not consider ego event in marginal contribution在边际贡献中不考虑自身事件的影响
        # 矩阵的乘法不会使用
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)   # 转换为对称矩阵
        adj = np.array(adj.todense())   # 将稀疏矩阵转换为密集矩阵，并将其转换为数组
        np.fill_diagonal(adj, 0.)    # 将数组adj的对角线元素设置为0,0的位置是几就将对角线的元素设为几
        adj = torch.FloatTensor(adj)  # no direction

        u_adjs.append(adj)                      # 不包含车辆id和时间步间隔两个元素
        u_features.append(fp_[:, :3 + 1 + 2])   # 切片：所有行+从0-5列的元素（6列，包含状态3，动作1，车站数1，车辆数1）
        # 下游，车辆id大于当前车辆id
        fp_ = fp[i][(fp[i][:, -3] >= 0)]   # 下游车辆，早于当前车辆发车
        edges = np.zeros([fp_.size(0), fp_.size(0)], dtype=np.int32)
        edges[0, :] = 1
        adj = sp.coo_matrix((np.ones(np.sum(edges)), (np.where(edges == 1)[0], np.where(edges == 1)[1])),
                            shape=(edges.shape[0], edges.shape[0]))
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = np.array(adj.todense())
        np.fill_diagonal(adj, 0.)
        adj = torch.FloatTensor(adj)  # no direction
        d_adjs.append(adj)
        d_features.append(fp_[:, :3 + 1 + 2])   # 步包含车辆id和时间步间隔

    return u_adjs, d_adjs, u_features, d_features


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size=400, output_size=1):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, output_size)   # 四个线性层
        self.elu = nn.ELU()     # 激活函数，将线性层的输出进行非线性的变换

    def forward(self, s):   # 神经网络的前面传播过程
        x = self.elu(self.linear1(s))   # 输入的是被传入了线性层，输出进行非线性变换，然后传入了线性层2....
        x = self.elu(self.linear2(x))
        x = self.elu(self.linear3(x))
        x = self.elu(self.linear4(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, nheads=1, n_stops=22):

        super(Critic, self).__init__()
        self.hidden = 400
        self.state_dim = state_dim

        # for ego critic
        self.fc0 = nn.Linear(state_dim + 1, self.hidden)
        self.fc1 = nn.Linear(self.hidden, self.hidden)
        self.fc2 = nn.Linear(self.hidden, 1)
        self.fc3 = nn.Linear(self.hidden, 1)   # 这个用来计算事件评价网络的值的

        self.u_attentions = [
            layers.GraphAttentionLayer(state_dim + 1 + 2, self.hidden, dropout=False, alpha=0.2, concat=True) for _ in
            range(nheads)]   # [GraphAttentionLayer (6 -> 400)]  这是一个多头注意力模块，用于生成多个加权后的特征表示
        for i, attention in enumerate(self.u_attentions):
            self.add_module('attention_{}'.format(i), attention)    # 将 self.u_attentions 中的每个注意力模块依次添加到当前类中作为子模块    self.attention_0:GraphAttentionLayer (6 -> 400)
        self.u_out_att = layers.GraphAttentionLayer(self.hidden * nheads, self.hidden, dropout=False, alpha=0.2,concat=False)  # 定义了输出层的注意力层  self.hidden * nheads

        self.d_attentions = [
            layers.GraphAttentionLayer(state_dim + 1 + 2, self.hidden, dropout=False, alpha=0.2, concat=True) for _ in
            range(nheads)]
        for i, attention in enumerate(self.d_attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.d_out_att = layers.GraphAttentionLayer(self.hidden * nheads, self.hidden, dropout=False, alpha=0.2,
                                                    concat=False)

        self.relu = nn.ReLU()     # ReLU 函数在输入小于零时返回零，否则返回输入值本身
        self.elu = nn.ELU()       # ELU 函数在输入小于零时返回指数级的负值，否则返回输入值本身
        self.n_stops = n_stops    # 46

    def d_egat(self, x, adj):
        x = torch.cat([att(x, adj) for att in self.d_attentions], dim=1)
        x = self.d_out_att(x, adj)   # fp中有几个元素，求出的值就有几个
        x = torch.sum(x, 0)   # 最终得到新的节点特征
        return x

    def u_egat(self, x, adj):
        x = torch.cat([att(x, adj) for att in self.u_attentions], dim=1)   # 每个 att(x, adj) 的结果都是一个加权后的特征表示，通过 torch.cat 函数将它们在维度 1 上进行拼接，得到最终的特征表示 x。这一步实现了多头注意力机制。
        x = self.u_out_att(x, adj)
        x = torch.sum(x, 0)   # 将张量x沿着维度0求和
        return x

    def event_critic(self, fp):    # fp代表的是节点特征：自身节点特征+在它两次到站的时间间隔内其相邻的前后向公交到站的节点特征
        u_adjs, d_adjs, u_features, d_features = prepare_eg(fp)   # 将fp分为上游和下游，也就是将相邻的车辆分为，在当前车辆的后还是前
        a = []
        reg = []
        for i in range(len(u_adjs)):
            u_x = u_features[i]   # 上游特征
            u_adj = u_adjs[i]
            d_x = d_features[i]   # 下游特征
            d_adj = d_adjs[i]
            if u_adj.size(0) >= 2:   # 大于等于证明fp中除了包含自身的特征，也有来自邻居的特征
                u_x = self.u_egat(u_x, u_adj)
            else:                    # 只含有自身的特征
                u_x = self.u_egat(u_x, u_adj)   # 只含有自身的特征
                reg.append(torch.square(u_x))    # torch.square对张量中的每个元素进行平方操作
                u_x = torch.zeros_like(u_x)   # 用于创建一个与给定输入张量具有相同形状的全零张量

            if d_adj.size(0) >= 2:
                d_x = self.d_egat(d_x, d_adj)
            else:
                d_x = self.d_egat(d_x, d_adj)
                reg.append(torch.square(d_x))
                d_x = torch.zeros_like(d_x)
            u_x = u_x.view(-1, self.hidden)   # 上游特征
            d_x = d_x.view(-1, self.hidden)   # 下游特征
            a.append(self.fc3(u_x + d_x))      # 得到的就是事件评价网络计算的Q值

        a = torch.stack(a, 0).view(-1, 1)   # torch.stack(a, 0)沿着第一位都维将a进行堆叠
        if len(reg) > 0:
            reg = torch.stack(reg, 0).view(-1, 1)
        else:
            reg = torch.zeros(1)
        return a, reg

    def ego_critic(self, ego):    # 输入的是智能体的局部状态（3）+动作（1）
        out1 = self.fc0(ego)
        out1 = self.relu(out1)
        out1 = self.fc1(out1)
        out1 = self.relu(out1)
        Q = self.fc2(out1)
        return Q              # shape(16,1)

    def forward(self, xs): # 代表了状态、动作 、以及节点特征
        x, a, fp = xs    # 分别为状态、动作、节点特征
        ego = torch.cat([x, a], 1)   # 将两个张量沿着维度1进行拼接
        Q = self.ego_critic(ego)     # 自我评价网络，输入状态值和动作值，得到Q值；   输入智能体的状态+动作（4）的拼接，得到状态价值
        A, reg = self.event_critic(fp)  # 事件评价网络输入了节点特征，包含自身以及相邻智能体的节点特征，特征中包含状态动作对
        G = Q + A
        return Q, A, G.view(-1, 1), reg


class Agent():
    def __init__(self, state_dim, name, seed=123, n_stops=22, buslist=None):
        random.seed(seed)
        self.seed = seed
        self.name = name
        self.gamma = 0.9
        self.state_dim = state_dim
        self.learn_step_counter = 0

        self.critic = Critic(state_dim, n_stops=n_stops)
        self.critic_target = Critic(state_dim, n_stops=n_stops)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=0.001)    # 可以使用 self.critic_optim 对象来对模型参数进行优化，例如调用 self.critic_optim.step() 来更新参数
        self.critic_target.load_state_dict(self.critic.state_dict())    # 将self.critic模型的参数赋值给self.critic_target(load_state_dict()用于加载参数字典到模型中,state_dict()用于返回模型的参数字典)

        self.actor = Actor(self.state_dim)
        self.actor_target = Actor(self.state_dim)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=0.0001)
        self.actor_target.load_state_dict(self.actor.state_dict())

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)    # unsqueeze(0)表示在张量为0维的维度上增加一个维度
        a = self.actor(state).squeeze(0).detach().numpy()          # squeeze(0)表示张量在0的维度上减去一个维度
        return a

    def learn(self, memories, batch=16):
        if len(memories) < batch:
            return 0, 0

        batch_s, batch_fp, batch_a, batch_r, batch_ns, batch_nfp = [], [], [], [], [], []
        # 在经验回放池中取出16批次数据
        memory = random.sample(memories, batch)     # 从memories中选择batch个唯一元素
        # memory中存储的是当前智能体对应的一个个((batch_s, batch_fp, batch_a, batch_r, batch_ns, batch_nfp ))
        batch_mask = []
        batch_mask_n = []
        batch_fp_critic_t = []
        batch_actor_a = []
        for s, fp, a, r, ns, nfp, in memory:   # 主要的操作就是将状态放在一起，动作放在一起，。。。。
            batch_s.append(s)
            _fp_ = copy.deepcopy(fp)
            _fp_ = torch.tensor(_fp_, dtype=torch.float32)
            _fp_[0, self.state_dim+1] = self.actor(torch.tensor(s, dtype=torch.float32)).detach()   #将_fp_中的第0行第4列的数据替换
            batch_fp_critic_t.append(_fp_)
            batch_actor_a.append(self.actor(torch.tensor(s, dtype=torch.float32)))   # 包含有梯度的
            batch_fp.append(torch.FloatTensor(fp))   # torch.FloatTensor用于生成包含浮点数的张量
            batch_mask.append(len(fp) - 1)
            batch_mask_n.append(len(nfp) - 1)
            batch_a.append(a)
            batch_r.append(r)
            batch_ns.append(ns)
            batch_nfp.append(torch.FloatTensor(nfp))
        b_fp_pad = batch_fp
        b_nfp_pad = batch_nfp
        # 进行形式上的转换
        batch_actor_a = torch.stack(batch_actor_a, 0)       # 在0维度上对张量进行堆叠，生成新的张量
        b_s = torch.tensor(batch_s, dtype=torch.float)      # s
        b_a = torch.tensor(batch_a, dtype=torch.float).view(-1, 1)   # a 重塑成一个列为1的张量，自动计算
        b_r = torch.tensor(batch_r, dtype=torch.float).view(-1, 1)   # r
        b_ns = torch.tensor(batch_ns, dtype=torch.float)             # ns

        def critic_learn():
            Q, A, G, reg = self.critic([b_s, b_a, b_fp_pad])   # 输入了经验回放池中的状态、动作、智能体的节点特征
            Q_, A_, G_, _ = self.critic_target(
                [b_ns, self.actor_target(b_ns).detach(), b_nfp_pad])  # 目标评价网络输入了敬仰回放池中智能体的下一个状态、下一状态的节点特征，以及有目标网络得到的下一状态生成的动作
            q_target = b_r + self.gamma * (G_.detach()).view(-1, 1)  #  b_r 是来自经验回放缓冲区的奖励，G_ 是下一个状态的折扣累积回报

            loss_fn = nn.MSELoss()    # 创建了一个均方误差损失函数，用于计算Q值的损失
            qloss = loss_fn(G, q_target) + 0.1 * reg.mean()  # G是Critic 网络根据当前状态估计的 Q值，q_target 是计算得到的目标 Q值，reg是正则化项，reg.mean() 表示正则化项的平均值  ？这儿的范数没有开根？
            self.critic_optim.zero_grad()   # 将 Critic 网络的优化器的梯度缓冲区清零，准备接收新的梯度。
            qloss.backward()             # 通过反向传播计算损失函数关于 Critic 网络参数的梯度
            self.critic_optim.step()     # 使用优化器更新 Critic 网络的参数，根据反向传播计算得到的梯度

            return qloss.item()   # 返回损失的标量值

        def actor_learn():
            policy_loss, _,  _, _ = self.critic([b_s, batch_actor_a, batch_fp_critic_t])    # policy_loss=Q,是由于critic根据输入的状态和动作计算出来的值   policy_loss是由ego_critic网络输出的，与event_critic无关
            policy_loss = -torch.mean(policy_loss)
            self.actor_optim.zero_grad()
            policy_loss.backward()
            self.actor_optim.step()
            return policy_loss.item()

        def soft_update(net_target, net, tau=0.02):
            for target_param, param in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)  # 软更新的方式w_=tw+(1-t)w_

        qloss = critic_learn()
        policy_loss = actor_learn()

        soft_update(self.critic_target, self.critic, tau=0.02)
        soft_update(self.actor_target, self.actor, tau=0.02)
        self.learn_step_counter += 1

        return policy_loss, qloss

    def save(self, model):
        abspath = os.path.abspath(os.path.dirname(__file__))

        path = abspath + "\\save\\" + str(self.name) + '_' + str(model) + str(self.seed) + "_actor.pth"
        torch.save(self.actor.state_dict(), path)

        path = abspath + "\\save\\" + str(self.name) + '_' + str(model) + str(self.seed) + "_critic.pth"
        torch.save(self.critic.state_dict(), path)

    def load(self, model):
        try:
            abspath = os.path.abspath(os.path.dirname(__file__))
            print('Load: ' + abspath + "/save/" + str(self.name) + '_' + str(model))
            path = abspath + "/save/" + str(self.name) + '_' + str(model) + str(self.seed) + "_actor.pth"
            state_dict = torch.load(path)
            self.actor.load_state_dict(state_dict)
        except:
            abspath = os.path.abspath(os.path.dirname(__file__))
            print('Load: ' + abspath + "/save/" + str(self.name) + '_' + str(model))
            path = abspath + "\\save\\" + str(self.name) + '_' + str(model) + str(self.seed) + "_actor.pth"
            state_dict = torch.load(path)
            self.actor.load_state_dict(state_dict)
