import torch
import torch.nn as nn
import numpy as np
import random
import os
from model import layers
import scipy.sparse as sp
import copy


def prepare_eg(fp):
    u_features = []
    d_features = []
    u_adjs = []
    d_adjs = []
    for i in range(len(fp)):  # 分为上下游车辆，上游车辆，晚于当前车辆发车
        fp_ = fp[i][(fp[i][:,-3] <= 0)]  # 将满足(fp[i][:, -3] <= 0)条件的张量选择出来赋值给fp_,[:,-3]是选择张量中倒数第三列的所有行    这儿的意思是寻找倒数第三列（两车之间的距离=当前车辆-相邻车辆） 等于0表示自身节点特征，小于0时，表示相邻车辆比当前车辆后发车，属于上游事件
        edges = np.zeros([fp_.size(0), fp_.size(0)], dtype=np.int32)
        edges[0, :] = 1  # 指的是第一行全部的列(第一行全部为1)
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
        adj = sp.coo_matrix((np.ones(np.sum(edges)), (np.where(edges == 1)[0], np.where(edges == 1)[1])),
                            # (np.where(edges == 1)[0], np.where(edges == 1)[1]))用来决定生成的矩阵中的非零元素的位置的
                            shape=(edges.shape[0], edges.shape[0]))  # np.ones()创建一个由1组成的数组,np.where()返回满足条件的之的索引
        # Do not consider ego event in marginal contribution在边际贡献中不考虑自身事件的影响
        # 矩阵的乘法不会使用
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)  # 转换为对称矩阵
        adj = np.array(adj.todense())  # 将稀疏矩阵转换为密集矩阵，并将其转换为数组
        np.fill_diagonal(adj, 0.)  # 将数组adj的对角线元素设置为0,0的位置是几就将对角线的元素设为几
        adj = torch.FloatTensor(adj)  # no direction

        u_adjs.append(adj)  # 不包含车辆id和时间步间隔两个元素
        u_features.append(fp_[:, :3 + 2])  # 切片：所有行+从0-5列的元素（6列，包含状态3，动作1，车站数1，车辆数1）
        # 下游，车辆id大于当前车辆id
        fp_ = fp[i][(fp[i][:, -3] >= 0)]  # 下游车辆，早于当前车辆发车
        edges = np.zeros([fp_.size(0), fp_.size(0)], dtype=np.int32)
        edges[0, :] = 1
        adj = sp.coo_matrix((np.ones(np.sum(edges)), (np.where(edges == 1)[0], np.where(edges == 1)[1])),
                            shape=(edges.shape[0], edges.shape[0]))
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = np.array(adj.todense())
        np.fill_diagonal(adj, 0.)
        adj = torch.FloatTensor(adj)  # no direction
        d_adjs.append(adj)
        d_features.append(fp_[:, :3 + 2])  # 步包含车辆id和时间步间隔

    return u_adjs, d_adjs, u_features, d_features


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size=400, output_size=1,seed=1):
        super(Actor, self).__init__()



        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, output_size)
        self.linear5 = nn.Linear(hidden_size, output_size)
        self.elu = nn.ELU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()


    def forward(self, s):

        x = self.elu(self.linear1(s))
        x = self.elu(self.linear2(x))
        x = self.elu(self.linear3(x))
        mean = abs(self.tanh(self.linear4(x)))
        std = self.softplus(self.linear5(x))
        return mean,std


class Critic(nn.Module):
    def __init__(self, state_dim, nheads=1, n_stops=22):
        super(Critic, self).__init__()

        self.hidden = 400
        self.state_dim = state_dim

        # for ego critic
        self.fc0 = nn.Linear(state_dim, self.hidden)      # 这儿将输入的维度进行了更改
        self.fc1 = nn.Linear(self.hidden, self.hidden)
        self.fc2 = nn.Linear(self.hidden, 1)
        self.fc3 = nn.Linear(self.hidden, 1)   # 这个用来计算事件评价网络的值的
        # 这儿也进行了更改
        self.u_attentions = [
            layers.GraphAttentionLayer(state_dim + 2, self.hidden, dropout=False, alpha=0.2, concat=True) for _ in
            range(nheads)]   # [GraphAttentionLayer (6 -> 400)]  这是一个多头注意力模块，用于生成多个加权后的特征表示
        for i, attention in enumerate(self.u_attentions):
            self.add_module('attention_{}'.format(i), attention)    # 将 self.u_attentions 中的每个注意力模块依次添加到当前类中作为子模块    self.attention_0:GraphAttentionLayer (6 -> 400)
        self.u_out_att = layers.GraphAttentionLayer(self.hidden * nheads, self.hidden, dropout=False, alpha=0.2,concat=False)  # 定义了输出层的注意力层  self.hidden * nheads

        self.d_attentions = [
            layers.GraphAttentionLayer(state_dim + 2, self.hidden, dropout=False, alpha=0.2, concat=True) for _ in
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
        # u_adjs = [_.to(**self.tpdv) for _ in u_adjs]
        # d_adjs = [_.to(**self.tpdv) for _ in d_adjs]
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
        x, fp = xs    # 分别为状态、动作、节点特征
        # x = x.to(**self.tpdv)  # 对输入的张量 cent_obs、rnn_states 和 masks 进行类型检查，并将它们转移到指定的设备上。
        # fp = [_.to(**self.tpdv) for _ in fp]
        # ego = torch.cat([x, a], 1)   # 将两个张量沿着维度1进行拼接
        Q = self.ego_critic(x)     # 自我评价网络，输入状态值和动作值，得到Q值；   输入智能体的状态+动作（4）的拼接，得到状态价值
        A, reg = self.event_critic(fp)  # 事件评价网络输入了节点特征，包含自身以及相邻智能体的节点特征，特征中包含状态动作对
        G = Q + A
        return G.view(-1, 1),reg


class Agent():
    def __init__(self, state_dim, name, seed=123, n_stops=22, buslist=None):
        random.seed(seed)


        self.seed = seed
        self.name = name
        self.gamma = 0.9
        self.gae_lambda = 0.95
        self.clip_param = 0.2    # 减小为0.1   # 一般将值设为小于等于0.2
        self.entrop_coef = 0.01
        self.state_dim = state_dim
        self.learn_step_counter = 0

        self.critic = Critic(state_dim, n_stops=n_stops)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=0.001)    # 可以使用 self.critic_optim 对象来对模型参数进行优化，例如调用 self.critic_optim.step() 来更新参数
        self.actor = Actor(self.state_dim)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=0.0001)


    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)    # unsqueeze(0)表示在张量为0维的维度上增加一个维度
        # a = self.actor(state).squeeze(0).detach().numpy()          # squeeze(0)表示张量在0的维度上减去一个维度
        mu, std = self.actor(state)
        action_dist = torch.distributions.Normal(mu, std)
        action = action_dist.sample().squeeze(0)
        action_log_prob = (action_dist.log_prob(action)).squeeze(0).detach().cpu().numpy()  # 求出动作对应的对数概率
        return action.cpu().numpy(),action_log_prob

    def learn(self, sample):
        obs_batch, feature_p_batch, actions_batch, old_action_log_probs_batch,next_feature_p_batch, value_pred_batch, returns_batch, adv_targ_batch = sample
        # obs_batch = obs_batch.to(**self.tpdv)
        # feature_p_batch = [_.to(**self.tpdv) for _ in feature_p_batch]
        # actions_batch = actions_batch.to(**self.tpdv)
        # old_action_log_probs_batch = old_action_log_probs_batch.to(**self.tpdv)
        # next_feature_p_batch = [_.to(**self.tpdv) for _ in next_feature_p_batch]
        # value_pred_batch = value_pred_batch.to(**self.tpdv)
        # returns_batch = returns_batch.to(**self.tpdv)
        # adv_targ_batch = adv_targ_batch.to(**self.tpdv)
        # batch_mask = []
        # batch_mask_n = []
        # for x,y in zip(feature_p_batch,next_feature_p_batch):
        #     batch_mask.append(len(x) - 1)
        #     batch_mask_n.append(len(y)-1)

        def actor_learn():   # 这里没有计算交叉熵
            action_log_probs = []
            dist_entropy = []
            mu_, std_ = self.actor(obs_batch)
            action_logit = torch.distributions.Normal(mu_, std_)
            action_log_probs.append(action_logit.log_prob(actions_batch))
            # 计算动作概率分布的熵，熵的值越高，表示该分布的不确定性越大 action_logit.entropy()
            dist_entropy.append(action_logit.entropy().mean())
            action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)
            dist_entropy = dist_entropy[0]

            values,reg = self.critic([obs_batch, feature_p_batch])
            imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
            surr1 = imp_weights * adv_targ_batch
            surr2 = torch.clamp(imp_weights,1.0-self.clip_param,1.0 + self.clip_param) * adv_targ_batch
            policy_action_loss = -torch.sum(torch.min(surr1, surr2),dim=-1,keepdim=True).mean()
            policy_loss = policy_action_loss
            self.actor_optim.zero_grad()
            (policy_loss - dist_entropy * self.entrop_coef).backward()
            self.actor_optim.step()

            return policy_loss,imp_weights,values,dist_entropy,reg

        def critic_learn():
            value_pred_clipped = value_pred_batch + (values - value_pred_batch).clamp(-self.clip_param,self.clip_param)
            error_clipped = returns_batch - value_pred_clipped
            error_original = returns_batch - values
            value_loss_clipped = error_clipped ** 2 / 2
            value_loss_original = error_original ** 2 / 2
            value_loss = (torch.max(value_loss_original,value_loss_clipped)).mean()
            self.critic_optim.zero_grad()
            value_loss.backward()
            self.critic_optim.step()
            return value_loss

        policy_loss, imp_weights,values,dist_entropy,reg = actor_learn()
        value_loss = critic_learn()

        return value_loss,policy_loss,imp_weights,dist_entropy




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
