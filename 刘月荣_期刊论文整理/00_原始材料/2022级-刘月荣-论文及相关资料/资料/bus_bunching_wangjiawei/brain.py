import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import numpy as np
'''
class PolicyNet(nn.Module):
    def __init__(self, input_size, hidden_size=400, output_size=1,seed=1):
        super(PolicyNet, self).__init__()

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


class ValueNet(nn.Module):
    def __init__(self, state_dim):

        super(ValueNet, self).__init__()
        self.hidden = 400
        self.state_dim = state_dim

        # for ego critic
        self.fc0 = nn.Linear(state_dim-1 , self.hidden)   # 自身状态
        self.fc2 = nn.Linear(self.hidden, self.hidden)
        self.fc3 = nn.Linear(self.hidden, 1)   # 这个用来计算事件评价网络的值的
        self.relu = nn.ReLU()     # ReLU 函数在输入小于零时返回零，否则返回输入值本身
        self.elu = nn.ELU()       # ELU 函数在输入小于零时返回指数级的负值，否则返回输入值本身




    def forward(self, x): # 代表了状态、动作 、以及节点特征

        out1 = self.fc0(x)
        out1 = self.relu(out1)
        out1 = self.fc2(out1)   # 将两个张量沿着维度1进行拼接
        out1 = self.relu(out1)     # 自我评价网络，输入状态值和动作值，得到Q值；   输入智能体的状态+动作（4）的拼接，得到状态价值
        Q = self.fc3(out1)  # 事件评价网络输入了节点特征，包含自身以及相邻智能体的节点特征，特征中包含状态动作对

        return Q


class PPO():
    def __init__(self, state_dim_loacl,state_dim,action_dim,actor_lr,critic_lr,epochs,eps,gamma, seed=123):
        random.seed(seed)

        self.seed = seed
        self.gamma = 0.9
        self.eps = 0.2
        self.epochs = epochs
        self.state_dim = state_dim

        self.critic = ValueNet(state_dim)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=0.001)    # 可以使用 self.critic_optim 对象来对模型参数进行优化，例如调用 self.critic_optim.step() 来更新参数
        self.actor = PolicyNet(state_dim_loacl,action_dim)    # state_dim_loacl, action_dim
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=0.0001)

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).unsqueeze(0)    # unsqueeze(0)表示在张量为0维的维度上增加一个维度
        # a = self.actor(state).squeeze(0).detach().numpy()          # squeeze(0)表示张量在0的维度上减去一个维度
        mu,std = self.actor(state)
        action_dist = torch.distributions.Normal(mu, std)
        action = action_dist.sample()
        # action_log_prob = (action_dist.log_prob(action)).squeeze(0).detach().numpy()  # 求出动作对应的对数概率
        return action.item()



    def update(self,states_all,n_states_all,states,actions,buffer_r_c):
        mean_log_probs, mean_ratio, mean_surr1, mean_surr2, mean_actor_loss, mean_critic_loss= [],[],[],[],[],[]

        states_all = torch.tensor(np.array(states_all), dtype=torch.float32).view(-1, 8)
        n_states_all = torch.tensor(np.array(n_states_all), dtype=torch.float32).view(-1, 8)
        states = torch.tensor(np.array(states), dtype=torch.float32).view(-1, 3)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).view(-1, 1)

        discounted_r = 0
        returns = []
        for r in buffer_r_c[::-1]:
            discounted_r = r + self.gamma * discounted_r
            returns.insert(0,discounted_r)
        # 累积折扣回报
        returns = torch.tensor(np.array(returns),dtype=torch.float32).view(-1, 1)
        old_values = self.critic(states_all)
        # 优势函数
        advantage = returns - old_values

        old_n_values = self.critic(n_states_all)
        target_values = torch.tensor(buffer_r_c,dtype=torch.float32).view(-1, 1) + self.gamma * old_n_values

        mu,std = self.actor(states)
        old_action_dist = torch.distributions.Normal(mu.detach(), std.detach())
        old_log_probs = old_action_dist.log_prob(actions)
        for _ in range(self.epochs):

            mu, std = self.actor(states)
            action_dist = torch.distributions.Normal(mu, std)
            log_probs = action_dist.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(F.mse_loss(self.critic(states_all), target_values.detach()))
            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()
            actor_loss.backward(retain_graph=True)
            critic_loss.backward(retain_graph=True)
            self.actor_optim.step()
            self.critic_optim.step()

            mean_log_probs.append(log_probs)
            mean_ratio.append(ratio)
            mean_surr1.append(surr1)
            mean_surr2.append(surr2)
            mean_actor_loss.append(actor_loss.item())
            mean_critic_loss.append(critic_loss.item())

        mean_log_probs = torch.cat(mean_log_probs).mean()
        mean_ratio = torch.cat(mean_ratio).mean()
        mean_surr1 = torch.cat(mean_surr1).mean()
        mean_surr2 = torch.cat(mean_surr2).mean()
        mean_actor_loss = torch.tensor(mean_actor_loss).mean()
        mean_critic_loss = torch.tensor(mean_critic_loss).mean()

        return (
            advantage,
            mean_log_probs,
            old_log_probs,
            mean_ratio,
            mean_surr1,
            mean_surr2,
            mean_actor_loss,
            mean_critic_loss
        )

'''



class PolicyNet(torch.nn.Module):
    def __init__(self,state_dim,action_dim,hidden_dim=400):
        super(PolicyNet,self).__init__()
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, action_dim)
        self.linear5 = nn.Linear(hidden_dim,  action_dim)
        self.elu = nn.ELU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()


    def forward(self,x):
        x = self.elu(self.linear1(x))
        x = self.elu(self.linear2(x))
        x = self.elu(self.linear3(x))
        mean = abs(self.tanh(self.linear4(x)))
        std = self.softplus(self.linear5(x))

        return mean, std

class ValueNet(torch.nn.Module):
    def __init__(self,state_dim):
        super(ValueNet, self).__init__()
        self.hidden = 400
        self.state_dim = state_dim
        self.fc0 = nn.Linear(state_dim - 1, self.hidden)  # 自身状态
        self.fc2 = nn.Linear(self.hidden, self.hidden)
        self.fc3 = nn.Linear(self.hidden, 1)  # 这个用来计算事件评价网络的值的
        self.relu = nn.ReLU()  # ReLU 函数在输入小于零时返回零，否则返回输入值本身
        self.elu = nn.ELU()

    def forward(self, x):  # 代表了状态、动作 、以及节点特征

        out1 = self.fc0(x)
        out1 = self.relu(out1)
        out1 = self.fc2(out1)  # 将两个张量沿着维度1进行拼接
        out1 = self.relu(out1)  # 自我评价网络，输入状态值和动作值，得到Q值；   输入智能体的状态+动作（4）的拼接，得到状态价值
        Q = self.fc3(out1)  # 事件评价网络输入了节点特征，包含自身以及相邻智能体的节点特征，特征中包含状态动作对

        return Q


class PPO():
    def __init__(self,state_dim_loacl,state_dim,action_dim,actor_lr,critic_lr,epochs,eps,gamma):
        self.actor = PolicyNet(state_dim_loacl, action_dim)
        self.critic = ValueNet(state_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=critic_lr)
        self.gamma = gamma
        self.epochs = epochs
        self.eps = eps
    def take_action(self,state):
        state = torch.tensor([state], dtype=torch.float)
        mu,sigma = self.actor(state)
        action_dist = torch.distributions.Normal(mu,sigma)
        action = action_dist.sample().squeeze(0).detach().numpy()    # 将动作限制在0-1之间
        return action.item()

    def update(self,states_all,n_states_all,states,actions,buffer_r_c):
        states_all = torch.tensor(np.array(states_all), dtype=torch.float32).view(-1, 8)
        n_states_all = torch.tensor(np.array(n_states_all), dtype=torch.float32).view(-1, 8)
        states = torch.tensor(np.array(states), dtype=torch.float32).view(-1, 3)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).view(-1, 1)
        # states_n_all = torch.tensor(np.array(states_n_all), dtype=torch.float32).view(-1, 8)
        # buffer_r_c = torch.tensor(np.array(buffer_r_c), dtype=torch.float32).view(-1, 1)

        discounted_r = 0
        returns = []
        for r in buffer_r_c[::-1]:
            discounted_r = r + self.gamma * discounted_r
            returns.insert(0,discounted_r)
        # 累积折扣回报
        returns = torch.tensor(np.array(returns),dtype=torch.float32).view(-1, 1)

        mu, std = self.actor(states)
        action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        # old_log_probs = torch.clamp(action_dists.log_prob(actions),-20,20)
        old_log_probs = action_dists.log_prob(actions)
        advantage = returns - self.critic(states_all).detach()
        target_values = torch.tensor(buffer_r_c, dtype=torch.float32).view(-1, 1) + self.gamma * self.critic(n_states_all)
        for _ in range(self.epochs):

            mu,std = self.actor(states)
            action_dists = torch.distributions.Normal(mu, std)
            log_probs = action_dists.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            # 策略网络的目标函数（要最大化，因此要最小化负值）
            actor_loss = torch.mean(-torch.min(surr1,surr2))

            # 价值网络的目标函数
            # buffer_r_c = torch.tensor(np.array(buffer_r_c), dtype=torch.float32).view(-1, 1)
            # td_target = buffer_r_c + self.gamma * self.critic(states_n_all)
            # critic_loss = torch.mean(F.mse_loss(values, td_target))  # 估计值和目标值的均方误差最小
            critic_loss = torch.mean(F.mse_loss(self.critic(states_all), target_values.detach()))

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            # 对策略网络的梯度进行裁剪
            # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=5.0)
            # # 对价值网络的梯度进行裁剪
            # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=5.0)

            self.actor_optimizer.step()
            self.critic_optimizer.step()
        return advantage,log_probs,old_log_probs,ratio,surr1,surr2,actor_loss, critic_loss




