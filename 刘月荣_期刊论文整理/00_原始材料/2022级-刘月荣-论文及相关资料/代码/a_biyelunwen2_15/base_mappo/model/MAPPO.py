import torch
import torch.nn as nn
import numpy as np
import random
import os
from model import layers
import scipy.sparse as sp
import copy


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size=400, output_size=1,seed=1):
        super(Actor, self).__init__()

        self.linear1 = nn.Linear(input_size+2, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, output_size)
        self.linear5 = nn.Linear(hidden_size, output_size)
        self.elu = nn.ELU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()


    def forward(self, s):
        # s, a_ = xs
        x = self.tanh(self.linear1(s))
        x = self.tanh(self.linear2(x))
        x = self.tanh(self.linear3(x))
        # x = self.tanh(self.linear4(x))
        mean = abs(self.tanh(self.linear4(x)))
        std = self.softplus(self.linear5(x))
        # action_dist = torch.distributions.Normal(mean, std)
        return mean,std

    def get_dist(self, s):
        mean,std = self.forward(s)
        dist = torch.distributions.Normal(mean, std)
        return dist


class Critic(nn.Module):
    def __init__(self, state_dim, nheads=1, n_stops=12,n_buses=6):

        super(Critic, self).__init__()

        self.hidden = 400
        self.state_dim = state_dim

        # for ego critic
        self.fc0 = nn.Linear(state_dim *3, self.hidden)   # 自身状态
        self.fc1 = nn.Linear(self.hidden, self.hidden)
        self.fc2 = nn.Linear(self.hidden, self.hidden)
        self.fc3 = nn.Linear(self.hidden, 1)   # 这个用来计算事件评价网络的值的

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()     # ReLU 函数在输入小于零时返回零，否则返回输入值本身
        self.elu = nn.ELU()       # ELU 函数在输入小于零时返回指数级的负值，否则返回输入值本身
        self.n_stops = n_stops    # 46


    def forward(self, s): # 代表了状态、动作 、以及节点特征

        s = self.tanh(self.fc0(s))
        s = self.tanh(self.fc1(s))
        s = self.tanh(self.fc2(s))
        out1 = self.fc3(s)
        return out1


class Agent():
    def __init__(self, state_dim, name, seed=123, n_stops=22, buslist=None):
        random.seed(seed)


        self.seed = seed
        self.name = name
        self.gamma = 0.9
        self.gae_lambda = 0.95
        self.clip_param = 0.2
        self.entrop_coef = 0.01
        self.state_dim = state_dim
        self.learn_step_counter = 0

        self.critic = Critic(state_dim, n_stops=n_stops)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=0.0001,eps=1e-5) # eps=1e-4   # 可以使用 self.critic_optim 对象来对模型参数进行优化，例如调用 self.critic_optim.step() 来更新参数
        self.actor = Actor(self.state_dim)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=0.0001,eps=1e-5)   # eps=1e-4


    def choose_action(self, state):
        # s, a_ = state
        #
        # s = torch.tensor(s, dtype=torch.float).unsqueeze(0)
        # a_ = torch.tensor(a_, dtype=torch.float).unsqueeze(0)
        # fp = torch.tensor(fp, dtype=torch.float).unsqueeze(0)
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)    # unsqueeze(0)表示在张量为0维的维度上增加一个维度
        # a = self.actor(state).squeeze(0).detach().numpy()          # squeeze(0)表示张量在0的维度上减去一个维度
        with torch.no_grad():
            dist = self.actor.get_dist(state)
            action = dist.sample()
            action_log_prob = dist.log_prob(action)

        return action.numpy().flatten(), action_log_prob.numpy().flatten()



    def learn(self, sample):
        obs_batch, w_obs_batch, actions_batch, old_action_log_probs_batch, value_pred_batch, returns_batch, adv_targ_batch = sample
        def actor_learn():   # 这里没有计算交叉熵
            action_log_probs = []
            dist_entropy = []
            action_logit = self.actor.get_dist(obs_batch)
            # action_logit = torch.distributions.Normal(mu_, std_)
            action_log_probs.append(action_logit.log_prob(actions_batch))
            # 计算动作概率分布的熵，熵的值越高，表示该分布的不确定性越大 action_logit.entropy()
            dist_entropy.append(action_logit.entropy().mean())
            action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)
            dist_entropy = dist_entropy[0]

            # values = self.critic([obs_batch, feature_p_batch])
            # values = self.critic(w_obs_batch)
            imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
            surr1 = imp_weights * adv_targ_batch
            surr2 = torch.clamp(imp_weights,1.0-self.clip_param,1.0 + self.clip_param) * adv_targ_batch
            policy_action_loss = -torch.sum(torch.min(surr1, surr2),dim=-1,keepdim=True).mean()
            policy_loss = policy_action_loss
            self.actor_optim.zero_grad()
            (policy_loss - dist_entropy * self.entrop_coef).backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(),4)
            self.actor_optim.step()

            return policy_loss,imp_weights,dist_entropy

        def critic_learn():
            values = self.critic(w_obs_batch)
            value_pred_clipped = value_pred_batch + (values - value_pred_batch).clamp(-self.clip_param,self.clip_param)
            error_clipped = returns_batch - value_pred_clipped
            error_original = returns_batch - values
            value_loss_clipped = error_clipped ** 2 / 2
            value_loss_original = error_original ** 2 / 2
            value_loss = (torch.max(value_loss_original,value_loss_clipped)).mean()
            self.critic_optim.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(),5)
            self.critic_optim.step()
            return value_loss

        policy_loss, imp_weights,dist_entropy = actor_learn()
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
