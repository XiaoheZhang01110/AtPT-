import torch
import numpy as np
import torch.nn.functional as F
class build_policy_estimator(torch.nn.Module):
    def __init__(self,state_dim,policy_dim):
        super(build_policy_estimator,self).__init__()
        self.fc1 = torch.nn.Linear(state_dim,512)
        self.fc2 = torch.nn.Linear(512,policy_dim)

    def forward(self,state):
        x = F.relu(self.fc1(state))
        estimate = F.elu(self.fc2(x))
        return estimate

class policy_estimator():
    def __init__(self,state_dim,policy_dim):
        self.pe = build_policy_estimator(state_dim,policy_dim)
        self.pe_optim = torch.optim.Adam(self.pe.parameters(), lr=0.0006)

    def estimate_action(self,s_temp):
        estimate = self.pe(s_temp)[0]
        return estimate

    def update(self,actual_action,estimate_action):
        actual_action = torch.tensor(actual_action,dtype=torch.float)
        pe_loss = torch.mean(F.mse_loss(actual_action,estimate_action))
        self.pe_optim.zero_grad()
        pe_loss.backward()
        self.pe_optim.step()
        return pe_loss