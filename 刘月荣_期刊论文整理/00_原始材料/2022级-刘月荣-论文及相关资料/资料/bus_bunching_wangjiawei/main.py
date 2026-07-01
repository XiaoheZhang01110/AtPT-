import warnings
warnings.filterwarnings('ignore')
from BUS import bus
from BUS_STOP import bus_stop
from Env import Env
from brain import PPO
from policy_estimator import policy_estimator
import torch
import numpy as np
import random
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc("font",**{"family":"sans-serif","sans-serif":["Helvetica","Arial"],"size":14})
matplotlib.rc('pdf', fonttype=42, use14corefonts=True, compression=6)
matplotlib.rc('ps', useafm=True, usedistiller='none', fonttype=42)
matplotlib.rc("axes", unicode_minus=False, linewidth=1, labelsize='medium')
matplotlib.rc("axes.formatter", limits=[-7,7])
# matplotlib.rc('savefig', bbox='tight', format='pdf', frameon=False, pad_inches=0.05)
# matplotlib.rc('lines', marker=None, markersize=4)
matplotlib.rc('text', usetex=False)
matplotlib.rc('xtick', direction='in')
matplotlib.rc('xtick.major', size=8)
matplotlib.rc('xtick.minor', size=2)
matplotlib.rc('ytick', direction='in')
matplotlib.rc('lines', linewidth=1)
matplotlib.rc('ytick.major', size=8)
matplotlib.rc('ytick.minor', size=2)
matplotlib.rcParams['lines.solid_capstyle'] = 'butt'
matplotlib.rcParams['lines.solid_joinstyle'] = 'bevel'
matplotlib.rc('mathtext', fontset='stixsans')
matplotlib.rc('legend', fontsize='medium', frameon=False,handleheight=0.5, handlelength=1, handletextpad=0.4,numpoints=1)

random.seed(1)
num_episodes = 300
updateStep = 1

pe = policy_estimator(state_dim=18,policy_dim=6)
state_dim_loacl = 3
state_dim = 3+6
models = []
for i in range(6):
    model = PPO(state_dim_loacl=state_dim_loacl,state_dim=state_dim,action_dim=1,eps=0.2,epochs=5,critic_lr=0.001,actor_lr=0.0001,gamma=0.9)
    models.append(model)

records = [0 for i in range(6)]
reward_set = []
reward_each_agent=[[] for i in range(6)]
reward_set_r = []
reward_set1=[]
reward_set2=[]
v_loss_set = []
flag = 0
asets = []
catching_time = []
policy_buffer = []
pe_loss_set=[]
state_collect = []
w1 = 0.8
w2 = 1.

control_id = [i for i in range(12)]
bus_stop_list = [np.pi * 2 / 12 * (i) for i in range(12)]
for i in range(num_episodes):
    env = Env(state_dim=3,action_dim=1,bus_num=6,bus_stop_num=12,r=120,emit_time_list=[0. * 3/30 for _ in range(6)],
              bus_dep_list=[0 for _ in range(6)], bus_stop_loc_list=bus_stop_list,
              sim_horizon=60*60*3,train_mode=0,control_id=control_id)

    rewards = 0
    j = 0
    buffer_r_c = [[] for _ in range(6)]
    buffer_a_c = [[] for _ in range(6)]
    buffer_s_c = [[] for _ in range(6)]
    buffer_s_c_all = [[] for _ in range(6)]
    temp_r = []
    temp_r1 = []
    temp_r2 = []
    while True:
        is_sim_over = env.sim()
        if is_sim_over < 0:
            cost = np.array(env.cost).reshape(-1, )
            if is_sim_over == -2 and env.flag == 1:
                r = np.array(env.reward).reshape(-1, )
            if is_sim_over == -3 and env.flag == 1:
                r = np.array(env.reward).reshape(-1, )
            a = np.array(a).reshape(-1,)
            s_temp = []
            for c in range(len(s)):
                if len(s[c]) == 0:
                    s_temp.append([-1., -1., -1.])
                else:
                    s_temp.append(s[c][:])
            policy_buffer.append(np.array(s_temp).reshape(-1).tolist()[:] + np.array(a).reshape(-1).tolist()[:])
            # 进行策略估计
            estimate = pe.estimate_action(torch.tensor(np.array(s_temp),dtype=torch.float).view(1,-1))
            estimate[estimate < 0.01] = 0

            s_next = env.state[:]
            env.state = []
            v_s_ = [0 for v_id in range(6)]
            for bus_id in range(6):
                if len(s[bus_id][:]) > 0:
                    buffer_s_c[bus_id].append(s[bus_id][:])
                    proxy = estimate[:bus_id].tolist() + estimate[bus_id + 1:].tolist()
                    buffer_s_c_all[bus_id].append(s[bus_id][:] + proxy)
                    buffer_a_c[bus_id].append(a[bus_id])
                if len(buffer_s_c[bus_id]) > len(buffer_r_c[bus_id]):
                    buffer_r_c[bus_id].append(w1 * np.exp(-buffer_a_c[bus_id][-1]) + w2 * np.exp(- abs(s_next[bus_id][-1] - s_next[bus_id][-2])))
                    temp_r1.append(np.exp(-buffer_a_c[bus_id][-1]))
                    temp_r2.append(np.exp(- abs(s_next[bus_id][-1] - s_next[bus_id][-2])))
                else:
                    v_s_[bus_id] = 0
                temp_r.append(r)

            rewards2 = 0
            k = 0
            temp_q = []
            for c in range(6):
                if len(buffer_r_c[c]) > 0:
                    rewards2 += sum(buffer_r_c[c])
                    k += len(buffer_r_c[c])
                    reward_each_agent[c].append(sum(buffer_r_c[c]) / len(buffer_r_c[c]))

                    records[c] += 1
                    advantage,log_probs,old_log_probs,ratio,surr1,surr2,actor_loss,critic_loss = models[0].update(states_all=buffer_s_c_all[c][:-1],n_states_all=buffer_s_c_all[c][1:],states=buffer_s_c[c][:-1],actions=buffer_a_c[c][:-1],buffer_r_c=buffer_r_c[c][:-1])

                    temp_q.append(critic_loss.item())
                    buffer_r_c[c] = []
                    buffer_a_c[c] = []
                    buffer_s_c[c] = []
                    buffer_s_c_all[c] = []
                    print('1 bus:%d  vloss: %g ploss: %g p1:%g p2:%g adv:%g sg1：%g sg2：%g' % (c, critic_loss,actor_loss,log_probs.mean(),old_log_probs.mean(),advantage.mean(),surr1.mean(),surr2.mean()))
            v_loss_set.append(sum(temp_q)/len(temp_q))
            if k > 0:
                rewards2 = rewards2 / k
            # 更新策联合动作追踪网络
            # 实际执行的动作
            actual_action = np.array(policy_buffer)[:,18:]
            # 估计动作
            policy_state_all = torch.tensor(np.array(policy_buffer)[:,0:18],dtype=torch.float)   # 状态
            estimate_action = pe.estimate_action(policy_state_all)
            pe_loss = pe.update(actual_action,estimate_action)
            print('pe loss %g' % pe_loss)

            pe_loss_set.append(pe_loss.detach().item())
            policy_buffer = []
            break

        if is_sim_over == 0:
            if env.flag == 0:
                bl = env.bus_loc_stop[:]
                s = env.state[:]
                state_collect.append((s[0][:]+[env.bus_list[0].travel_sum]))
                a = []
                for c in range(6):
                    if len(s[c][:]) > 0:
                        a.append(np.clip(abs(models[0].take_action(s[c][:])),0.,1.))
                    else:
                        a.append(0.)
                env.control(a)
                env.state = []
                env.flag = 1
            else:
                cost = np.array(env.cost).reshape(-1,)
                a = np.array(a).reshape(-1,)
                r = np.array(env.reward).reshape(-1,)

                s_temp = []
                for c in range(len(s)):
                    if len(s[c]) == 0:
                        s_temp.append([-1,-1,-1])
                    else:
                        s_temp.append(s[c][:])
                policy_buffer.append(np.array(s_temp).reshape(-1).tolist()[:] + np.array(a).reshape(-1).tolist()[:])
                # 进行策略估计
                estimate = pe.estimate_action(torch.tensor(np.array(s_temp),dtype=torch.float).view(1,-1))
                estimate[estimate < 0.01] = 0

                for bus_id in range(6):
                    if len(s[bus_id][:]) <= 0:
                        estimate[bus_id] = 0
                s_next = env.state[:]
                bl = env.bus_loc_stop[:]
                for bus_id in range(6):
                    if len(s[bus_id][:]) > 0:
                        buffer_s_c[bus_id].append(s[bus_id][:])
                        proxy = estimate[:bus_id].tolist() + estimate[bus_id + 1:].tolist()
                        buffer_s_c_all[bus_id].append(s[bus_id][:] + proxy)
                        buffer_a_c[bus_id].append(a[bus_id])
                    if len(s_next[bus_id][:]) > 0 and len(buffer_s_c[bus_id]) > 0:
                        buffer_r_c[bus_id].append(w1 * np.exp(-buffer_a_c[bus_id][-1]) + w2 * np.exp(-abs(s_next[bus_id][-1] - s_next[bus_id][-2])))
                        temp_r.append(r)
                        temp_r1.append(np.exp(-buffer_a_c[bus_id][-1]))
                        temp_r2.append(np.exp(-abs(s_next[bus_id][-1] - s_next[bus_id][-2])))

                a = []
                s = s_next[:]
                state_collect.append((s[0][:] + [env.bus_list[0].travel_sum]))
                for bus_id in range(6):
                    if len(s[bus_id]) > 0:
                        a.append(np.clip(abs(models[0].take_action(s[bus_id][:])),0.,1.))
                    else:
                        a.append(0.)
                env.control(a)
                env.state = []
                j += 1

    print(' num_episodes:%d   r:%g realvar: %g' % (i, rewards2, sum(temp_r) / len(temp_r)))
    print(records)
    reward_set.append(sum(temp_r) / len(temp_r))
    reward_set1.append(sum(temp_r1) / len(temp_r1))
    reward_set2.append(sum(temp_r2) / len(temp_r2))
    reward_set_r.append(rewards2)
csfont = {'size': 18}
f = plt.figure()
ax = plt.subplot(111)
ax.tick_params(length=4, width=0.5)
plt.xlabel('Training episode')
plt.ylabel('Mean squared error')
smoothing_window = 10
v_loss_set_smoothed = pd.Series(v_loss_set).rolling(smoothing_window, min_periods=smoothing_window).mean()

data = pd.DataFrame(v_loss_set)
data.to_csv('v_loss_set.csv')
plt.plot(v_loss_set, alpha=0.2)  # 用于绘制 v_loss_set 数据的折线图，alpha=0.2 设置线条的透明度为 0.2（值函数的损失）
plt.plot(v_loss_set_smoothed, color='orange')
plt.grid()  # 用于画坐标网格线
plt.show()  # 用于显示图片
f.savefig("critic.pdf", bbox_inches='tight')

f = plt.figure()
ax = plt.subplot(111)
ax.tick_params(length=4, width=0.5)  # 是用于设置坐标轴刻度线的参数。 length=4 设置刻度线的长度为 4。width=0.5 设置刻度线的宽度为 0.5。
plt.xlabel('Training episode')
plt.ylabel('Cumulative global reward')
smoothing_window = 10
rewards_smoothed = pd.Series(reward_set_r).rolling(smoothing_window, min_periods=smoothing_window).mean()  # 智能体获得的平均奖励
rewards_smoothed1 = pd.Series(reward_set1).rolling(smoothing_window, min_periods=smoothing_window).mean()
rewards_smoothed2 = pd.Series(reward_set2).rolling(smoothing_window, min_periods=smoothing_window).mean()

plt.plot(reward_set_r, alpha=0.2)
plt.plot(rewards_smoothed, color='orange', label='total reward')

plt.plot(reward_set1, alpha=0.2)
plt.plot(rewards_smoothed1, color='red',label='reward for holding penalty')

plt.plot(reward_set2, alpha=0.2)
plt.plot(rewards_smoothed2, color='green',label='reward for headway equalization')
plt.grid()
ax.legend(loc='best',  fancybox=True, shadow=False, ncol=1, prop={'size': 12})
plt.show()



f=plt.figure()
ax = plt.subplot(111)
ax.tick_params(length=4, width=0.5)
# ax.tick_params(axis='x', which='major', labelsize=14)
# ax.tick_params(axis='y', which='major', labelsize=14)
plt.xlabel('Training episode' )
plt.ylabel('Average of cumulative reward for each agent in each episode' )
smoothing_window = 10
for i in range(6):
    rewards_smoothed = pd.Series(reward_each_agent[i]).rolling(smoothing_window,min_periods=smoothing_window).mean()    # 该处智能体的奖励是论文中算出来的奖励
    # plt.plot(reward_each_agent[i], alpha=0.2)
    plt.plot(rewards_smoothed, label='bus  ' + str(i))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
              fancybox=True, shadow=False, ncol=3)
plt.grid()
plt.show()



f = plt.figure()
ax = plt.subplot(111)
ax.tick_params(length=4, width=0.5)
smoothing_window = 10
plt.xlabel('Training episode',**csfont)
plt.ylabel('Cumulative global reward in each episode',**csfont)
rewards_smoothed = pd.Series(reward_set).rolling(smoothing_window, min_periods=smoothing_window).mean()   # 与车头时距相关的奖励
plt.plot(reward_set, alpha=0.2)
plt.plot(rewards_smoothed,color='orange')
plt.grid()
plt.show()



f = plt.figure()
ax = plt.subplot(111)
ax.tick_params(length=4, width=0.5)
plt.xlabel('Training step'  )
plt.ylabel('Mean squared error'  )

pe_smoothed = pd.Series(pe_loss_set).rolling(smoothing_window, min_periods=smoothing_window).mean()   # 这儿有问题
pe_loss_set_df = pd.DataFrame(pe_loss_set)
pe_loss_set_df.to_csv('pe_loss_set.csv')
plt.plot(pe_loss_set, alpha=0.2)
plt.plot(pe_smoothed,color='orange')
plt.grid()
plt.show()