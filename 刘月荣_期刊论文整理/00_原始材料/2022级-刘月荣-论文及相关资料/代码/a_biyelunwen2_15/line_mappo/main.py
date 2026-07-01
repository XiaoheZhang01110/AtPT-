import argparse
import os
from sim import Sim_Engine
from sim import util as U
import numpy as np
import copy
from random import seed
import torch
import warnings
warnings.filterwarnings("ignore")

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import numpy.ma as ma
parser = argparse.ArgumentParser(description='param')
parser.add_argument("--seed", type=int, default=1)  # random seed
parser.add_argument("--model", type=str, default='mappo')  # amappo caac  ddpg maddpg
parser.add_argument("--data", type=str, default='A_0_1')  # used data prefix
parser.add_argument("--para_flag", type=str, default='A_0_1')  # stored parameter prefix

parser.add_argument("--episode", type=int, default=301)  # training episode
parser.add_argument("--overtake", type=int, default=0)  # overtake=0: not allow overtaking
parser.add_argument("--arr_hold", type=int, default=1)  # arr_hold=1: determine holding once bus arriving bus stop
parser.add_argument("--train", type=int, default=1)  # train=1: training phase
parser.add_argument("--restore", type=int, default=0)  # restore=1: restore the model
parser.add_argument("--all", type=int,default=0)  # all=0 for considering only forward/backward buses; all=1 for all buses
parser.add_argument("--vis", type=int, default=0)  # vis=1 to visualize bus trajectory in test phase
parser.add_argument("--weight", type=int, default=2)  # weight for action penalty

parser.add_argument("--control", type=int,default=2)  # 0 for no control;  1 for FH; 2 for RL (ddpg, maddpg) 3BH;4HH;5SH;
parser.add_argument("--share_scale", type=int, default=1)  # 0 non-share, 1 route-share
parser.add_argument("--use_tanh", type=float, default=True)
parser.add_argument("--use_orthogonal_init", type=bool, default=True)
parser.add_argument("--policy_dist", type=str, default="Beta", help="Beta or Gaussian")
parser.add_argument("--use_state_norm", type=float, default=False)
parser.add_argument("--use_reward_norm", type=float, default=False)
parser.add_argument("--use_reward_scaling", type=float, default=False)
parser.add_argument("--speed_type", type=int, default=2)  # 默认1——为匀速
parser.add_argument("--stop_num", type=int, default=39)  # 默认25
parser.add_argument("--bus_num", type=int, default=24)  # 默认30
args = parser.parse_args()

if args.model == 'caac':
    from model.CAAC import Agent
if args.model == 'ddpg':
    from model.DDPG import Agent
if args.model == 'maddpg':
    from model.MADDPG import Agent
if args.model == 'mappo':
    from model.MAPPO import Agent

# if torch.cuda and torch.cuda.is_available():  # 判断是否cuda可以使用，就是是否能用GPU
#     print("choose to use gpu...")
#     device = torch.device("cuda:0")  # 指定使用哪个GPU
# else:
#     print("choose to use cpu...")
#     device = torch.device("cpu")

def train(args):
    stop_list, pax_num = U.getStopList(args)    # stop_list=46,字典   pax_num=0
    print('Stops prepared, total bus stops: %g' % (len(stop_list)))
    bus_routes = U.getBusRoute(stop_list,args)    # 0_1:里面包含59辆公交车
    print('Bus routes prepared, total routes :%g' % (len(bus_routes)))
    stop_list_ = copy.deepcopy(stop_list)
    dispatch_times, bus_list, route_list, simulation_step = U.init_bus_list(bus_routes,args)
    print('init...')

    agents = {}    # 下面的if只是为了创建agent
    if args.model != '':
        eng = Sim_Engine.Engine(bus_list=bus_list, busstop_list=stop_list_, control_type=args.control,
                                dispatch_times=dispatch_times,
                                demand=0, simulation_step=simulation_step, route_list=route_list,
                                hold_once_arr=args.arr_hold, is_allow_overtake=args.overtake,
                                share_scale=args.share_scale, weight=args.weight, use_state_norm=args.use_state_norm,
                                use_reward_norm=args.use_reward_norm, use_reward_scaling=args.use_reward_scaling)

        bus_list = eng.bus_list
        bus_stop_list = eng.busstop_list
        state_dim = 3
        # non share
        if args.share_scale == 0:  # false
            for k, v in eng.bus_list.items():
                agent = Agent(state_dim=state_dim, name='', n_stops=len(bus_stop_list), buslist=bus_list,
                              seed=args.seed)
                agents[k] = agent

        # share in route
        if args.share_scale == 1:  # true
            agents = {}
            for k, v in eng.route_list.items():  # 只有一条线路  线路为一个智能体
                agent = Agent(state_dim=state_dim, name='', n_stops=len(bus_stop_list), buslist=bus_list,
                              seed=args.seed)   # 每个agent都有一个crtic一个target_crtic,以及一个actor一个target_actor
                agents[k] = agent
    for ep in range(args.episode):    # args.episode=401
        stop_list_ = copy.deepcopy(stop_list)   # 对stop_list的深拷贝，保持了对象之间的独立性，避免副作用和数据共享问题
        bus_list_ = copy.deepcopy(bus_list)
        r =  [0.1 / 60, 0.5 / 60, 0.1 / 60, 0.6/ 60, 0.4 / 60, 0.1 / 60, 0.5 / 60, 0.7 / 60, 0.2 / 60, 0.1 / 60,
             0.4 / 60, 0.3 / 60, 0.2 / 60, 1.0 / 60, 1.5 / 60, 0.8 / 60, 1.4 / 60, 2.0 / 60, 1.7 / 60, 1.3 / 60,
             2.1 / 60, 1.7 / 60, 1.1 / 60, 1.5 / 60, 1.7 / 60, 1.5 / 60, 0.5 / 60, 0.8 / 60, 0.2 / 60, 0.3 / 60,
             0.6 / 60, 0.5 / 60, 0.3 / 60, 0.2 / 60, 0.1 / 60, 0.8 / 60, 0.3 / 60, 0.2 / 60, 0.0]
        # r = [0.1 / 60, 0.3 / 60, 0.1 / 60, 0.7 / 60, 0.1 / 60, 0.1/ 60, 1./ 60, 0.1 / 60, 0.2 / 60, 0.1 / 60,
        #  0.4 / 60, 0.4 / 60, 0.2/ 60, 1.2 / 60, 1.1 / 60, 1.2 / 60, 1.3 / 60, 0.7 / 60, 1.7 / 60, 1.3 / 60,
        #  1.1 / 60, 2.3 / 60, 1.3 / 60, 2.9 / 60, 2.2 / 60, 2.5 / 60, 1.2 / 60, 0.8 / 60,0.2/ 60,0.3 / 60,
        #  0.6 / 60, 0.5 / 60, 0.3 / 60, 0.2 / 60,0.1 / 60, 2.2 / 60, 0.3 / 60, 0.2 / 60, 0.0]

        for _, stop in stop_list_.items():
            stop.set_rate(r[int(_)-1])

        eng = Sim_Engine.Engine(bus_list=bus_list_, busstop_list=stop_list_, control_type=args.control,
                                dispatch_times=dispatch_times,
                                demand=0, simulation_step=simulation_step, route_list=route_list,
                                hold_once_arr=args.arr_hold, is_allow_overtake=args.overtake,
                                share_scale=args.share_scale, weight=args.weight, use_state_norm=args.use_state_norm,
                                use_reward_norm=args.use_reward_norm, use_reward_scaling=args.use_reward_scaling)

        eng.agents = agents
        if ep > 0:    # ep=0时，也就是第一轮训练时，下面的两个if条件都不满足要求，都不执行
            if memory_copy != None:
                eng.GM = memory_copy
            for bid, b in eng.bus_list.items():
                if b.is_virtual == 0:
                    eng.GM.temp_memory[bid] = {'s':[],'a':[],'a_log_prob':[],'fp':[],'r':[]}
        # args.restore == 0
        if args.restore == 1 and args.control > 1:
            for k, v in agents.items():
                print(str(args.para_flag) + str('_') + str(args.share_scale) + str('_') + str(args.model) + str('_'))
                v.load(str(args.para_flag) + str('_') + str(args.share_scale) + str('_') + str(args.weight) + str(
                    '_') + str(args.model) + str('_'))

        Flag = True
        while Flag:
            Flag = eng.sim()
        ploss_log = []
        qloss_log = []
        if args.control ==2  and args.restore == 0:  # args.control =2 应该是表示强化学习
            if ep >= 0:  # 代表仿真的轮数
                train_infos = eng.learn()
                ploss_log.append(train_infos['policy_loss'])
                qloss_log.append(train_infos['value_loss'])
            if ep % 20 == 0 and ep > 10 and args.restore == 0:
                # store model
                for k, v in agents.items():
                    v.save(str(args.para_flag) + str('_') + str(args.share_scale) + str('_') + str(args.weight) + str(
                        '_') + str(args.model) + str('_'))

        if args.control > 1:
            memory_copy = eng.GM
        else:
            memory_copy = None

        log = eng.cal_statistic(
            name=str(args.para_flag) + str('_') + str(args.share_scale) + str('_') + str(args.model) + str('_'),train=args.train)
        abspath = os.path.abspath(os.path.dirname(__file__))
        name = abspath + "/log/" + args.data + args.model  + str('_') + str(args.control) + str('_')

        name += str(int(args.weight))
        if args.all == 1:
            name += 'all_r'
        # 每运行一个ep就记录一次
        U.train_result_track(eng=eng, ep=ep, qloss_log=qloss_log, ploss_log=ploss_log, log=log, name=name,
                             seed=args.seed)
        eng.close()
    U.visualize_trajectory(eng, name=abspath + "/log" + "/trajectory/")
    # 累积奖励
    df1 = pd.read_csv(name + str(args.seed) + '.csv')
    f = plt.figure()
    ax = plt.subplot(111)
    ax.tick_params(length=4, width=0.5)
    plt.xlabel('Training episode')
    plt.ylabel('Cumulative global reward')
    smoothing_window = 10
    rewards_smoothed = pd.Series(df1['reward']).rolling(smoothing_window,
                                                        min_periods=smoothing_window).mean()  # 智能体获得的平均奖励
    rewards_smoothed1 = pd.Series(df1['reward1']).rolling(smoothing_window,
                                                          min_periods=smoothing_window).mean()
    rewards_smoothed2 = pd.Series(df1['reward2']).rolling(smoothing_window,
                                                          min_periods=smoothing_window).mean()
    plt.plot(df1['reward'], alpha=0.2)
    plt.plot(rewards_smoothed, color='orange', label='reward')
    plt.plot(df1['reward1'], alpha=0.2)
    plt.plot(rewards_smoothed1, color='red', label='reward1')
    plt.plot(df1['reward2'], alpha=0.2)
    plt.plot(rewards_smoothed2, color='green', label='reward2')
    plt.grid()
    ax.legend(loc='best', fancybox=True, shadow=False, ncol=1, prop={'size': 12})
    plt.show()

    # 价值网络的损失函数
    f = plt.figure()
    ax = plt.subplot(111)
    ax.tick_params(length=4, width=0.5)
    plt.xlabel('Training episode')
    plt.ylabel('Qloss')
    smoothing_window = 10
    q_loss_set_smoothed = pd.Series(df1['qloss']).rolling(smoothing_window,
                                                          min_periods=smoothing_window).mean()  # 智能体获得的平均奖励
    plt.plot(df1['qloss'], alpha=0.2)
    plt.plot(q_loss_set_smoothed, color='orange', label='AMAPPO')
    plt.grid()
    ax.legend(loc='best', fancybox=True, shadow=False, ncol=1, prop={'size': 12})
    plt.show()

    ### 串车次数
    f = plt.figure()
    ax = plt.subplot(111)
    ax.tick_params(length=4, width=0.5)
    plt.xlabel('Training episode')
    plt.ylabel('bunching')
    smoothing_window = 10
    rewards_smoothed = pd.Series(df1['bunching']).rolling(smoothing_window,
                                                          min_periods=smoothing_window).mean()  # 智能体获得的平均奖励
    plt.plot(df1['bunching'], alpha=0.2)
    plt.plot(rewards_smoothed, color='orange', label='AMAPPO')
    plt.grid()
    ax.legend(loc='best', fancybox=True, shadow=False, ncol=1, prop={'size': 12})
    plt.show()

    ### 所有乘客的平均等待时间(等待上车)
    f = plt.figure()
    ax = plt.subplot(111)
    ax.tick_params(length=4, width=0.5)
    plt.xlabel('Training episode')
    plt.ylabel('Wait')
    smoothing_window = 10
    rewards_smoothed = pd.Series(df1['wait']).rolling(smoothing_window,
                                                      min_periods=smoothing_window).mean()  # 智能体获得的平均奖励

    plt.plot(df1['wait'], alpha=0.2)
    plt.plot(rewards_smoothed, color='orange', label='AMAPPO')
    plt.grid()
    ax.legend(loc='best', fancybox=True, shadow=False, ncol=1, prop={'size': 12})
    plt.show()

    ### 所有智能体的平均在车时间
    f = plt.figure()
    ax = plt.subplot(111)
    ax.tick_params(length=4, width=0.5)
    plt.xlabel('Training episode')
    plt.ylabel('Travel')
    smoothing_window = 10
    rewards_smoothed = pd.Series(df1['travel']).rolling(smoothing_window,
                                                        min_periods=smoothing_window).mean()  # 智能体获得的平均奖励
    plt.plot(df1['travel'], alpha=0.2)
    plt.plot(rewards_smoothed, color='orange', label='AMAPPO')
    plt.grid()
    ax.legend(loc='best', fancybox=True, shadow=False, ncol=1, prop={'size': 12})
    plt.show()

    ### 所有乘客由于驻站时间造成的延误  下车-到达-在车车辆运行
    f = plt.figure()
    ax = plt.subplot(111)
    ax.tick_params(length=4, width=0.5)
    plt.xlabel('Training episode')
    plt.ylabel('Delay')
    smoothing_window = 10
    rewards_smoothed = pd.Series(df1['delay']).rolling(smoothing_window,
                                                       min_periods=smoothing_window).mean()  # 智能体获得的平均奖励
    plt.plot(df1['delay'], alpha=0.2)
    plt.plot(rewards_smoothed, color='orange', label='AMAPPO')
    plt.grid()
    ax.legend(loc='best', fancybox=True, shadow=False, ncol=1, prop={'size': 12})
    plt.show()

    df11 = pd.read_csv(name + str(args.seed) + 'res.csv')
    df11 = df11.rename(columns={'Unnamed: 0': 'stop'})
    # df11['sth'] = df11['sth'].str.strip('[]').astype(float)
    df11['sth'] = df11['sth'].astype(float)

    df11 = df11.groupby('stop').mean()
    f = plt.figure()
    ax = plt.subplot(111)
    ax.tick_params(length=4, width=0.5)
    plt.xlabel('Stops')
    plt.ylabel('AWT')
    plt.plot(df11['stw'], color='orange', label='AMAPPO')
    plt.grid()
    ax.legend(loc='best', fancybox=True, shadow=False, ncol=1, prop={'size': 12})
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()

    f = plt.figure()
    ax = plt.subplot(111)
    ax.tick_params(length=4, width=0.5)
    plt.xlabel('Stops')
    plt.ylabel('AHT')
    plt.plot(df11['sth'], color='orange', label='AMAPPO')
    plt.grid()
    ax.legend(loc='best', fancybox=True, shadow=False, ncol=1, prop={'size': 12})
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()

    f = plt.figure()
    ax = plt.subplot(111)
    ax.tick_params(length=4, width=0.5)
    plt.xlabel('Stops')
    plt.ylabel('AOD')
    plt.plot(df11['sto'], color='orange', label='AMAPPO')
    plt.grid()
    ax.legend(loc='best', fancybox=True, shadow=False, ncol=1, prop={'size': 12})
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()


def evaluate(args):
    stop_list, pax_num = U.getStopList(args.data)
    print('Stops prepared, total bus stops: %g' % (len(stop_list)))
    bus_routes = U.getBusRoute(args.data)
    print('Bus routes prepared, total routes :%g' % (len(bus_routes)))
    dispatch_times, bus_list, route_list, simulation_step = U.init_bus_list(bus_routes)
    agents = {}
    if args.model != '':
        stop_list_ = copy.deepcopy(stop_list)
        eng = Sim_Engine.Engine(bus_list=bus_list, busstop_list=stop_list_, control_type=args.control,
                                dispatch_times=dispatch_times,
                                demand=0, simulation_step=simulation_step, route_list=route_list,
                                hold_once_arr=args.arr_hold, is_allow_overtake=args.overtake,
                                share_scale=args.share_scale, weight=args.weight)

        bus_list = eng.bus_list
        # non share
        if args.share_scale == 0:
            for k, v in eng.bus_list.items():
                state_dim = 3
                agent = Agent(state_dim=state_dim, name=k, n_stops=len(eng.busstop_list), buslist=eng.bus_list,
                              seed=args.seed)
                agents[k] = agent

        # share in route
        if args.share_scale == 1:
            agents = {}
            for k, v in eng.route_list.items():
                state_dim = 3
                if args.model == 'accf':
                    agent = Agent(state_dim=state_dim, name='', n_stops=len(eng.busstop_list), buslist=eng.bus_list,
                                  seed=args.seed)
                else:
                    agent = Agent(state_dim=state_dim, name='', n_stops=len(eng.busstop_list), buslist=eng.bus_list,
                                  seed=args.seed)
                agents[k] = agent

    rs = [np.random.randint(10, 20) / 10. for _ in range(10)]



    AWT = []
    AHD = []
    AOD = []
    headways_var = []
    REWARD = []
    for ep in range(10):
        stop_list_ = copy.deepcopy(stop_list)
        bus_list_ = copy.deepcopy(bus_list)
        r = rs[ep]
        if args.vis == 1:
            r = 1.5

        for _, stop in stop_list_.items():
            stop.set_rate(r)

        eng = Sim_Engine.Engine(bus_list=bus_list_, busstop_list=stop_list_, control_type=args.control,
                                dispatch_times=dispatch_times,
                                demand=0, simulation_step=simulation_step, route_list=route_list,
                                hold_once_arr=args.arr_hold, is_allow_overtake=args.overtake,
                                share_scale=args.share_scale, weight=args.weight)

        eng.agents = agents
        s = str(args.para_flag) + str('_') + str(args.share_scale) + str('_') + str(args.weight) + str('_') + str(
            args.model) + str('_')
        if args.restore == 1 and args.control > 1:
            for k, v in agents.items():
                v.load(s)

        Flag = True
        while Flag:
            Flag = eng.sim()

        log,AWT_,AHD_,AOD_,headways_var_ = eng.cal_statistic(
            name=str(args.para_flag) + str('_') + str(args.share_scale) + str('_') + str(args.model) + str('_'),
            train=args.train)

        AWT.append(AWT_)
        AHD.append(AHD_)
        AOD.append(AOD_)
        headways_var.append(headways_var_)

        abspath = os.path.abspath(os.path.dirname(__file__))
        if args.control == 0:
            name = abspath + "/logt/" + args.data + 'nc'

        if args.control == 2:
            name = abspath + "/logt/" + args.data + args.model

            name += str(int(args.weight))
            if args.all == 1:
                name += 'all'

        if args.control == 1:
            name = abspath + "/logt/" + args.data + 'fc'

        reward_,_,_,_,_,_,_ = U.train_result_track(eng=eng, ep=ep, qloss_log=[0], ploss_log=[0], log=log, name=name,
                             seed=args.seed)
        REWARD.append(reward_)
        if args.vis == 1 and args.data == 'SG0':
            if args.control == 0:
                name = abspath + "/vis/visnc/"
            if args.control == 1:
                name = abspath + "/vis/visfc/"
            if args.control == 2:
                name = abspath + "/vis/vis" + args.model + '/'
            try:
                os.makedirs(name)
            except:
                print(name, ' has existed')
            U.visualize_trajectory(engine=eng, name=name + '_' + str(args.data) + str('_'))
            break

        eng.close()

    f = plt.figure()
    ax = plt.subplot(111)
    ax.tick_params(length=4, width=0.5)
    plt.xlabel('evaluate episode')
    plt.ylabel('AWT')
    smoothing_window = 10
    AWT_smoothed = pd.Series(AWT).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(AWT, alpha=0.2)
    plt.plot(AWT_smoothed, color='orange')
    plt.grid()  # 用于画坐标网格线
    plt.show()  # 用于显示图片

    f = plt.figure()
    ax = plt.subplot(111)
    ax.tick_params(length=4, width=0.5)
    plt.xlabel('evaluate episode')
    plt.ylabel('AHD')
    smoothing_window = 10
    AHD_smoothed = pd.Series(AHD).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(AHD, alpha=0.2)
    plt.plot(AHD_smoothed, color='orange')
    plt.grid()  # 用于画坐标网格线
    plt.show()  # 用于显示图片

    f = plt.figure()
    ax = plt.subplot(111)
    ax.tick_params(length=4, width=0.5)
    plt.xlabel('evaluate episode')
    plt.ylabel('AOD')
    smoothing_window = 10
    AOD_smoothed = pd.Series(AOD).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(AOD, alpha=0.2)
    plt.plot(AOD_smoothed, color='orange')
    plt.grid()  # 用于画坐标网格线
    plt.show()  # 用于显示图片

    f = plt.figure()
    ax = plt.subplot(111)
    ax.tick_params(length=4, width=0.5)
    plt.xlabel('evaluate episode')
    plt.ylabel('headways_var')
    smoothing_window = 10
    headways_smoothed = pd.Series(headways_var).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(headways_var, alpha=0.2)
    plt.plot(headways_smoothed, color='orange')
    plt.grid()  # 用于画坐标网格线
    plt.show()  # 用于显示图片

    f = plt.figure()
    ax = plt.subplot(111)
    ax.tick_params(length=4, width=0.5)
    plt.xlabel('evaluate episode')
    plt.ylabel('REWARD')
    smoothing_window = 10
    REWARD_smoothed = pd.Series(REWARD).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(REWARD, alpha=0.2)
    plt.plot(REWARD_smoothed, color='orange')
    plt.grid()  # 用于画坐标网格线
    plt.show()  # 用于显示图片

if __name__ == '__main__':

    seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.train == 1:    # 默认设置为1
        train(args)

    else:
        evaluate(args)
