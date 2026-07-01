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
import pandas as pd
import numpy.ma as ma
import math
matplotlib.rc("font",**{"family":"sans-serif","sans-serif":["Helvetica","Arial"],"size":14})
matplotlib.rc('pdf', fonttype=42, use14corefonts=True, compression=6)
matplotlib.rc('ps', useafm=True, usedistiller='none', fonttype=42)
matplotlib.rc("axes", unicode_minus=False, linewidth=1, labelsize='medium')
matplotlib.rc("axes.formatter", limits=[-7,7])
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
matplotlib.rc('legend', fontsize='medium', frameon=False,handleheight=0.5, handlelength=1, handletextpad=0.4, numpoints=1)



parser = argparse.ArgumentParser(description='param')
parser.add_argument("--seed", type=int, default=1)  # random seed
parser.add_argument("--model", type=str, default='')  # caac  ddpg maddpg
parser.add_argument("--data", type=str, default='A_0_1')  # used data prefix
parser.add_argument("--para_flag", type=str, default='A_0_1')  # stored parameter prefix
parser.add_argument("--episode", type=int, default=5)  # training episode
parser.add_argument("--overtake", type=int, default=0)  # overtake=0: not allow overtaking
parser.add_argument("--arr_hold", type=int, default=1)  # arr_hold=1: determine holding once bus arriving bus stop
parser.add_argument("--train", type=int, default=1)  # train=1: training phase
parser.add_argument("--restore", type=int, default=0)  # restore=1: restore the model
parser.add_argument("--all", type=int,default=1)  # all=0 for considering only forward/backward buses; all=1 for all buses
parser.add_argument("--vis", type=int, default=0)  # vis=1 to visualize bus trajectory in test phase
parser.add_argument("--weight", type=int, default=2)  # weight for action penalty
parser.add_argument("--control", type=int, default=0)  # 0 for no control;  1 for FH; 2 for RL (MAPPO, AMAPPO); 3 for BH ;4 for HH ;5 SH
parser.add_argument("--share_scale", type=int, default=1)  # 0 non-share, 1 route-share
parser.add_argument("--speed_type", type=int, default=1)  # 默认1——为匀速

args = parser.parse_args()


if args.model == 'mappo_caac':
    from model.MAPPO_CAAC import Agent
if args.model == 'mappo':
    from model.MAPPO import Agent


def train(args):
    stop_list, pax_num = U.getStopList()    # stop_list=46,字典   pax_num=0
    print('Stops prepared, total bus stops: %g' % (len(stop_list)))
    bus_routes = U.getBusRoute(stop_list,args)    # 0_1:里面包含59辆公交车
    print('Bus routes prepared, total routes :%g' % (len(bus_routes)))
    stop_list_ = copy.deepcopy(stop_list)
    dispatch_times, bus_list, route_list, simulation_step = U.init_bus_list(bus_routes)
    print('init...')
    agents = {}
    if args.model != '':
        eng = Sim_Engine.Engine(bus_list=bus_list, busstop_list=stop_list_, control_type=args.control,
                                dispatch_times=dispatch_times,
                                demand=0, simulation_step=simulation_step, route_list=route_list,
                                hold_once_arr=args.arr_hold, is_allow_overtake=args.overtake,
                                share_scale=args.share_scale, weight=args.weight)
        bus_list = eng.bus_list
        bus_stop_list = eng.busstop_list
        state_dim = 3
        # non share
        if args.share_scale == 0:
            for k, v in eng.bus_list.items():
                agent = Agent(state_dim=state_dim, name='', n_stops=len(bus_stop_list), buslist=bus_list,
                              seed=args.seed)
                agents[k] = agent

        # share in route
        if args.share_scale == 1:  # true
            agents = {}
            for k, v in eng.route_list.items():  # 只有一条线路  线路为一个智能体
                agent = Agent(state_dim=state_dim, name='', n_stops=len(bus_stop_list), buslist=bus_list,
                              seed=args.seed)  # 每个agent都有一个crtic一个target_crtic,以及一个actor一个target_actor
                agents[k] = agent
    AWT = []
    AHD = []
    AOD = []
    ATT = []
    reward_avg_list = []
    reward1_avg_list = []
    reward2_avg_list = []
    ploss_log_avg_list = []
    qloss_log_avg_list = []
    wait_cost_avg_list = []
    travel_cost_avg_list = []
    # print(agents['0_1'].actor.state_dict())
    # print(agents['0_1'].critic.state_dict())
    for ep in range(args.episode):    # args.episode=401
        stop_list_ = copy.deepcopy(stop_list)   # 对stop_list的深拷贝，保持了对象之间的独立性，避免副作用和数据共享问题
        bus_list_ = copy.deepcopy(bus_list)
        # r = np.random.randint(10, 40) / 10.     # 生成一个介于10（包括10）和40（不包括40）之间的随机整数/10.
        # 每个站点生成一个乘客到达率
        arr_rates = [1 / 60 / 2, 1 / 60 / 2, 1 / 60 / 1.2, 1 / 60, 1 / 60, 1 / 60 * 3, 1 / 60 * 4, 1 / 60 * 2, 1 / 60,
                     1 / 60 / 1.5, 1 / 60 / 1.8, 1 / 60 / 2]
        for _, stop in enumerate(stop_list_.values()):
            stop.set_rate(arr_rates[_])       # 为每个公交站都设置了一个不同的到达率（乘客）

        eng = Sim_Engine.Engine(bus_list=bus_list_, busstop_list=stop_list_, control_type=args.control,
                                dispatch_times=dispatch_times,
                                demand=0, simulation_step=simulation_step, route_list=route_list,
                                hold_once_arr=args.arr_hold, is_allow_overtake=args.overtake,
                                share_scale=args.share_scale, weight=args.weight)
        eng.agents = agents
        if ep > 0:    # ep=0时，也就是第一轮训练时，下面的两个if条件都不满足要求，都不执行
            if memory_copy != None:
                eng.GM = memory_copy
            for bid, b in eng.bus_list.items():
                if b.is_virtual == 0:
                    eng.GM.temp_memory[bid] = {'s': [], 'a': [],'a_log_prob':[], 'fp': [], 'r': []}

        if args.restore == 1 and args.control > 1:  # False   args.restore=0
            for k, v in agents.items():
                print(str(args.para_flag) + str('_') + str(args.share_scale) + str('_') + str(args.model) + str('_'))
                v.load(str(args.para_flag) + str('_') + str(args.share_scale) + str('_') + str(args.weight) + str(
                    '_') + str(args.model) + str('_'))
        Flag = True
        while Flag:
            Flag = eng.sim()
        ploss_log = []
        qloss_log = []
        if args.control == 2 and args.restore == 0:
        # if args.control > 1 and args.restore == 0:  # args.control =2 应该是表示强化学习
            if ep >= 0:     # 代表仿真的轮数
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

        log,_,_,_,_ = eng.cal_statistic(
            name=str(args.para_flag) + str('_') + str(args.share_scale) + str('_') + str(args.model) + str('_'),train=args.train)
        abspath = os.path.abspath(os.path.dirname(__file__))
        name = abspath + "/log/" + args.data + args.model + str('_') + str(args.control) + str('_')

        name += str(int(args.weight))
        if args.all == 1:
            name += 'all_'

        reward_avg,reward1_avg,reward2_avg,ploss_log_avg,qloss_log_avg,wait_cost_avg,travel_cost_avg = \
            U.train_result_track(eng=eng, ep=ep, qloss_log=qloss_log, ploss_log=ploss_log, log=log, name=name,seed=args.seed)
        # AWT.append(log['stw'])  # 列表包含46个值
        # AHD.append(log['sth'])
        # AOD.append(log['sto'])
        # ATT.append(log['att'])
        reward_avg_list.append(reward_avg)
        reward1_avg_list.append(reward1_avg)
        reward2_avg_list.append(reward2_avg)
        ploss_log_avg_list.append(ploss_log_avg)
        qloss_log_avg_list.append(qloss_log_avg)
        wait_cost_avg_list.append(wait_cost_avg)
        travel_cost_avg_list.append(travel_cost_avg)

        eng.close()
    # print(agents['0_1'].actor.state_dict())
    # print(agents['0_1'].critic.state_dict())
    # 绘制图像
    AWT = (log['stw'])  # 列表包含46个值
    AHD = (log['sth'])
    AOD = (log['sto'])
    ATT = (log['att'])
    # 轨迹图
    # sorted_list = sorted(eng.bus_list.items(), key=lambda x: x[1].dispatch_time)
    # sorted_buslist_dispatch = [item[1] for item in sorted_list]
    f = plt.figure()
    ax = plt.gca()
    f.set_size_inches((5, 5))  # 设置图像的尺寸
    plt.xlim([0, 28800])  # 设置x轴的范围
    plt.ylim(0, 6.1)
    # ticks = [n.loc for n in eng.busstop_list.values()]
    # labels = [str(_) for _ in range(len(eng.busstop_list))]
    # show_ticks = ticks[::3]  # 每隔2个刻度显示一个
    # show_labels = labels[::3]  # 每隔2个标签显示一个
    plt.yticks([0 + n * np.pi * 2 / 12 for n in range(12)],
               ['0','1', '2', '3', '4', '5', '6', '7',
                '8', '9', '10', '11'])  # 设置y轴的刻度和标签
    ax.tick_params(length=4, width=0.5)  # 设置刻度线的长度和宽度
    bus_trajectory = []
    bus_hold_action = []

    for b in (eng.bus_list.values()):
        x = np.array(b.time_step)
        y = np.array(b.loc)
        bus_trajectory.append(b.loc)
        masky = np.ma.array(y, mask=y >= 6.2)  # 将大于6.2的值遮盖起来

        bus_hold_action.append(b.hold_remain)

        plt.plot(x, masky, '-', label='bus  ' + str(b.id))
    plt.xlabel('Time step')
    plt.ylabel('Station')
    plt.show()
    # # AWT
    # f = plt.figure()
    # ax = plt.subplot(111)
    # ax.tick_params(length=4, width=0.5)
    # plt.xlabel('Stops')
    # plt.ylabel('AWT')
    # smoothing_window = 10
    # AWT_smoothed = pd.Series(AWT).rolling(smoothing_window, min_periods=smoothing_window).mean()
    # plt.plot(AWT, alpha=0.2)
    # plt.plot(AWT_smoothed, color='orange')
    # plt.grid()  # 用于画坐标网格线
    # plt.show()  # 用于显示图片
    # # AHD
    # f = plt.figure()
    # ax = plt.subplot(111)
    # ax.tick_params(length=4, width=0.5)
    # plt.xlabel('Stops')
    # plt.ylabel('AHD')
    # smoothing_window = 10
    # AHD_smoothed = pd.Series(AHD).rolling(smoothing_window, min_periods=smoothing_window).mean()
    # plt.plot(AHD, alpha=0.2)
    # plt.plot(AHD_smoothed, color='orange')
    # plt.grid()  # 用于画坐标网格线
    # plt.show()  # 用于显示图片
    # # AOD
    # f = plt.figure()
    # ax = plt.subplot(111)
    # ax.tick_params(length=4, width=0.5)
    # plt.xlabel('Stops')
    # plt.ylabel('AOD')
    # smoothing_window = 10
    # AOD_smoothed = pd.Series(AOD).rolling(smoothing_window, min_periods=smoothing_window).mean()
    # plt.plot(AOD, alpha=0.2)
    # plt.plot(AOD_smoothed, color='orange')
    # plt.grid()  # 用于画坐标网格线
    # plt.show()  # 用于显示图片
    # # ATT
    # f = plt.figure()
    # ax = plt.subplot(111)
    # ax.tick_params(length=4, width=0.5)
    # plt.xlabel('Stops')
    # plt.ylabel('ATT')
    # smoothing_window = 10
    # ATT_smoothed = pd.Series(ATT).rolling(smoothing_window, min_periods=smoothing_window).mean()
    # plt.plot(ATT, alpha=0.2)
    # plt.plot(ATT_smoothed, color='orange')
    # plt.grid()  # 用于画坐标网格线
    # plt.show()  # 用于显示图片
    # 奖励函数
    f = plt.figure()
    ax = plt.subplot(111)
    ax.tick_params(length=4, width=0.5)
    plt.xlabel('Training episode')
    plt.ylabel('Cumulative global reward')
    smoothing_window = 10
    rewards_smoothed = pd.Series(reward_avg_list).rolling(smoothing_window,
                                                          min_periods=smoothing_window).mean()  # 智能体获得的平均奖励
    rewards_smoothed1 = pd.Series(reward1_avg_list).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed2 = pd.Series(reward2_avg_list).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(reward_avg_list, alpha=0.2)
    plt.plot(rewards_smoothed, color='orange', label='total reward')
    plt.plot(reward1_avg_list, alpha=0.2)
    plt.plot(rewards_smoothed1, color='red', label='reward1')
    plt.plot(reward2_avg_list, alpha=0.2)
    plt.plot(rewards_smoothed2, color='green', label='reward2')
    plt.grid()
    ax.legend(loc='best', fancybox=True, shadow=False, ncol=1, prop={'size': 12})
    plt.show()

    # # 策略网络的损失函数
    # f = plt.figure()
    # ax = plt.subplot(111)
    # ax.tick_params(length=4, width=0.5)
    # plt.xlabel('Training episode')
    # plt.ylabel('Ploss')
    # smoothing_window = 10
    # p_loss_set_smoothed = pd.Series(ploss_log_avg_list).rolling(smoothing_window,
    #                                                             min_periods=smoothing_window).mean()
    # plt.plot(ploss_log_avg_list, alpha=0.2)
    # plt.plot(p_loss_set_smoothed, color='orange')
    # plt.grid()  # 用于画坐标网格线
    # plt.show()  # 用于显示图片

    # 价值网络的损失函数
    f = plt.figure()
    ax = plt.subplot(111)
    ax.tick_params(length=4, width=0.5)
    plt.xlabel('Training episode')
    plt.ylabel('Qloss')
    smoothing_window = 10
    q_loss_set_smoothed = pd.Series(qloss_log_avg_list).rolling(smoothing_window,
                                                                min_periods=smoothing_window).mean()
    plt.plot(qloss_log_avg_list, alpha=0.2)
    plt.plot(q_loss_set_smoothed, color='orange')
    plt.grid()  # 用于画坐标网格线
    plt.show()  # 用于显示图片

    # f = plt.figure()
    # ax = plt.subplot(111)
    # ax.tick_params(length=4, width=0.5)
    # plt.xlabel('Training episode')
    # plt.ylabel('Wait cost')
    # smoothing_window = 10
    # wait_cost_set_smoothed = pd.Series(wait_cost_avg_list).rolling(smoothing_window,
    #                                                                min_periods=smoothing_window).mean()
    # plt.plot(wait_cost_avg_list, alpha=0.2)
    # plt.plot(wait_cost_set_smoothed, color='orange')
    # plt.grid()  # 用于画坐标网格线
    # plt.show()  # 用于显示图片
    #
    # f = plt.figure()
    # ax = plt.subplot(111)
    # ax.tick_params(length=4, width=0.5)
    # plt.xlabel('Training episode')
    # plt.ylabel('Travel cost')
    # smoothing_window = 10
    # travel_cost_set_smoothed = pd.Series(travel_cost_avg_list).rolling(smoothing_window,
    #                                                                    min_periods=smoothing_window).mean()
    # plt.plot(travel_cost_avg_list, alpha=0.2)
    # plt.plot(travel_cost_set_smoothed, color='orange')
    # plt.grid()  # 用于画坐标网格线
    # plt.show()  # 用于显示图片


def evaluate(args):
    stop_list, pax_num = U.getStopList()  # stop_list=46,字典   pax_num=0
    print('Stops prepared, total bus stops: %g' % (len(stop_list)))
    bus_routes = U.getBusRoute(stop_list, args)  # 0_1:里面包含59辆公交车
    print('Bus routes prepared, total routes :%g' % (len(bus_routes)))
    stop_list_ = copy.deepcopy(stop_list)
    dispatch_times, bus_list, route_list, simulation_step = U.init_bus_list(bus_routes)
    print('init...')
    agents = {}
    if args.model != '':
        eng = Sim_Engine.Engine(bus_list=bus_list, busstop_list=stop_list_, control_type=args.control,
                                dispatch_times=dispatch_times,
                                demand=0, simulation_step=simulation_step, route_list=route_list,
                                hold_once_arr=args.arr_hold, is_allow_overtake=args.overtake,
                                share_scale=args.share_scale, weight=args.weight)
        bus_list = eng.bus_list
        bus_stop_list = eng.busstop_list
        state_dim = 3
        # non share
        if args.share_scale == 0:
            for k, v in eng.bus_list.items():
                agent = Agent(state_dim=state_dim, name='', n_stops=len(bus_stop_list), buslist=bus_list,
                              seed=args.seed)
                agents[k] = agent

        # share in route
        if args.share_scale == 1:  # true
            agents = {}
            for k, v in eng.route_list.items():  # 只有一条线路  线路为一个智能体
                agent = Agent(state_dim=state_dim, name='', n_stops=len(bus_stop_list), buslist=bus_list,
                              seed=args.seed)  # 每个agent都有一个crtic一个target_crtic,以及一个actor一个target_actor
                agents[k] = agent
                agents[k].actor.load_state_dict(torch.load(r"D:\Edge down\pycharm\Py_Projects\2024_12_23\circle_mappo\model\save\_A_0_1_1_2_mappo_1_actor.pth"))
                agents[k].critic.load_state_dict(torch.load(r"D:\Edge down\pycharm\Py_Projects\2024_12_23\circle_mappo\model\save\_A_0_1_1_2_mappo_1_critic.pth"))
    # print(agents['0_1'].actor.state_dict())
    # print(agents['0_1'].critic.state_dict())
    for ep in range(20):    # args.episode=401
        stop_list_ = copy.deepcopy(stop_list)   # 对stop_list的深拷贝，保持了对象之间的独立性，避免副作用和数据共享问题
        bus_list_ = copy.deepcopy(bus_list)
        # r = np.random.randint(10, 40) / 10.     # 生成一个介于10（包括10）和40（不包括40）之间的随机整数/10.
        # 每个站点生成一个乘客到达率
        arr_rates = [1 / 60 / 2, 1 / 60 / 2, 1 / 60 / 1.2, 1 / 60, 1 / 60, 1 / 60 * 3, 1 / 60 * 4, 1 / 60 * 2, 1 / 60,
                     1 / 60 / 1.5, 1 / 60 / 1.8, 1 / 60 / 2]
        for _, stop in enumerate(stop_list_.values()):
            stop.set_rate(arr_rates[_])       # 为每个公交站都设置了一个不同的到达率（乘客）

        eng = Sim_Engine.Engine(bus_list=bus_list_, busstop_list=stop_list_, control_type=args.control,
                                dispatch_times=dispatch_times,
                                demand=0, simulation_step=simulation_step, route_list=route_list,
                                hold_once_arr=args.arr_hold, is_allow_overtake=args.overtake,
                                share_scale=args.share_scale, weight=args.weight)
        eng.agents = agents
        Flag = True
        while Flag:
            Flag = eng.sim()
        log, _, _, _, _ = eng.cal_statistic(
            name=str(args.para_flag) + str('_') + str(args.share_scale) + str('_') + str(args.model) + str('_'),
            train=args.train)
        abspath = os.path.abspath(os.path.dirname(__file__))
        name = abspath + "/log/" + args.data + args.model + str('_') + str(args.control) + str('_')

        name += str(int(args.weight))
        if args.all == 1:
            name += 'all_'+ str(args.train)

        reward_avg, reward1_avg, reward2_avg, ploss_log_avg, qloss_log_avg, wait_cost_avg, travel_cost_avg = \
            U.train_result_track(eng=eng, ep=ep, qloss_log=0, ploss_log=0, log=log, name=name,
                                 seed=args.seed)
        eng.close()



if __name__ == '__main__':

    seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.train == 1:    # 默认设置为1
        train(args)

    else:
        evaluate(args)
