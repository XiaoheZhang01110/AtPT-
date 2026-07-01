import pandas as pd
import numpy as np
import os
from sim.Bus import Bus
from sim.Route import Route
from sim.Busstop import Bus_stop
from sim.Passenger import Passenger
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None


def getBusRoute(stop_list,args):

    bus_routes = {}
    route_id = '0_1'
    block_id = ''
    dir = ''

    for i in range(6):
        b = Bus(id=i, route_id=route_id, stop_list=list(stop_list.keys()),
                dispatch_time =360* i, speed_type=args.speed_type,block_id=block_id, dir=dir)
        b.left_stop = []

        b.speed = np.pi * 2 / 12 / 180  # 站间距除以公交到达两个站点的时间差
        b.c_speed = b.speed
        # stop_list_id = list(stop_list.keys())[1:] + [list(stop_list.keys())[0]]
        stop_list_id = list(stop_list.keys())
        for i in stop_list_id:
            b.left_stop.append(i)
            b.stop_dist[i] = stop_list[i].loc
        # 为公交车设置到达每个公交站的距离
            # b.schedule[str(b.stop_list[i])] = schedule[i]      # 为公交车设置到达每个站的时间

        # b.stop_list = b.left_stop[:]
        b.set()
        if route_id in bus_routes:
            bus_routes[route_id].append(b)
        else:
            bus_routes[route_id] = [b]

    # Do not consider the route with only 1 trip
    bus_routes_ = {}
    for k, v in bus_routes.items():
        if len(v) > 1:
            bus_routes_[k] = v
    return bus_routes_


def getStopList(read=0):

    stop_list = {}

     # 首先去除'stop_id'列重复的行，只保留第一次出现的行，接着按升序排列
    for i in range(12):
        stop = Bus_stop(id=str(i), lat=0.0,lon=0.0)  # 遍历每一行对应的列，实例化公交站
        # if i == 0:
        #     stop.loc = np.pi * 2
        # else:
        #     stop.loc = np.pi * 2 / 12 * (i)   # 设置了公交站的位置
        stop.loc = np.pi * 2 / 12 * (i)
        if i != 11:
            stop.next_stop = str(i+1)
        else:
            stop.next_stop = str(0)
        stop_list[str(i)] = stop   # 将每个公交站存储在字典中

    pax_num = 0

    return stop_list, pax_num


def demand_analysis(engine=None):
    if engine is not None:
        stop_list = list(engine.busstop_list.keys())
        stop_hash = {}
        i = 0
        for p in stop_list:
            stop_hash[p] = i
            i += 1

    # output data for stack area graph
    demand = []
    for t in range(24):
        d = np.zeros(len(stop_list))
        for s in stop_list:
            for pid, pax in engine.busstop_list[s].pax.items():
                if int((pax.plan_board_time - 0) / 3600) == t:
                    d[stop_hash[s]] += 1
        demand.append(d)
    df = pd.DataFrame(demand, columns=[str(i) for i in range(len(stop_list))])
    df.to_csv('demand.csv')

    return


def sim_validate(engine, data):
    actual_onboard = []
    sim_onboard = []
    sim_travel_cost = []
    actual_travel_cost = []
    for pid, pax in engine.pax_list.items():
        actual_onboard.append(pax.plan_board_time)
        sim_onboard.append(pax.onboard_time)

        sim_travel_cost.append(abs(pax.onboard_time - pax.alight_time))
        actual_travel_cost.append(pax.realcost)

    actual_onboard = np.array(actual_onboard)
    sim_onboard = np.array(sim_onboard)
    actual_travel_cost = np.array(actual_travel_cost)

    sim_travel_cost = np.array(sim_travel_cost)
    print('Boarding RMSE:%g' % (np.sqrt(np.mean((actual_onboard - sim_onboard) ** 2))))
    print('Travel RMSE:%g' % (np.sqrt(np.mean((actual_travel_cost - sim_travel_cost) ** 2))))

    sim_comp = pd.DataFrame()
    sim_comp['actual_onboard'] = actual_onboard
    sim_comp['sim_onboard'] = sim_onboard
    sim_comp['sim_travel_cost'] = sim_travel_cost
    sim_comp['actual_travel_cost'] = actual_travel_cost
    sim_comp.to_csv('G:\\mcgill\\MAS\\gtfs_testbed\\result\\sim_comp' + str(data) + '.csv')
    print('ok')


def visualize_pax(engine):
    for pax_id, pax in engine.pax_list.items():
        if pax.onboard_time < 999999999:
            plt.plot([int(pax_id), int(pax_id)], [pax.arr_time, pax.onboard_time])

    plt.show()


def train_result_track(eng, ep, qloss_log, ploss_log, log, name='', seed=0):
    reward_bus_wise = []   # 每个车辆一个episode获得的平均奖励
    reward_bus_wisep1 = []  # 每个车辆一个episode获得的与等待时间相关的平均奖励
    reward_bus_wisep2 = []  # # 每个车辆一个episode获得的与在车时间相关的平均奖励
    rs = []   # 所有车辆的所有奖励

    wait_cost = log['wait_cost']         # 每个乘客的等待时间
    travel_cost = log['travel_cost']     # 每个乘客的在车时间
    delay = log['delay']                 # 每个乘客的延误时间（等待时间+驻站时间）
    hold_cost = log['hold_cost']         # 所有公交在所有站点的驻站时间
    headways_var = log['headways_var']   # 每个站点所有车辆的后向车头时距的标准差
    headways_mean = log['headways_mean'] # 每个站点所有车辆的后向车头时距的方差
    AOD = log["AOD"]                     # 所有站点（所有公交到站时是的在车乘客数量的方差/均值）的均值
    for bid, r in eng.reward_signal.items():    # eng.reward_signal存储的是每辆车获得的所有总奖励
        if len(r) > 0:  # .bus_list[bid].forward_bus!=None and  engine.bus_list[bid].backward_bus!=None :
            reward_bus_wise.append(np.mean(r))
            rs += r
            reward_bus_wisep1.append(np.mean(eng.reward_signalp1[bid]))
            reward_bus_wisep2.append(np.mean(eng.reward_signalp2[bid]))

    if ep % 1 == 0:   # 每一个episode都执行次操作
        train_log = pd.DataFrame()
        train_log['bunching'] = [log['bunching']]
        train_log['ploss'] = [np.mean(ploss_log)]
        train_log['qloss'] = [np.mean(qloss_log)]
        train_log['reward'] = [np.mean(reward_bus_wise)]
        train_log['reward1'] = [np.mean(reward_bus_wisep1)]
        train_log['reward2'] = [np.mean(reward_bus_wisep2)]
        train_log['avg_hold'] = np.mean([np.mean(arr) for arr in hold_cost])
        train_log['action'] = np.mean(np.array(eng.action_record))
        train_log['wait'] = [np.mean(wait_cost)]
        train_log['travel'] = [np.mean(travel_cost)]
        train_log['delay'] = [np.mean(delay)]
        train_log['AOD'] = AOD
        # train_log['AOD'] = AOD


        for k, v in headways_mean.items():
            train_log['headway_mean' + str(k)] = [np.mean(v)]
        for k, v in headways_var.items():
            train_log['headway_var' + str(k)] = [np.mean(v)]

        res = pd.DataFrame()
        res['stw'] = log['stw']
        res['sto'] = log['sto']
        res['sth'] = log['aht']  # 每个站点平均驻站时间
        res['att'] = log['att']
        res['Bus0_AOD'] = log['Bus0_AOD']
        res['Bus1_AOD'] = log['Bus1_AOD']
        res['Bus2_AOD'] = log['Bus2_AOD']
        res['Bus3_AOD'] = log['Bus3_AOD']
        res['Bus4_AOD'] = log['Bus4_AOD']
        res['Bus5_AOD'] = log['Bus5_AOD']
        print(
            'Episode: %g | reward: %g | reward_var: %g | reward1: %g | reward2: %g | ploss: %g | qloss: %g |\n  wait '
            'cost: %g | travel cost: %g | max hold :%g| min hold :%g| avg hold :%g | var hold :%g' % (
                ep - 1, np.mean(reward_bus_wise), np.var(rs), np.mean(reward_bus_wisep1), np.mean(reward_bus_wisep2),
                np.mean(ploss_log), np.mean(qloss_log), np.mean(wait_cost), np.mean(travel_cost),
                np.max([np.max(arr) for arr in hold_cost]), np.min([np.min(arr) for arr in hold_cost]),
                np.mean([np.mean(arr) for arr in hold_cost]), np.var([item for arr in hold_cost for item in arr])))
        arr_log = pd.DataFrame(log['arr_times'])
        try:
            if ep > 1:
                train_log.to_csv(name + str(seed) + '.csv', mode='a', header=False)   # 对之前提到的 train_log 对象进行追加写入操作，mode='a'表示以追加的方式写入文件，header=False的一个关键字参数，表示不写入列名称（header）到文件中
                res.to_csv(name + str(seed) + 'res.csv', mode='a', header=False)
            else: # 每次执行这些文件都会被覆盖
                res.to_csv(name + str(seed) + 'res.csv')    # 将数据对象保存为csv文件
                train_log.to_csv(name + str(seed) + '.csv')
                arr_log.to_csv(name + str(seed) + 'arr.csv')
        except Exception as e:
            print(e)
    return np.mean(reward_bus_wise),np.mean(reward_bus_wisep1),np.mean(reward_bus_wisep2),np.mean(ploss_log),np.mean(qloss_log),np.mean(wait_cost),np.mean(travel_cost)


def visualize_trajectory(engine, name=''):
    for r_id, r in engine.route_list.items():
        trajectory = pd.DataFrame()
        for b_id in engine.bus_list:
            if engine.bus_list[b_id].route_id != r_id:
                continue
            df = pd.DataFrame()
            b = engine.bus_list[b_id]
            y = np.array(b.loc)
            df[str(b_id) + '_time'] = b.time_step
            df[str(b_id) + '_loc'] = y.tolist()
            trajectory = pd.concat([trajectory, df], ignore_index=True, axis=1)
    for r_id, r in engine.route_list.items():
        for b_id in engine.bus_list:
            if engine.bus_list[b_id].route_id != r_id:
                continue
            df = pd.DataFrame()
            b = engine.bus_list[b_id]
            occp = np.array(b.occp)
            df['time'] = b.time_step
            df['loc'] = b.loc
            df['op'] = occp
            df['stop'] = b.stops_record
            if b.is_virtual == 1:
                df.to_csv(name + str(b_id) + '#.csv')
            else:
                df.to_csv(name + str(b_id) + '.csv')

        break


def init_bus_list(bus_routes):
    stop_record = []
    route_list = {}
    dispatch_times = {}
    bus_list = {}
    for k, v in bus_routes.items():
        route_list[k] = Route(id=k, stop_list=v[0].stop_list, dist_list=v[0].stop_dist)
        stop_record.append(v[0].stop_list)
        min_dispatch_time = 1000000
        simulation_step = 9999999999
        dispatch_time = []
        bus_dispatch = {}
        for bus in v:
            bus.set()
            bus_list[bus.id] = bus    # 为公交列表中添加公交
            bus.last_vist_interval = bus.dispatch_time   # 初始值为-1，设置为公交的发车时间
            if min_dispatch_time > bus.dispatch_time:
                min_dispatch_time = bus.dispatch_time    # 设置公交的发车时间
            if simulation_step > bus.dispatch_time:
                simulation_step = bus.dispatch_time   # 仿真时间设置为所有公交车中发车时间最小的时间，也就是第一辆发车的公交车的发车时间
            route_list[k].bus_list.append(bus.id)    # 为线路中的公交列表添加公交车id
            # route_list[k].schedule.append(bus.schedule)   #  为线路中的时刻表添加每辆车在各个站点的发车时间
            # s = sorted(list(bus.schedule.values()))      # 将公交车在个站点的发车时间按升序排列
            dispatch_time.append(bus.dispatch_time)    #  dispatch_time这个列表记录的是每个公交车的初始发车时间
            dispatch_time = sorted(dispatch_time)   # 将每个公交车的初始发车时间按升序排列
            bus_dispatch[bus.dispatch_time] = bus.id  # {发车时间：公交车id}    # 这个自定记录的是每个公交车的初始发车时刻和对应的公交车
            if bus.route_id in dispatch_times:
                dispatch_times[bus.route_id].append(bus.dispatch_time)   # dispatch_times 记录的是线路上所有公交车的初始发车时间
            else:
                dispatch_times[bus.route_id] = [bus.dispatch_time]
            dispatch_times[bus.route_id] = sorted(dispatch_times[bus.route_id])  # 将线路上所有公交车的初始发车时间按升序排列

    for bus_id, bus in bus_list.items():   # 设置最小的车头时距，最小的车头时距等于的是左右车辆中初始发车时间相差最小的时间间距


        if bus_id == 0 :
            bus.backward_bus = bus_id + 1
            bus.forward_bus = 5
        elif bus_id == 5:
            bus.backward_bus = 0
            bus.forward_bus = bus_id - 1
        else:
            bus.backward_bus = bus_id + 1
            bus.forward_bus = bus_id - 1


    return dispatch_times, bus_list, route_list, simulation_step
