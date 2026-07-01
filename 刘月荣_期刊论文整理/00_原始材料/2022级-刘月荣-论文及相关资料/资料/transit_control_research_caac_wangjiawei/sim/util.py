import pandas as pd
import numpy as np
import os
from sim.Bus import Bus
from sim.Route import Route
from sim.Busstop import Bus_stop
from sim.Passenger import Passenger
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None


def getBusRoute(data):
    my_path = os.path.abspath(os.path.dirname(__file__))
    path = my_path + "/data/" + data + "/"
    _path_trips = path + 'trips.txt'
    _path_st = path + 'stop_times.txt'
    trips = pd.DataFrame(pd.read_csv(_path_trips))
    stop_times = pd.DataFrame(pd.read_csv(_path_st))

    stop_times.dropna(subset=['arrival_time'], inplace=True)
    bus_routes = {}

    trip_ids = set(stop_times['trip_id'])

    try:
        service_id = trips.iloc[np.random.randint(0, trips.shape[0])]['service_id']
        trips = trips[trips['service_id'] == service_id]
    except:
        pass

    # each route_id may correspond to multiple trip_id
    for trip_id in trip_ids:    # trip_ids总共有59个
        # A completely same route indicates the same shape_id in trip file, but this field is not 100% provided by opendata
        try:
            if 'shape_id' in trips.columns:
                route_id = str(trips[trips['trip_id'] == trip_id].iloc[0]['shape_id'])
                block_id = ''
                dir = ''
            else:
                route_id = str(trips[trips['trip_id'] == trip_id].iloc[0]['route_id'])
                block_id = str(trips[trips['trip_id'] == trip_id].iloc[0]['block_id'])
                dir = str(trips[trips['trip_id'] == trip_id].iloc[0]['trip_headsign'])
        except:
            continue
        # Identifies a set of dates when service is available for one or more routes.
        trip = stop_times[stop_times['trip_id'] == trip_id]
        try:
            trip['arrival_time'] = pd.to_datetime(trip['arrival_time'], format='%H:%M:%S')
        except:
            trip['arrival_time'] = pd.to_datetime(trip['arrival_time'], format="%Y-%m-%d %H:%M:%S")
        trip = trip.sort_values(by='arrival_time')

        trip_dist = trip.iloc[:]['shape_dist_traveled'].to_list()     # 每个公交站的位置
        if len(trip_dist) <= 0 or np.isnan(trip_dist[0]):
            continue

        schedule = ((trip.iloc[:]['arrival_time'].dt.hour * 60 + trip.iloc[:]['arrival_time'].dt.minute) * 60 +
                    trip.iloc[:]['arrival_time'].dt.second).to_list()
        if len(schedule) <= 2 or np.isnan(schedule[0]):
            continue
        b = Bus(id=trip_id, route_id=route_id, stop_list=trip.iloc[:]['stop_id'].to_list(),
                dispatch_time=schedule[0], block_id=block_id, dir=dir)
        b.left_stop = []

        b.speed = (trip_dist[1] - trip_dist[0]) / (schedule[1] - schedule[0])   # 站间距除以公交到达两个站点的时间差
        b.c_speed = b.speed
        for i in range(len(trip_dist)):
            if str(b.stop_list[i]) in b.stop_dist:
                b.left_stop.append(str(b.stop_list[i]) + '_' + str(i))
                b.stop_dist[str(b.stop_list[i]) + '_' + str(i)] = trip_dist[i]
                b.schedule[str(b.stop_list[i]) + '_' + str(i)] = schedule[i]
            else:
                b.left_stop.append(str(b.stop_list[i]))
                b.stop_dist[str(b.stop_list[i])] = trip_dist[i]    # 为公交车设置到达每个公交站的距离
                b.schedule[str(b.stop_list[i])] = schedule[i]      # 为公交车设置到达每个站的时间

        b.stop_list = b.left_stop[:]
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


def getStopList(data, read=0):
    my_path = os.path.abspath(os.path.dirname(__file__))
    path = my_path + "/data/" + data + "/"
    _path_stops = path + 'stops.txt'
    _path_st = path + 'stop_times.txt'
    _path_trips = path + 'trips.txt'
    stops = pd.DataFrame(pd.read_csv(_path_stops))    # 包含id,经度、纬度
    stop_times = pd.DataFrame(pd.read_csv(_path_st))
    trips = pd.DataFrame(pd.read_csv(_path_trips))
    stop_list = {}

    select_stops = pd.merge(stops, stop_times, on=['stop_id'], how='left')   # 合并数据集，以stops中'stop_id'为依据进行合并，保留stop_times中包含有stops中'stop_id'的所有行
    select_stops = select_stops.sort_values(by='shape_dist_traveled', ascending=False)    # 按照列名降序排序
    select_stops = select_stops.drop_duplicates(subset='stop_id', keep="first").sort_values(by='shape_dist_traveled',ascending=True)   # 首先去除'stop_id'列重复的行，只保留第一次出现的行，接着按升序排列
    for i in range(select_stops.shape[0]):
        stop = Bus_stop(id=str(select_stops.iloc[i]['stop_id']), lat=select_stops.iloc[i]['stop_lat'],lon=select_stops.iloc[i]['stop_lon'])  # 遍历每一行对应的列，实例化公交站
        stop.loc = select_stops.iloc[i]['shape_dist_traveled']   # 设置了公交站的位置
        try:
            stop.next_stop = str(select_stops.iloc[i + 1]['stop_id'])
        except:
            stop.next_stop = None
        stop_list[str(select_stops.iloc[i]['stop_id'])] = stop   # 将每个公交站存储在字典中

    _path_demand = path + 'demand.csv'

    pax_num = 0

    try:
        demand = pd.DataFrame(pd.read_csv(_path_demand))
    except:
        print('No available demand file')
        return stop_list, 0
    try:
        demand['Ride_Start_Time'] = pd.to_datetime(demand['Ride_Start_Time'], format='%H:%M:%S')
    except:
        demand['Ride_Start_Time'] = pd.to_datetime(demand['Ride_Start_Time'], format="%Y-%m-%d %H:%M:%S")

    demand['Ride_Start_Time_sec'] = (demand.iloc[:]['Ride_Start_Time'].dt.hour * 60 + demand.iloc[:][
        'Ride_Start_Time'].dt.minute) * 60 + demand.iloc[:]['Ride_Start_Time'].dt.second   # 增加新的一列，将原来的时间转换成秒

    demand.dropna(subset=['ALIGHTING_STOP_STN'], inplace=True)    # 删除ALIGHTING_STOP_STN'列中包含缺失值的行，直接在原始对象上进行修改
    demand = demand[demand.ALIGHTING_STOP_STN != demand.BOARDING_STOP_STN]  # 筛选这两列不相等的行
    demand = demand.sort_values(by='Ride_Start_Time_sec')

    for stop_id, stop in stop_list.items():
        demand_by_stop = demand[demand['BOARDING_STOP_STN'] == int(stop_id)]

        # macro demand setting
        if read == 0:
            t = 0
            while t < 24:    # 这个24是什么，应该代表的是24小时
                d = demand_by_stop[(demand_by_stop['Ride_Start_Time_sec'] >= t * 3600) & (demand_by_stop['Ride_Start_Time_sec'] < (t + 1) * 3600)]   # 表示第一小时内、第二小时内...第24小时内到达某个站点的乘客

                stop.dyna_arr_rate.append(d.shape[0] / 3600.)    # 设置车站的乘客到达率，每个公交站点都有24个到达率（每小时一个）——————————————————————————————————————————————————
                for dest_id in stop_list.keys():                 # 乘客站点的数量共有46个

                    od = d[demand['ALIGHTING_STOP_STN'] == int(dest_id)]    # 将在stop站上i车的乘客按其不同的下车站点进行分类
                    if od.empty:
                        continue
                    if dest_id not in stop.dest:
                        stop.dest[dest_id] = [0 for _ in range(24)]    # 从stop站上车，在dest_id下车的乘客概率（每小时）
                    stop.dest[dest_id][t] = od.shape[0] / 3600.     # 应该是设置乘客的下车率——————————————————————————————————————————————————————————————————————————————
                t += 1
        # 下面这段代码没有被运行，所以pax_num=0
        else:
            # micro demand setting
            for i in range(demand_by_stop.shape[0]):
                pax = Passenger(id=demand_by_stop.iloc[i]['TripID'], origin=stop_id,
                                plan_board_time=float(demand_by_stop.iloc[i]['Ride_Start_Time_sec']))
                pax.dest = str(int(demand_by_stop.iloc[i]['ALIGHTING_STOP_STN']))
                pax.realcost = float(demand_by_stop.iloc[i]['Ride_Time']) * 60.   
                pax.route = str(demand_by_stop.iloc[i]['Srvc_Number']) + '_' + str(int(demand_by_stop.iloc[i]['Direction']))
                stop.pax[pax.id] = pax
                pax_num += 1

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
    reward_bus_wise = []
    reward_bus_wisep1 = []
    reward_bus_wisep2 = []
    rs = []

    wait_cost = log['wait_cost']
    travel_cost = log['travel_cost']
    delay = log['delay']
    hold_cost = log['hold_cost']
    headways_var = log['headways_var']
    headways_mean = log['headways_mean']
    AOD = log["AOD"]
    for bid, r in eng.reward_signal.items():
        if len(r) > 0:  # .bus_list[bid].forward_bus!=None and  engine.bus_list[bid].backward_bus!=None :
            reward_bus_wise.append(np.mean(r))
            rs += r
            reward_bus_wisep1.append(np.mean(eng.reward_signalp1[bid]))
            reward_bus_wisep2.append(np.mean(eng.reward_signalp2[bid]))

    if ep % 1 == 0:
        train_log = pd.DataFrame()
        train_log['bunching'] = [log['bunching']]
        train_log['ploss'] = [np.mean(ploss_log)]
        train_log['qloss'] = [np.mean(qloss_log)]
        train_log['reward'] = [np.mean(reward_bus_wise)]
        train_log['reward1'] = [np.mean(reward_bus_wisep1)]
        train_log['reward2'] = [np.mean(reward_bus_wisep2)]
        train_log['avg_hold'] = np.mean([np.array([arr]) if not isinstance(arr, np.ndarray) else arr for arr in hold_cost])
        train_log['action'] = np.mean(np.array(eng.action_record))
        train_log['wait'] = [np.mean(wait_cost)]
        train_log['travel'] = [np.mean(travel_cost)]
        train_log['delay'] = [np.mean(delay)]
        train_log['AOD'] = AOD

        for k, v in headways_mean.items():
            train_log['headway_mean' + str(k)] = [np.mean(v)]
        for k, v in headways_var.items():
            train_log['headway_var' + str(k)] = [np.mean(v)]

        res = pd.DataFrame()
        res['stw'] = log['stw']
        res['sto'] = log['sto']
        res['sth'] = log['sth']
        print(
            'Episode: %g | reward: %g | reward_var: %g | reward1: %g | reward2: %g | ploss: %g | qloss: %g |\n  wait '
            'cost: %g | travel cost: %g | max hold :%g| min hold :%g| avg hold :%g | var hold :%g' % (
                ep - 1, np.mean(reward_bus_wise), np.var(rs), np.mean(reward_bus_wisep1), np.mean(reward_bus_wisep2),
                np.mean(ploss_log), np.mean(qloss_log), np.mean(wait_cost), np.mean(travel_cost), np.max([np.array([arr]) if not isinstance(arr, np.ndarray) else arr for arr in hold_cost]),
                np.min([np.array([arr]) if not isinstance(arr, np.ndarray) else arr for arr in hold_cost]),
                np.mean([np.array([arr]) if not isinstance(arr, np.ndarray) else arr for arr in hold_cost]), np.var([np.array([arr]) if not isinstance(arr, np.ndarray) else arr for arr in hold_cost])))
        arr_log = pd.DataFrame(log['arr_times'])
        try:
            if ep > 1:
                train_log.to_csv(name + str(seed) + '.csv', mode='a', header=False)
                res.to_csv(name + str(seed) + 'res.csv', mode='a', header=False)
            else:
                res.to_csv(name + str(seed) + 'res.csv')
                train_log.to_csv(name + str(seed) + '.csv')
                arr_log.to_csv(name + str(seed) + 'arr.csv')
        except Exception as e:
            print(e)


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
            route_list[k].schedule.append(bus.schedule)   #  为线路中的时刻表添加每辆车在各个站点的发车时间
            s = sorted(list(bus.schedule.values()))      # 将公交车在个站点的发车时间按升序排列
            dispatch_time.append(s[0])    #  dispatch_time这个列表记录的是每个公交车的初始发车时间
            dispatch_time = sorted(dispatch_time)   # 将每个公交车的初始发车时间按升序排列
            bus_dispatch[s[0]] = bus.id  # {发车时间：公交车id}    # 这个自定记录的是每个公交车的初始发车时刻和对应的公交车
            if bus.route_id in dispatch_times:
                dispatch_times[bus.route_id].append(bus.dispatch_time)   # dispatch_times 记录的是线路上所有公交车的初始发车时间
            else:
                dispatch_times[bus.route_id] = [bus.dispatch_time]
            dispatch_times[bus.route_id] = sorted(dispatch_times[bus.route_id])  # 将线路上所有公交车的初始发车时间按升序排列

    for bus_id, bus in bus_list.items():   # 设置最小的车头时距，最小的车头时距等于的是左右车辆中初始发车时间相差最小的时间间距
        min_headway = 9999999999999
        busid = -1
        for bus_id_, bus_ in bus_list.items():
            if bus_id_ != bus.id and bus.route_id == bus_.route_id and 0 < (
                    bus_.dispatch_time - bus.dispatch_time) < min_headway:   #满足条件说明 bus_id_的发车时间晚于bus_id
                min_headway = abs(bus_.dispatch_time - bus.dispatch_time)
                busid = bus_id_
        if busid != -1:
            bus_list[bus.id].backward_bus = busid   # bus_id的后向车辆为bus_id_
            bus_list[busid].forward_bus = bus.id   # bus_id_的前向车辆为bus_id

    return dispatch_times, bus_list, route_list, simulation_step
