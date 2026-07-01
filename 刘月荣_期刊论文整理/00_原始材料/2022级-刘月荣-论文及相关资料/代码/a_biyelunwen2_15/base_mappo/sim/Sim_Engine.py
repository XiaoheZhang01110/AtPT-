import numpy as np
from sim.Passenger import Passenger
from sim.Bus import Bus
from sim.Route import Route
import matplotlib.pyplot as plt
from model.Group_MemoryC import Memory
import pandas as pd
import time
import math
import torch
import warnings
warnings.filterwarnings("ignore")
class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)
class RewardScaling:
    def __init__(self,shape,gamma):
        self.shape = shape
        self.gamma = gamma
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)
    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)
        return x
    def reset(self):
        self.R = np.zeros(self.shape)
class Engine():
    def __init__(self, bus_list,busstop_list,route_list,simulation_step,dispatch_times, demand=0,agents=None,share_scale=0, is_allow_overtake=0,hold_once_arr=1,control_type=1,seed=1,all=0,weight=0,ppo_epoch=2,num_mini_batch=3):

        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.all=all
        self.busstop_list = busstop_list
        self.simulation_step = simulation_step
        self.pax_list = {}  # passenger on road
        self.arr_pax_list = {}  # passenger who has finished trip
        self.dispatch_buslist = {}
        self.agents = {}
        self.route_list = route_list
        self.dispatch_buslist = {}
        self.is_allow_overtake = is_allow_overtake   # 0——不允许超车？
        self.hold_once_arr = hold_once_arr    # 1
        self.control_type = control_type      # 2
        self.agents = agents
        self.bus_list = bus_list
        self.bunching_times = 0
        self.arrstops = 0
        self.reward_signal = {}
        self.reward_signalp1={}
        self.reward_signalp2={}
        self.qloss = {}
        self.weight = weight/10.   # 0.2
        self.demand = demand
        self.records = []
        self.share_scale =share_scale     # 1
        self.step = 0
        self.dispatch_times = dispatch_times    # 包含每辆车的发车时间
        self.cvlog=[]
        self.alpha = 0.5
        self.max_hold_time = 180


        members = list(self.bus_list.keys())    # 公交车的编号，共59个
        self.GM = Memory(members)
        self.rs = {}
        for b_id,b in self.bus_list.items():
            self.reward_signal[b_id]=[]
            self.reward_signalp1[b_id] = []
            self.reward_signalp2[b_id] = []

        self.arrivals = {}

        # stop hash
        self.stop_hash = {}
        k = 0
        for bus_stop_id, bus_stop in self.busstop_list.items():
            self.stop_hash[bus_stop_id]=k
            k+=1

        self.bus_hash={}
        k = 0
        for bus_id, bus in self.bus_list.items():
            self.bus_hash[bus_id]=k
            k+=1


        self.action_record = []
        self.reward_record = []
        self.state_record = []

    def cal_statistic(self,name,train=1):
        print('total pax:%d'%(len(self.pax_list)))   # 一个episode中整个模型生成的乘客数量
        wait_cost = []      # 用来存储每个乘客的等待时间
        travel_cost = []    # 完成行程的乘客的在车时间
        headways_var = {}   # 每个站点（键） 所有车辆到站时间间隔的标准差（值）
        headways_mean = {}  # # 每个站点（键） 所有车辆到站时间间隔的均值（值）
        boards = []         # 记录所有登上车辆的时间
        arrs = []
        origins = []
        dests = []
        still_wait = 0     # 在站点未上车的乘客数量
        stop_wise_wait = {}    # 记录每个站点处（键）所有等车乘客的等待时间（值）
        stop_wise_hold = {}    # 记录每个站点处（键）所有车辆的驻站时间（值）
        delay = []           # 完成行程的乘客延误时间：乘客的下车时间-乘客的到达时间，未乘客总的行程时间，总的行程时间-乘客在路上时间，即为乘客等待车辆到来和驻站控制产生的延误
        stop_addition_travel = {}
        for pax_id, pax in self.pax_list.items():
            w = min(pax.onboard_time - pax.arr_time, self.simulation_step-pax.arr_time)   # 乘客的等待上车的时间  但是为什么要用当前的时间步-乘客的到达时间?因为，如果乘客初始的上车时间很大，最终未上车的乘客的等待时间即为当前时间步-到站时间
            wait_cost.append(w)
            if pax.origin in stop_wise_wait:    # pax.origin乘客初始所在的公交站
                stop_wise_wait[pax.origin].append(w)
            else:
                stop_wise_wait[pax.origin]=[w]
            if pax.onboard_time<99999999:       # 表示乘客上车了
                boards.append(pax.onboard_time )
                if pax.alight_time<999999:
                    travel_cost.append(pax.alight_time-pax.onboard_time )
                    add_travel_time = pax.alight_time - pax.onboard_time - pax.onroad_cost
                    delay.append(pax.alight_time-pax.arr_time-pax.onroad_cost)   # pax.onroad_cost是乘客在车上车辆运行时的时间
                    if pax.origin in stop_addition_travel:
                        # stop_addition_travel[pax.origin].append(add_travel_time)
                        # stop_addition_travel[pax.origin].append(travel_cost)
                        stop_addition_travel[pax.origin].append(pax.alight_time-pax.arr_time)
                    else:
                        # stop_addition_travel[pax.origin] = [add_travel_time]
                        # stop_addition_travel[pax.origin] = [travel_cost]
                        stop_addition_travel[pax.origin] = [pax.alight_time - pax.arr_time]
            else:

               still_wait+=1

        hold_cost = []   # 记录所有公交在所有站点的驻站时间
        for bus_id, bus in self.bus_list.items():
            tt = [ ]    # 获取的是一个公交在所有站点的驻站时间
            for k,v in bus.stay.items():    # bus.stay存储的是公交停留过的站点
                if v>0:
                    tt.append(bus.hold_cost[k])   # bus.hold_cost记录的是公交在每个站点的驻站时间
                    hold_cost.append(bus.hold_cost[k])
                    if k in stop_wise_hold:
                        stop_wise_hold[k].append(bus.hold_cost[k])
                    else:
                        stop_wise_hold[k] = [bus.hold_cost[k]]   # 每个站点每辆车的驻站时间

        stop_wise_wait_order = []  # 记录的是每个站点每个乘客的平均等待时间
        stop_wise_hold_order = []  # 记录的是每个站点每个车辆的平均驻站时间

        arr_times = []  # 每个元素里面都包含了 站点id和所有车辆到达站点的时间步
        buslog = pd.DataFrame()
        if len(bus.pass_stop) > 12:
            for bus_stop_id in bus.pass_stop[:12]:
                # buslog[bus_stop_id] = self.busstop_list[bus_stop_id].arr_log[bus.route_id]   # 每个长度不一样，放在dataframe中会报错
                arr_times.append([bus_stop_id]+self.busstop_list[bus_stop_id].arr_log[bus.route_id])
                try:
                    stop_wise_wait_order.append(np.mean(stop_wise_wait[bus_stop_id]))
                except:
                    stop_wise_wait_order.append(0)
                try:
                    stop_wise_hold_order.append(np.mean(stop_wise_hold[bus_stop_id]))
                except:
                    stop_wise_hold_order.append(0)
                for k, v in self.busstop_list[bus_stop_id].arr_log.items():
                    h = np.array(v)[1:] - np.array(v)[:-1]  # 后一辆车的到站时间与前一辆差值（每两辆车之间都计算）  在该站点，每两辆车的到站时间间隔

                    try:
                        headways_var[bus_stop_id].append(np.var(h))  # 计算的是每个站点 所有车辆到站时间的标准差（后向车头时距？）
                        headways_mean[bus_stop_id].append(np.mean(h))
                    except:
                        headways_var[bus_stop_id] = [np.var(h)]
                        headways_mean[bus_stop_id] = [np.mean(h)]
        else:
            for bus_stop_id in bus.pass_stop:  # 遍历公交线路上运行的最后一辆车经过的站点
                # buslog[bus_stop_id] = self.busstop_list[bus_stop_id].arr_log[bus.route_id]
                arr_times.append([bus_stop_id]+self.busstop_list[bus_stop_id].arr_log[bus.route_id])
                try:
                    stop_wise_wait_order.append(np.mean(stop_wise_wait[bus_stop_id]))
                except:
                    stop_wise_wait_order.append(0)
                try:
                    stop_wise_hold_order.append(np.mean(stop_wise_hold[bus_stop_id]))
                except:
                    stop_wise_hold_order.append(0)
                for k, v in self.busstop_list[bus_stop_id].arr_log.items():
                    h = np.array(v)[1:] - np.array(v)[:-1]  # 后一辆车的到站时间与前一辆差值（每两辆车之间都计算）  在该站点，每两辆车的到站时间间隔

                    try:
                        headways_var[bus_stop_id].append(np.var(h))  # 计算的是每个站点 所有车辆到站时间的标准差（后向车头时距？）
                        headways_mean[bus_stop_id].append(np.mean(h))
                    except:
                        headways_var[bus_stop_id] = [np.var(h)]
                        headways_mean[bus_stop_id] = [np.mean(h)]




        log = {}
        log['wait_cost'] = wait_cost          # 每个乘客的等待时间
        log['travel_cost'] = travel_cost      # 每个乘客的在车时间
        log['hold_cost'] = hold_cost          # 所有公交在所有站点的驻站时间
        log['headways_var'] = headways_var    # 每个站点所有车辆的后向车头时距的标准差
        log['headways_mean'] = headways_mean  # 每个站点所有车辆的后向车头时距的方差
        log['stw'] = stop_wise_wait_order     # 每个站点乘客的平均等待时间
        log['sth'] = stop_wise_hold_order     # 每个站点车辆的平均驻站时间
        log['bunching'] = self.bunching_times # 发生串车的次数
        log['delay'] = delay                  # 每个乘客的延误时间（等待时间+驻站时间）

        print('bunching times串车次数:%g headway mean车头时距均值 :%g hedaway var车头时距方差 :%g EV方差/均值的平方:%g'%(self.bunching_times, np.mean(list(headways_mean.values())),np.mean(list(headways_var.values())), (np.mean(list(headways_var.values()))/(np.mean(list(headways_mean.values()))**2))   ))

        AWT = []   # 每个站点乘客的平均等待时间
        AHD = []   # 每个站所有车辆的平均驻站时间
        AOD = []   # 每个站点所有公交到站时是的在车乘客数量的方差/均值
        ATT = []
        Bus0_AOD = []
        Bus1_AOD = []
        Bus2_AOD = []
        Bus3_AOD = []
        Bus4_AOD = []
        Bus5_AOD = []
        for k in bus.pass_stop[:12]:
            data = [np.mean(arr) for arr in stop_wise_hold[k]]   # 每个元素arr，代码首先检查arr是否是numpy数组（np.ndarray）。如果arr不是numpy数组，那么它会被封装在一个数组中，即np.array([arr])。如果arr已经是numpy数组，那么它会保持不变
            AHD.append(np.mean(data))
            try:
                if math.isnan(np.var(self.busstop_list[k].arr_bus_load) / np.mean(self.busstop_list[k].arr_bus_load)):   # math.isnan用于jiance括号中的值是不是nan,是nan返回true
                    AOD.append(0)
                else:
                    AOD.append(np.var(self.busstop_list[k].arr_bus_load) / np.mean(self.busstop_list[k].arr_bus_load))  # 每个站点公交到站时是的在车乘客数量的方差/均值
            except:
                AOD.append(0.)
            try:
                AWT.append(np.mean(stop_wise_wait[k]))
            except:
                AWT.append(0.)
            try:
                ATT.append(np.mean(stop_addition_travel[k]))
            except:
                ATT.append(0.)
            for ii,jj in self.busstop_list[k].arr_everybus_load.items():
                if ii == 0:
                    Bus0_AOD.append(np.sum(jj))
                elif ii==1:
                    Bus1_AOD.append(np.sum(jj))
                elif ii == 2:
                    Bus2_AOD.append(np.sum(jj))
                elif ii == 3:
                    Bus3_AOD.append(np.sum(jj))
                elif ii == 4:
                    Bus4_AOD.append(np.sum(jj))
                else:
                    Bus5_AOD.append(np.sum(jj))

        log['sto'] = AOD
        log['AOD'] = np.mean(AOD)
        log['att'] = ATT
        log['aht'] = AHD
        log['Bus0_AOD'] = Bus0_AOD
        log['Bus1_AOD'] = Bus1_AOD
        log['Bus2_AOD'] = Bus2_AOD
        log['Bus3_AOD'] = Bus3_AOD
        log['Bus4_AOD'] = Bus4_AOD
        log['Bus5_AOD'] = Bus5_AOD



        if train==0  :
            print('AWT:%g'%(np.mean(wait_cost)))
            print('AHD:%g' % (np.mean(AHD)))
            print('AOD:%g' % (np.mean(AOD)))
            print('headways_var:%g' % (np.sqrt(np.mean(list(headways_var.values())))))

        log['arr_times'] = arr_times    # 每个站点；所有车辆到达的时间步

        return log,np.mean(wait_cost),np.mean(AHD),np.mean(AOD),np.sqrt(np.mean(list(headways_var.values())))

    def close(self):
        return

    # update passengers when bus arriving at stops
    def serve(self,bus,stop):
        board_cost = 0
        alight_cost = 0
        board_pax = []
        alight_pax = []
        if bus!=None:
            # 乘客下车
            alight_pax = bus.pax_alight_fix(stop, self.pax_list)    # 返回包含下车乘客的编号列表    self.pax_list存储乘客的字典
            for p in alight_pax:
                self.pax_list[p].alight_time = self.simulation_step    # 将要下车的乘客的下车时间设置为当前时间步   这儿所有下车乘客的下车时间相同
                bus.onboard_list.remove(p)     # remove接受的是要删除的元素的值
                self.arr_pax_list[p] = self.pax_list[p]    # 用于存储完成行程的乘客
            # 乘客下车时间
            alight_cost = len(alight_pax) * bus.alight_period     # 下车花费的时间=下车人数*每位乘客的下车时间
            # 乘客上车
            new_arr = stop.pax_gen_od(bus, sim_step=self.simulation_step)
            num = len(self.pax_list) + 1
            for t in sorted(new_arr):  # self.pax_list列表中
                self.pax_list[num] = Passenger(id=num, origin=stop.id, arr_time=t)  # 为self.pax_list字典中添加一个乘客，乘客的序号为num
                # self.pax_list[num].took_bus = bus.id
                self.pax_list[num].route = bus.route_id
                # 为乘客设置下车站点
                s = np.random.randint(1, len(self.busstop_list))
                alight_index = (int(stop.id) + s) % len(self.busstop_list)
                self.pax_list[num].dest = str(alight_index)  # 设置乘客的下车站点___________________________________________________设置了乘客的下车站点
                self.busstop_list[stop.id].waiting_list.append(num)  # 更新在该站点等待乘客的列表，增加该乘客的id
                num += 1
            pax_leave_stop = []
            # 在站点等待上车的乘客——到达站时间从小到达排列
            waitinglist = sorted(self.busstop_list[stop.id].waiting_list)[:]
            for num in waitinglist:  # waitinglist是在该站点等待上车的乘客。self.pax_list是包含整个环境中所有的乘客
                # add logic to consider multiline impact (i.e. the passenger can not board bus this time can board the bus with same destination later?)
                if bus != None and self.pax_list[num].route == bus.route_id:
                    self.pax_list[num].miss += 1  # 记录乘客不能登上到达公交车站点公交车的次数
                # 乘客登上到达的车辆
                if bus != None and bus.capacity - len(bus.onboard_list) > 0 and self.pax_list[num].route == bus.route_id:  # 还未超过车辆的容纳能力
                    self.pax_list[num].onboard_time = self.simulation_step  # 设置乘客的上车时间——当车辆容量够时，在站点等待的乘客的上车时间是相同的,到站实践是不同的——————————————————————————————————————————设置乘客上车时间
                    self.pax_list[num].took_bus = bus.id
                    bus.onboard_list.append(num)  # 更新在车乘客的列表
                    board_cost += bus.board_period  # 上车时间每人3秒
                    pax_leave_stop.append(num)  # 更新从站点离开的乘客列表
            for num in pax_leave_stop:
                self.busstop_list[stop.id].waiting_list.remove(num)
        return alight_cost, board_cost,len(alight_pax),len(waitinglist)

    def sim(self):
        # update bus state

        ## dispatch bus  更新已经发车的车辆列表，包括增加新发车的车辆以及移除完成行程的车辆
        for bus_id, bus in self.bus_list.items():
            # 这个if用来让公交发车，设置公交车的当前速度，并将该公交车添加进已经发车的公交列表中
            if bus.is_dispatch==0 and bus.dispatch_time<=self.simulation_step:   # self.simulation_step:24149  is_dispatch=0表示公交车还未发出
                bus.is_dispatch=1    # 当公交车的发车时间小于等于仿真时间步，设置公交车的属性is_dispatch=1，表示发车了
                bus.arrival_schedule = self.simulation_step
                if bus.is_virtual != 1:  # true
                    if bus.speed_type == 1:
                        bus.current_speed = bus.speed  # 由此可以得到，公交在站间的运行时间是介于[201,400]之间的
                    else:
                        bus.current_speed = bus.speed * np.random.randint(90, 110) / 100.  # 随机数不包括120  #    bus.speed=0.005555555555555556km/s
                else:
                    bus.current_speed = bus.speed*0.8
                self.dispatch_buslist[bus_id]=bus    # 用来存储已经发车的公交列表？

        ## bus dynamic
        for bus_id, bus in self.dispatch_buslist.items():    #
            bus.serve_remain = max(bus.serve_remain - 1,0)   # 用来记录公交车在站点的剩余服务时间的   最小值为0
            bus.hold_remain = max(bus.hold_remain - 1, 0)    # 用来记录公交车在站点的剩余驻站时间的   最小值为0
            # 初始时bus.is_virtual==0，arr=0表示公交在路上，arr=1表示公交到站
            if bus.is_virtual==1 and bus.arr==0 and abs(bus.loc[-1]-bus.stop_dist[bus.left_stop[0]])<bus.speed :
                curr_stop = self.busstop_list[bus.left_stop[0]]
                bus.hold_remain = 0
                bus.serve_remain = 0
                bus.pass_stop.append(curr_stop.id)
                bus.left_stop = bus.left_stop[1:]
                bus.arr = 1

            ### on-arrival   相当于公交车已经到站了   bus.left_stop指公交剩余未到达的站点的序号，bus.stop_dist存储公交站的位置   bus.loc记录公交车实时的位置
            if bus.is_virtual==0 and bus.arr==0 and abs(bus.loc[-1]-bus.stop_dist[bus.left_stop[0]])<bus.speed :   # 如果公交车的实时位置与将要到达的站点之间的距离小于公交的速度，就让公交车到站, bus_stop_dist记录的是每个公交站的位置
                #### determine boarding and alight cost

                if bus.left_stop[0] not in self.busstop_list:  # 如果该站点不在公交站的列表中，就将该站点添加进公交站列表
                    self.busstop_list[bus.left_stop[0]] = self.busstop_list[bus.left_stop[0].split('_')[0]]
                # 当前站点为公交车到达的站点
                curr_stop = self.busstop_list[bus.left_stop[0]]     # 设置公交到达的当前站点 即为 剩余未到达的站点中的第一个站点
                # 记录公交站的属性：公交到达站点是公交车的在车人数
                self.busstop_list[bus.left_stop[0]].arr_bus_load.append(len(bus.onboard_list))    # 记录公交到达站点时，公交车上的人数
                '''增加一个字典，记录每个每辆公交车到站时的在车人数'''
                self.busstop_list[bus.left_stop[0]].arr_everybus_load[bus.id].append(len(bus.onboard_list))

                '''增加一个存储公交到站时还未进行上车车过程的在车乘客列表'''
                bus.previous_onboard_list = bus.onboard_list[:]
                # 记录公交站的属性：车辆到达站点的时间
                if bus.route_id in self.busstop_list[curr_stop.id].arr_log:    # arr_log是公交站的属性，记录每个公交线路上，公交车到达该站的时间
                    self.busstop_list[curr_stop.id].arr_log[bus.route_id].append(self.simulation_step)        #([bus.id, self.simulation_step])
                else:   # 为公交站的属性arr_log添加元素，表示某条公交线路上某个公交到达该站点的时间
                    self.busstop_list[curr_stop.id].arr_log[bus.route_id] =[self.simulation_step]# [[bus.id, self.simulation_step]]
                '''增加一个计算公交到站间隔的'''
                if len(self.busstop_list[curr_stop.id].arr_log[bus.route_id]) >= 2:
                    self.busstop_list[curr_stop.id].bus_arr_interval.append(abs(
                        self.busstop_list[curr_stop.id].arr_log[bus.route_id][-1] -
                        self.busstop_list[curr_stop.id].arr_log[bus.route_id][-2]))
                else:
                    self.busstop_list[curr_stop.id].bus_arr_interval.append(
                        abs(self.busstop_list[curr_stop.id].arr_log[bus.route_id][-1] - 0))

                # 公交到站时，计算在该站点，服务乘客上下车的时间  wait_num是在该公交站点等待上车的全部乘客
                alight_cost,board_cost,alight_num,wait_num = self.serve(bus,curr_stop)   # 公交到站时，上下客共需要花费多长时间
                # 更新公交车辆的状态
                bus.arr=1     # 表示公交车到站了    当公交离开站点时，arr=0
                '''增加一个公交的到站时间'''
                bus.arr_time = self.simulation_step
                bus.arr_time_list.append(bus.arr_time)
                # 更新公交车的服务时间，在上车时间和下车时间中取最大值。 ？？为什么+1？？
                bus.serve_remain = max(board_cost,alight_cost)+1.    # 表示公交车在站点的服务时间（但为什么要加1） 加1可能是启停过程需要的时间？
                bus.serve_over_time.append(float(self.simulation_step + bus.serve_remain))
                bus.stay[curr_stop.id] = 1     # 记录公交车在相应站点停车，停了为1
                # 公交在站点服务乘客花费的时间
                bus.cost[curr_stop.id] = bus.serve_remain    # 记录公交在每个站点的服务时间   # 这儿可以改写为 += bus.serve_remain
                bus.pass_stop.append(curr_stop.id)   # 公交车已经经过的站点列表+1
                bus.left_stop = bus.left_stop[1:] + [curr_stop.id]

                # 判断是否进行驻站控制（条件：1、公交到达站点；2、到达的站点不是公交第一个到达的站点，也就是公交车不是刚从第一站发车；3、车辆不是线路上第一辆车
                ## if determine holding once arriving            # 公交到站了，且该站不是该车辆第一个到达的公交站，且该公交不是该线路上第一个发出的车辆（公交车到达的第一个站点和线路上的第一辆公交车不进行驻站控制）
                # if self.hold_once_arr==1 and len(bus.pass_stop)>1 and self.dispatch_times[bus.route_id].index(bus.dispatch_time)>0 :#and len(self.dispatch_buslist)>2 and len(bus.pass_stop)>2 and len(bus.left_stop)>1 and bus.forward_bus!=None:
                '''如果线路是循环的，应该改为在始发站不进行驻站控制，也就是在第一个公交站不进行驻站控制'''

                # if self.hold_once_arr == 1 and len(bus.pass_stop) > 1:

                # if (self.hold_once_arr == 1 and bus.id != 0) or (self.hold_once_arr == 1 and bus.id == 0 and bus.forward_bus in self.dispatch_buslist):
                if self.simulation_step > 60 * 25:
                    if self.hold_once_arr == 1 and len(bus.pass_stop) > 1:
                        # 记录每个时间步公交到站的情况，包含{当前站点，车辆id,在车乘客数量}
                        if self.simulation_step in self.arrivals:    # self.arrivals是一个字典，记录公交到站时的时间步对应的站点id,公交id，以及在车乘客数量
                            self.arrivals[self.simulation_step].append([curr_stop.id, bus_id, len(bus.onboard_list)])
                        else:
                            self.arrivals[self.simulation_step] = [[curr_stop.id, bus_id, len(bus.onboard_list)]]   # 用于存储公交驻站控制的时间步对应的站点id，公交id，公交车上的乘客数量
                        # 公交驻站时间  秒   id=0的公交车在最后一辆车id=6发出前不进行驻站控制

                        bus.hold_remain = self.control(bus, curr_stop,type=self.control_type)   # 返回的应该是公交的驻站时间   self.control_type=2强化学习控制，0是无控制
                        bus.arrival_schedule = self.simulation_step + 60. * 6
                        # 记录公交到达过该站点
                        if bus.hold_remain > 0:
                            bus.stay[curr_stop.id] = 1
                        # 公交的驻站时间小于10min时需略驻站控制
                        if bus.hold_remain<30:    # 控制时间小于10则忽略？
                            bus.hold_remain = 0

                        # 记录公交在每个站点驻站时长
                        bus.hold_cost[curr_stop.id].append(bus.hold_remain)
                        # bus.hold_cost[curr_stop.id] += bus.hold_remain    # 用来记录公交在每个站点的驻站控制的时长  '''这儿更改为 += '''
                        # 未进行驻站控制是0，进行驻站控制为1
                        bus.is_hold = 1
                    # 上述if条件表示：线路上发出的第一辆车没有驻站控制；所有公交的始发站不进行驻站控制

            # 公交车由于上下客或驻站控制在站点停留
            if bus.hold_remain>0 or bus.serve_remain>0:   # 控制时间或者服务时间大于0，公交在站点停车
                bus.stop()

            # 表示公交车要离开站点
            else: # 公交开始运动，但是要先判断是否允许公交超车
                if self.is_allow_overtake == 1:    # =1应该是允许超车
                    bus.dep()
                else:# 不允许超车
                    # 如果前面的车发车了，并且当前车辆的位置大于等于前车的位置
                    if bus.id == 0:
                        if bus.forward_bus in self.dispatch_buslist and (2 * np.pi - bus.travel_sum + self.dispatch_buslist[bus.forward_bus].travel_sum -bus.current_speed) <=0:
                            bus.stop()
                            if bus.speed_type == 1:
                                bus.current_speed = bus.speed  # 由此可以得到，公交在站间的运行时间是介于[201,400]之间的
                            else:
                                bus.current_speed = bus.speed * np.random.randint(90, 110) / 100.
                            if bus.b == 0:  # 初始时bus.b=0
                                self.bunching_times += 1
                                bus.b = 1
                        else:
                            if bus.arr == 1:
                                bus.leave_stop_time = float(self.simulation_step)
                            bus.b = 0
                            bus.dep(bus.current_speed)  # 公交开始按照给定的速度运动，更新公交车的实时位置
                            for p in bus.onboard_list:
                                self.pax_list[p].onroad_cost += 1  # 车上乘客的在车时间成本+1
                            if len(bus.pass_stop) > 0:
                                if bus.route_id in self.busstop_list:  # 这儿永远都不可能满足要求呀
                                    self.busstop_list[bus.pass_stop[-1]].dep_log[bus.route_id].append(
                                        [bus.id, self.simulation_step])
                                else:  # 记录每辆公交车在每个站点的发车时间
                                    self.busstop_list[bus.pass_stop[-1]].dep_log[bus.route_id] = [
                                        [bus.id, self.simulation_step]]

                    else:
                        if bus.current_speed + bus.travel_sum >= self.dispatch_buslist[bus.forward_bus].travel_sum:
                            bus.stop()
                            if bus.speed_type == 1:
                                bus.current_speed = bus.speed  # 由此可以得到，公交在站间的运行时间是介于[201,400]之间的
                            else:
                                bus.current_speed = bus.speed * np.random.randint(90, 110) / 100.

                            if bus.b == 0:  # 初始时bus.b=0
                                self.bunching_times += 1
                                bus.b = 1
                        else:
                            if bus.arr == 1:
                                bus.leave_stop_time = float(self.simulation_step)
                            bus.b = 0
                            bus.dep(bus.current_speed)  # 公交开始按照给定的速度运动，更新公交车的实时位置
                            for p in bus.onboard_list:
                                self.pax_list[p].onroad_cost += 1  # 车上乘客的在车时间成本+1
                            if len(bus.pass_stop) > 0:
                                if bus.route_id in self.busstop_list:  # 这儿永远都不可能满足要求呀
                                    self.busstop_list[bus.pass_stop[-1]].dep_log[bus.route_id].append([bus.id, self.simulation_step])

                                else:  # 记录每辆公交车在每个站点的发车时间
                                    self.busstop_list[bus.pass_stop[-1]].dep_log[bus.route_id] = [[bus.id, self.simulation_step]]


        self.simulation_step+=1   # 仿真时间步+1
        Flag = True
        # 一个episode为3个小时
        if self.simulation_step > 60 * 60 * 3:
            Flag = False
        # for bus_id, bus in self.bus_list.items():
        #     if bus.is_dispatch!=-1:       # 公交的行程结束时为-1，否则为0或1（发车）
        #         Flag = True               # 当公交列表中全部车辆的行程都结束，Flag的值才会未Flase,否则Flag为True
        return Flag
    # 确定控制方法种类
    def control(self,bus,bus_stop,type=0):
        fh_ = bus_stop.bus_arr_interval[-1]
        fh, bh = self.cal_headway(bus)
        if type==0:
            return 0
        if type==1:
            fh, bh = self.cal_headway(bus)
            if bus.forward_bus==None:
                return 0
            else:
                return max(0, min(self.max_hold_time, 58 + 0.05 * (360. - fh_)))#max(0, 58 + 0.05 * ( (self.mfh - fh)))#
                # return max(0, min(self.max_hold_time,5.8*60 + 0.8 * (360. - fh_)))#2020年论文中的取值
        # 强化学习进行驻站控制
        if type==2:   # 类型2应该指的是使用强化学习进行驻站控制
            return self.rl_control(bus,bus_stop)     # 返回的是驻站时间
        if type==3:   # BH
            if bus.backward_bus == None:
                return 0
            else:
                # return max(0,min(self.max_hold_time,self.alpha * bh))
                return max(0, min(self.max_hold_time, 58 + self.alpha * (bh - 360.)))  # 这儿要不要减360.
        if type==4:   # HH   固定车头时距的控制——当前向车头时距小于期望的车头时距时进行驻站控制d = H0-h-
            # fh = bus_stop.bus_arr_interval[-1]
            if bus.forward_bus==None:
                return 0
            else:
                return max(0,min(self.max_hold_time,360.-fh_))
                # return max(0,min(self.max_hold_time,0.8 * 360.-fh))

        if type == 5:  # 基于时刻表的驻站控制
            return max(0,min(self.max_hold_time, bus.arrival_schedule - self.simulation_step))

        return 0

    # 强化学习进行驻站控制
    def rl_control(self, bus, bus_stop):
        # reward_scaling = RewardScaling(shape=1, gamma=self.agents[bus.route_id].gamma)  # args.gamma
        # reward_scaling.reset()
        # retrieve historical state
        current_interval = self.simulation_step
        state = []   # 第一项是公交车在车乘客占公交容量的百分比
        for record in self.arrivals[current_interval]:   # self.arrivals记录的是相应的时间步下，公交到站的情况；公交站点、公交车、以及在车乘客数量的信息列表（在公交车到站时，将相应信息记录到改字典中）
            # record是一条记录，是一个列表，分别是：公交到达的站点，车辆id,在车人数
            bus_stop_id_ = record[0]  # 公交到达得站点
            bus_id_ = record[1]       # 车辆id
            onboard = record[2]       # 在车人数
            if bus_id_ == bus.id:
                state = [onboard / bus.capacity]    # 获取车上乘客的占比    此时的在车乘客已经是更新后的在车乘客了
                break   # 退除for循环
        # 获取公交前后向车头时距
        fh, bh = self.cal_headway(bus)         # 返回前向车头时距和后向车头时距
        var, mean = self.route_info(bus)       # 对象路所有已经发车的车辆的后向车头时距求均值和方差
        # # 将前向车头时距和后向车头时距添加仅车辆的状态列表中
        # state += [min(fh / 600., 2.), min(bh / 600., 2.)]    # 为什么要这样处理？    600s=10min    车头时距的大小不大于2min?
        # state += [fh/600, bh/600]
        state += [fh, bh]
        self.state_record.append(state)
        '''公交到站了，记录公交从开始到达站点的时间内，其他智能体的到站情况'''

        if len(self.GM.temp_memory[bus.id]['a']) > 0:
            fp = [state + [0.] + [0.] + [0.] + [0.] + [bus.id]]
            temp = bus.last_vist_interval  # 最初的bus.last_vist_interval 为公交的发车时间
            while temp <= current_interval:  # 为了寻找这个在这个公交上一次驻站控制到这一次进行驻站控制期间，其他进行驻站控制的公交
                if temp in self.arrivals:  # self.arrivals记录相应时间步，进行驻站控制的车辆、公交站、公交的在车人数
                    for record in self.arrivals[temp]:
                        bus_stop_id_ = record[0]
                        bus_id_ = record[1]
                        onboard = record[2]
                        if bus_id_ == bus.id:  # 需要考虑的是其他智能体，当是自身智能体时，推出当前循环，进行下一次循环
                            continue
                        # 只关心两次驻站期间，与之相邻的前后公交车的驻站情况
                        if (bus_id_ == bus.forward_bus or bus_id_ == bus.backward_bus) or (
                                self.all == 1):  # self.all=0,所以前面的条件必须满足其一
                            curr_bus = self.dispatch_times[bus.route_id].index(
                                bus.dispatch_time)  # 获取当前车辆发车时间在线路发车时间列表中的索引
                            neigh_bus = self.dispatch_times[bus.route_id].index(
                                self.bus_list[bus_id_].dispatch_time)  # 获取相邻车辆发车时间的索引
                            # 辆车之间的距离，索引差/车辆数
                            bus_dist = [(curr_bus - neigh_bus) / len(self.bus_list)]  # 发车时间索引之差/公交数量(两控制点之间公交车的数量)
                            stop_dist = [
                                (bus.stop_list.index(bus.pass_stop[-2]) - bus.stop_list.index(bus_stop_id_)) / len(
                                    self.busstop_list)]  # 当前公交上一次到站的站点索引与相邻公交主站控制的索引之差/公交站数量(两控制点之间公交站的数量)
                            fp.append(self.bus_list[bus_id_].his[temp] + stop_dist + bus_dist + [
                                abs(temp - current_interval)] + [
                                          bus_id_])  # 边际贡献：相邻公交站驻站是的状态动作+bus_dist+stop_dist+当前时间步与该公交站驻站时间步之差+相邻公交的索引

                temp += 1
            # reward1 = (-var / mean / mean) * (1 - self.weight) * 5   # 5是什么？    # 论文公式二的第一项（多乘了5）
            # reward2 = (-abs(self.GM.temp_memory[bus.id]['a'][-1])) * self.weight  # 论文公式二的第二项
            # reward = reward1 + reward2

            wait_time = 0
            # on_time = 0
            wait_time1 = 0
            for num_ in bus.previous_onboard_list:
                if self.pax_list[num_].origin == bus.pass_stop[-2]:  # 在前一站上车的乘客
                    # 在前一站上车的乘客的等待时间=乘客上车时间-乘客的到站时间 (等待车辆到站的时间)
                    wait_time += abs(self.pax_list[num_].onboard_time - self.pax_list[num_].arr_time)  # / 3600
                    # 在车时间 = 当前时间步 - 乘客的上车时间
                    # on_time += abs(self.simulation_step - bus.leave_stop_time)/3600
                    if bus.leave_stop_time > bus.serve_over_time[-2]:  # 表示有驻站时间
                        # 因为驻站造成的乘客额外等待时间 = 车辆离开站点的时间步 - 乘客上下车过程完成的时间步
                        wait_time1 += (bus.leave_stop_time - bus.serve_over_time[-2])  # / 3600
                else:  # 这是在车乘客的等待时间，等于额外增加的驻站时间
                    # on_time += abs(self.simulation_step - bus.leave_stop_time)/3600  # 在车时间仅考虑车辆上一次到站时间与当前到站时间之差
                    if bus.leave_stop_time > bus.serve_over_time[-2]:
                        wait_time1 += (bus.leave_stop_time - bus.serve_over_time[-2])  # / 3600
            reward1 = (-var / mean / mean) * (1 - self.weight) * 5
            # reward1 = (-abs(fh - 360.) / 360.) * (1 - self.weight)
            # reward2 = (- wait_time1 - wait_time ) * self.weight
            try:
                reward2 = (- wait_time1 - wait_time) / len(bus.previous_onboard_list) * self.weight * 0.01

            except:

                reward2 = 0
            reward = reward1 + reward2

            self.reward_record.append(reward)
            self.reward_signal[bus.id].append(reward)
            self.reward_signalp1[bus.id].append(reward1)
            self.reward_signalp2[bus.id].append(reward2)

            self.GM.temp_memory[bus.id]['r'].append(reward)
        else:
            # stop_dist = [0.]
            # bus_dist = [0.]
            fp = [state + [0.] + [0.] + [0.] + [0.] + [bus.id]]
            # temp = bus.last_vist_interval    # 最初的bus.last_vist_interval 为公交的发车时间
            temp = bus.arr_time_list[-2]
            while temp <= current_interval:  # 为了寻找这个在这个公交上一次驻站控制到这一次进行驻站控制期间，其他进行驻站控制的公交
                if temp in self.arrivals:  # self.arrivals记录相应时间步，进行驻站控制的车辆、公交站、公交的在车人数
                    for record in self.arrivals[temp]:
                        bus_stop_id_ = record[0]
                        bus_id_ = record[1]
                        onboard = record[2]
                        if bus_id_ == bus.id:  # 需要考虑的是其他智能体，当是自身智能体时，推出当前循环，进行下一次循环
                            continue
                        # 只关心两次驻站期间，与之相邻的前后公交车的驻站情况
                        if (bus_id_ == bus.forward_bus or bus_id_ == bus.backward_bus) or (self.all == 1):  # self.all=0,所以前面的条件必须满足其一
                            curr_bus = self.dispatch_times[bus.route_id].index(bus.dispatch_time)  # 获取当前车辆发车时间在线路发车时间列表中的索引
                            neigh_bus = self.dispatch_times[bus.route_id].index(self.bus_list[bus_id_].dispatch_time)  # 获取相邻车辆发车时间的索引
                            # 辆车之间的距离，索引差/车辆数
                            bus_dist = [(curr_bus - neigh_bus) / len(self.bus_list)]  # 发车时间索引之差/公交数量(两控制点之间公交车的数量)
                            stop_dist = [(bus.stop_list.index(bus.pass_stop[-1]) - bus.stop_list.index(bus_stop_id_)) / len(self.busstop_list)]  # 当前公交上一次到站的站点索引与相邻公交主站控制的索引之差/公交站数量(两控制点之间公交站的数量)
                            fp.append(self.bus_list[bus_id_].his[temp] + stop_dist + bus_dist + [abs(temp - current_interval)] + [bus_id_])  # 边际贡献：相邻公交站驻站是的状态动作+bus_dist+stop_dist+当前时间步与该公交站驻站时间步之差+相邻公交的索引

                temp += 1


        if self.share_scale == 0:
            action = np.array(self.agents[bus.route_id].choose_action(np.array(state).reshape(-1, )))
        # 根据智能体的状态输出智能体的动作
        if self.share_scale == 1:    # 默认设置值为1，应该表示参数是共享的
            a_ = [0., 0.]
            if len(fp) <= 1:
                state += a_
            else:
                for n_ in range(len(fp)):
                    f_p = fp[n_]  # 节点特征
                    b_i = fp[n_][-1]  # 车辆id
                    b_gap = fp[n_][-2]
                    if n_ == 0:
                        b_i_d = b_i
                    if n_ > 0 and b_gap == 0:
                        if self.bus_list[b_i].dispatch_time > self.bus_list[b_i_d].dispatch_time:
                            a_[-1] = f_p[3]
                        else:
                            a_[0] = f_p[3]
                state += a_
            # action,action_log_prob = np.array(self.agents.choose_action(np.array(state).reshape(-1, )))
            action, action_log_prob = np.array(self.agents[bus.route_id].choose_action(state))

        # mark记录了状态动作对
        mark = list(np.array(state + list(action)).reshape(-1, ))    # 状态+动作
        # mark = list(np.array(state).reshape(-1, ))
        self.bus_list[bus.id].his[self.simulation_step] = mark  # his是用来存储对应时间步下的状态和动作的字典

        self.GM.temp_memory[bus.id]['s'].append(state)
        self.GM.temp_memory[bus.id]['a'].append(action)
        self.GM.temp_memory[bus.id]['a_log_prob'].append(action_log_prob)
        self.GM.temp_memory[bus.id]['fp'].append(fp)

        self.action_record.append(action)
        action = np.clip(abs(action), 0., 1.)  # 为什么是0到3
        self.bus_list[bus.id].last_vist_interval = self.simulation_step  # 更新该公交的上一个驻站控制时间为当前时间步
        return int(np.asarray(180. * action).item())  # NumPy 2.x: extract scalar before int conversion
    # 获取公交的前后向车头时距：需要判断车辆的id
    def cal_headway(self,bus):
        if bus.id == 0:
            if bus.backward_bus in self.dispatch_buslist:
                bh = (bus.travel_sum - self.bus_list[bus.backward_bus].travel_sum) / bus.c_speed
            else:
                bh = abs(bus.loc[-1] - 0.) / bus.c_speed
            if bus.forward_bus in self.dispatch_buslist:
                fh = (self.bus_list[bus.forward_bus].travel_sum + np.pi * 2 - bus.travel_sum) / bus.c_speed
            else:
                fh = bh  # 设置成跟bh一样还是设置成发车间隔
                # fh = (np.pi * 2 - bus.loc[-1]) / bus.c_speed
        elif bus.id == len(self.bus_list) - 1:
            fh = (self.bus_list[bus.forward_bus].travel_sum - bus.travel_sum) / bus.c_speed
            bh = (bus.travel_sum + np.pi * 2 - self.bus_list[bus.backward_bus].travel_sum) / bus.c_speed
        else:
            if bus.backward_bus in self.dispatch_buslist:
                bh = (bus.travel_sum - self.bus_list[bus.backward_bus].travel_sum) / bus.c_speed
            else:
                bh = abs(bus.loc[-1] - 0.) / bus.c_speed
            fh = (self.bus_list[bus.forward_bus].travel_sum - bus.travel_sum) / bus.c_speed
        return fh, bh


    def route_info(self,bus):
        fh = [360 for _ in range(3)]  # fh = [500,500,500]
        bh = [360 for _ in range(3)]  # bh = [500,500,500]
        # fh = []
        # bh = []

        for bus_id, bus_ in self.dispatch_buslist.items():  # 遍历 所有 已经发出的公交车列表
            # if bus_.route_id == bus.route_id:
            if bus_.id == 0:
                if bus_.backward_bus in self.dispatch_buslist:
                    bh_ = (bus_.travel_sum - self.bus_list[bus_.backward_bus].travel_sum) / bus_.c_speed
                    bh.append(bh_)
                # else:
                #     bh_ = abs(bus_.loc[-1] - 0.) / bus_.c_speed
                #     bh.append(bh_)
                # if bus_.forward_bus in self.dispatch_buslist:
                #     fh.append(
                #         (self.bus_list[bus_.forward_bus].travel_sum + np.pi * 2 - bus_.travel_sum) / bus_.c_speed)
                # else:
                #     fh_ = bh_  # 设置成跟bh一样还是设置成发车间隔
                #     fh.append(fh_)
                    # fh = (np.pi * 2 - bus.loc[-1]) / bus.c_speed

            elif bus_.id == len(self.bus_list) - 1:
                if bus_.backward_bus in self.dispatch_buslist:
                # fh.append((self.bus_list[bus_.forward_bus].travel_sum - bus_.travel_sum) / bus_.c_speed)
                    bh.append((bus_.travel_sum + np.pi * 2 - self.bus_list[bus_.backward_bus].travel_sum) / bus_.c_speed)
            else:
                if bus_.backward_bus in self.dispatch_buslist:
                    bh.append((bus_.travel_sum - self.bus_list[bus_.backward_bus].travel_sum) / bus_.c_speed)
                # else:
                #     bh.append(abs(bus_.loc[-1] - 0.) / bus_.c_speed)
                # fh.append((self.bus_list[bus_.forward_bus].travel_sum - bus_.travel_sum) / bus_.c_speed)
        if len(bh) < 2:  # 不可能满足条件，长度至少也为3
            return 999999, 999999

        return np.var(bh), np.mean(bh)


    def learn(self):
        ploss_set = []
        qloss_set = []
        # self.share_scale == 1
        if self.share_scale == 0:  # false
            for bus_id, bus in self.bus_list.items():
                if (len(self.GM.memory[bus_id]) + 1) > 16:
                    ploss, qloss = self.agents[bus.id].learn(self.GM.memory[bus_id])
                    try:
                        self.qloss[bus.id].append(np.mean(qloss))
                    except:
                        self.qloss[bus.id] = [np.mean(qloss)]
                    ploss_set.append(ploss)
                    qloss_set.append(qloss)
        # 随机选择一个公交车，根据经验回放池中存储的数据进行训练
        if self.share_scale == 1:  # true
            '''为了计算每个智能体的优势函数，累积回报'''
            obs,w_obs, actions, old_action_log_probs,value_pred, returns, adv_targ = [], [], [], [], [], [], []
            for rid, r in self.route_list.items():  # 只有一次循环
                for _ in self.GM.temp_memory.values():   # 分别为每辆车的数据
                    if len(_['s']) <= 2:
                        continue
                    # print((r.bus_list).index(b))
                    returns_, adv_targ_ = [], []  # 用来记录每个智能体每一步的回报和优势值
                    s_ = _['s'][:-1]  # 包含了当前状态 和 下一状态(列表)
                    fp_ = _['fp'][:-1]
                    # fp_ = _['fp']
                    ws_ = [s__[:3] for s__ in s_]
                    # ws_ = s_[:]  # z
                    for i_ in range(len(fp_)):
                        ws = [[0., 0., 0.] for i in range(2)]
                        if len(fp_[i_]) <= 1:
                            ws_[i_] = [ws_[i_]] + ws
                        else:
                            for n in range(len(fp_[i_])):
                                _fp = fp_[i_][n]  # 节点特征
                                b__ = _['fp'][i_][n][-1]  # 车辆id
                                gap = _['fp'][i_][n][-2]
                                if n == 0:
                                    b_id = b__
                                if n > 0 and gap == 0:
                                    if self.bus_list[b__].dispatch_time > self.bus_list[b_id].dispatch_time:
                                        ws[-1] = _fp[:3]
                                    else:
                                        ws[0] = _fp[:3]
                            ws_[i_] = [ws_[i_]] + ws

                    # 计算状态价值
                    values_ = self.agents[rid].critic(torch.tensor(ws_, dtype=torch.float32).reshape(len(ws_), -1)).cpu().detach()
                    values = values_.numpy()  # 包含了当前状态价值 和 下一状态价值
                    value_pred += values_.tolist()[:-1]
                    r_ = _['r'][:-1]

                    # 计算折扣回报
                    gae = 0
                    for i_ in reversed(range(len(r_))):
                        delta = r_[i_] + self.agents[rid].gamma * values[i_ + 1] - values[i_]
                        gae = delta + self.agents[rid].gamma * self.agents[rid].gae_lambda * gae
                        return_ = gae + values[i_]
                        returns_.insert(0, return_)
                        adv_targ_.insert(0, gae)
                    returns += returns_
                    adv_targ_copy = adv_targ_.copy()
                    mean_advantages = np.mean(adv_targ_copy)  # 忽略nan值  计算（非缺失值）元素的平均值
                    std_advantages = np.std(adv_targ_copy)
                    adv_targ_ = list((adv_targ_ - mean_advantages) / (std_advantages + 1e-5))
                    adv_targ += adv_targ_
                    actions += _['a'][:-2]
                    old_action_log_probs += _['a_log_prob'][:-2]
                    obs += s_[:-1]
                    # feature_p += whole_s[:-1]
                    # next_feature_p += whole_s[1:]
                    w_obs += ws_[:-1]

            obs = torch.tensor(obs, dtype=torch.float)
            # feature_p = torch.tensor(feature_p, dtype=torch.float)
            actions = torch.tensor(actions, dtype=torch.float)
            old_action_log_probs = torch.tensor(old_action_log_probs, dtype=torch.float)
            # next_feature_p = np.array(next_feature_p)
            # next_feature_p = torch.tensor(next_feature_p, dtype=torch.float)
            value_pred = torch.tensor(value_pred, dtype=torch.float)
            returns = torch.tensor(returns, dtype=torch.float)
            adv_targ = torch.tensor(adv_targ, dtype=torch.float)
            w_obs = torch.tensor(w_obs, dtype=torch.float).reshape(len(w_obs),-1)

            def feed_forward_generator(obs, w_obs, actions, old_action_log_probs,value_pred,
                                       returns, adv_targ, num_mini_batch):
                batch_size = len(obs)
                # for index in BatchSampler(SubsetRandomSampler(range(batch_size)), num_mini_batch, True):
                #     obs_batch = obs[index]
                #     w_obs_batch = w_obs[index]
                #     # feature_p_batch = feature_p[indices].tolist()
                #     # feature_p_batch = feature_p[indices].reshape(len(indices), -1)
                #     actions_batch = actions[index]
                #     old_action_log_probs_batch = old_action_log_probs[index]
                #     # next_feature_p_batch = next_feature_p[indices].reshape(len(indices), -1)
                #     value_pred_batch = value_pred[index]
                #     returns_batch = returns[index]
                #     adv_targ_batch = adv_targ[index]
                #
                #     yield obs_batch, w_obs_batch, actions_batch, old_action_log_probs_batch, value_pred_batch, returns_batch, adv_targ_batch

                mini_batch_size = batch_size // num_mini_batch  # 16是batch
                rand = torch.randperm(batch_size).numpy()
                sampler = [rand[_ * mini_batch_size:(_ + 1) * mini_batch_size] for _ in range(num_mini_batch)]
                for indices in sampler:
                    obs_batch = obs[indices]
                    w_obs_batch = w_obs[indices]
                    # feature_p_batch = feature_p[indices].tolist()
                    # feature_p_batch = feature_p[indices].reshape(len(indices), -1)
                    actions_batch = actions[indices]
                    old_action_log_probs_batch = old_action_log_probs[indices]
                    # next_feature_p_batch = next_feature_p[indices].reshape(len(indices), -1)
                    value_pred_batch = value_pred[indices]
                    returns_batch = returns[indices]
                    adv_targ_batch = adv_targ[indices]
                    yield obs_batch,w_obs_batch,  actions_batch, old_action_log_probs_batch,value_pred_batch, returns_batch, adv_targ_batch

            train_info = {}
            train_info['value_loss'] = 0
            train_info['policy_loss'] = 0
            train_info['dist_entropy'] = 0
            train_info['ratio'] = 0
            for rid, r in self.route_list.items():
                for _i in range(self.ppo_epoch):
                    data_generator = feed_forward_generator(obs,w_obs,actions, old_action_log_probs,
                                                            value_pred, returns, adv_targ,
                                                            self.num_mini_batch)
                    for sample in data_generator:
                        value_loss, policy_loss, imp_weights, dist_entropy = self.agents[rid].learn(sample)
                        train_info['value_loss'] += value_loss.item()
                        train_info['policy_loss'] += policy_loss.item()
                        train_info['dist_entropy'] += dist_entropy.item()
                        train_info['ratio'] += imp_weights.mean()

                num_updates = self.ppo_epoch * self.num_mini_batch
                # num_updates = self.ppo_epoch *  math.ceil(obs.shape[0] / self.num_mini_batch)
                for k in train_info.keys():
                    train_info[k] /= num_updates
        return train_info
