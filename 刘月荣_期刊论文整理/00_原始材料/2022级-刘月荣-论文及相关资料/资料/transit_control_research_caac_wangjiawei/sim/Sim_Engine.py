import numpy as np
from sim.Passenger import Passenger
from sim.Bus import Bus
from sim.Route import Route
import matplotlib.pyplot as plt
from model.Group_MemoryC import Memory
import pandas as pd
import time
import math
import warnings
warnings.filterwarnings("ignore")
class Engine():
    def __init__(self, bus_list,busstop_list,route_list,simulation_step,dispatch_times, demand=0,agents=None,share_scale=0, is_allow_overtake=0,hold_once_arr=1,control_type=1,seed=1,all=0,weight=0):

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
        print('total pax:%d'%(len(self.pax_list)))
        wait_cost = []
        travel_cost = []     # 记录乘客在车时间
        headways_var = {}
        headways_mean = {}
        boards = []     # 记录乘客上车的时间
        arrs = []
        origins = []
        dests = []
        still_wait = 0
        stop_wise_wait = {}    # 记录乘客在公交站等待的时长
        stop_wise_hold = {}
        delay = []
        for pax_id, pax in self.pax_list.items():
            w = min(pax.onboard_time - pax.arr_time, self.simulation_step-pax.arr_time)   # 乘客的等待上车的时间  但是为什么要用当前的时间步-乘客的到达时间
            wait_cost.append(w)
            if pax.origin in stop_wise_wait:    # pax.origin乘客初始所在的公交站
                stop_wise_wait[pax.origin].append(w)
            else:
                stop_wise_wait[pax.origin]=[w]
            if pax.onboard_time<99999999:
                boards.append(pax.onboard_time )
                if pax.alight_time<999999:
                    travel_cost.append(pax.alight_time-pax.onboard_time )
                    delay.append(pax.alight_time-pax.arr_time-pax.onroad_cost)
            else:
               still_wait+=1

        hold_cost = []
        for bus_id, bus in self.bus_list.items():
            tt = [ ]
            for k,v in bus.stay.items():
                if v>0:
                    tt.append(bus.hold_cost[k])
                    hold_cost.append(bus.hold_cost[k])
                    if k in stop_wise_hold:
                        stop_wise_hold[k].append(bus.hold_cost[k])
                    else:
                        stop_wise_hold[k] = [bus.hold_cost[k]]

        stop_wise_wait_order = []
        stop_wise_hold_order = []

        arr_times = []
        buslog = pd.DataFrame()
        for bus_stop_id in bus.pass_stop:
            buslog[bus_stop_id]=self.busstop_list[bus_stop_id].arr_log[bus.route_id]    # arr_log记录的是公交车到站的时间
            arr_times.append([bus_stop_id]+self.busstop_list[bus_stop_id].arr_log[bus.route_id])
            try:
                stop_wise_wait_order.append(np.mean(stop_wise_wait[ bus_stop_id ]))
            except:
                stop_wise_wait_order.append(0)
            try:
                stop_wise_hold_order.append(np.mean(stop_wise_hold[bus_stop_id]))
            except:
                stop_wise_hold_order.append(0)

            for k,v in self.busstop_list[bus_stop_id].arr_log.items():
                h =  np.array(v )[1:]  -  np.array(v)[:-1]   # 第一项不包含第一个元素，第二项不包含最后一个元素（也就是错位相减）

                try:
                    headways_var[bus_stop_id].append(np.var(h))
                    headways_mean[bus_stop_id].append(np.mean(h))
                except:
                    headways_var[bus_stop_id]=[np.var(h)]
                    headways_mean[bus_stop_id]=[np.mean(h)]

        log = {}
        log['wait_cost'] = wait_cost
        log['travel_cost'] = travel_cost
        log['hold_cost'] = hold_cost
        log['headways_var'] = headways_var
        log['headways_mean'] = headways_mean
        log['stw'] = stop_wise_wait_order
        log['sth'] = stop_wise_hold_order
        log['bunching'] = self.bunching_times
        log['delay'] = delay
        print('bunching times:%g headway mean:%g hedaway var:%g EV:%g'%(self.bunching_times, np.mean(list(headways_mean.values())),np.mean(list(headways_var.values())), (np.mean(list(headways_var.values()))/(np.mean(list(headways_mean.values()))**2))   ))

        AWT = []
        AHD = []
        AOD = []
        for k in bus.pass_stop:
            data = [np.array([arr]) if not isinstance(arr, np.ndarray) else arr for arr in stop_wise_hold[k]]
            AHD.append(np.mean(data))
            try:
                if math.isnan(np.var(self.busstop_list[k].arr_bus_load) / np.mean(self.busstop_list[k].arr_bus_load)):   # math.isnan用于jiance括号中的值是不是nan,是nan返回true
                    AOD.append(0)
                else:
                    AOD.append(np.var(self.busstop_list[k].arr_bus_load) / np.mean(self.busstop_list[k].arr_bus_load))
            except:
                AOD.append(0.)
            try:
                AWT.append(np.mean(stop_wise_wait[k]))
            except:
                AWT.append(0.)

        log['sto'] = AOD
        log['AOD'] = np.mean(AOD)

        if train==0  :
            print('AWT:%g'%(np.mean(wait_cost)))
            print('AHD:%g' % (np.mean(AHD)))
            print('AOD:%g' % (np.mean(AOD)))
            print('headways_var:%g' % (np.sqrt(np.mean(list(headways_var.values())))))

        log['arr_times'] = arr_times

        return log

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
            alight_pax = bus.pax_alight_fix(stop, self.pax_list)    # 返回包含下车乘客的列表    self.pax_list存储乘客的字典
            for p in alight_pax:
                self.pax_list[p].alight_time = self.simulation_step    # 将要下车的乘客的下车时间设置为当前时间步   这儿所有下车乘客的下车时间相同
                bus.onboard_list.remove(p)     # remove接受的是要删除的元素的值
                self.arr_pax_list[p] = self.pax_list[p]    # 用于存储完成行程的乘客
            # 乘客下车时间
            alight_cost = len(alight_pax) * bus.alight_period     # 下车花费的时间=下车人数*每位乘客的下车时间
            # 乘客上车
            # boarding procedure
            for d in stop.dest.keys():          # 是一个字典
                # 站点等待上车的的乘客到站时间
                new_arr = stop.pax_gen_od(bus, sim_step=self.simulation_step,dest_id=d)   # 生成的是一个列表，等价于pax[包含每个乘客的到站时间]   new_arr中每个元素记录的是每个乘客的到站时间，列表的长度为乘客的数量
                # new_arr的列表长度代表要在该站点等待的乘客数量，其中的元素代表乘客的到达该站点的时间
                if len(new_arr)==0:
                    continue    # 跳过当前循环，执行下一次循环
                # 生成上车乘客，更新等待上车的乘客字典
                num = len(self.pax_list) + 1      # self.pax_list应该是用来存储生成的乘客的
                for t in new_arr:
                    self.pax_list[num] = Passenger(id=num, origin=stop.id, arr_time=t)   # 为self.pax_list字典中添加一个乘客，乘客的序号为num
                    self.pax_list[num].took_bus = bus.id
                    self.pax_list[num].route = bus.route_id
                    self.pax_list[num].dest= d      # 设置乘客的下车站点___________________________________________________设置了乘客的下车站点
                    self.busstop_list[stop.id].waiting_list.append(num)   # 更新在该站点等待乘客的列表，增加该乘客的id
                    num += 1
            pax_leave_stop = []
            # 在站点等待上车的乘客——到达站时间从小到达排列
            waitinglist = sorted(self.busstop_list[stop.id].waiting_list)[:]    # 对列表进行升序操作并复制后赋值

            for num in waitinglist:      # waitinglist是在该站点等待上车的乘客。self.pax_list是包含整个环境中所有的乘客
                # add logic to consider multiline impact (i.e. the passenger can not board bus this time can board the bus with same destination later?)
                if bus != None and self.pax_list[num].route == bus.route_id:
                    self.pax_list[num].miss += 1            # 乘客不能坐上公交车的次数+1
                # 乘客登上到达的车辆
                if bus != None and bus.capacity - len(bus.onboard_list) > 0 and self.pax_list[num].route == bus.route_id:     # 还未超过车辆的容纳能力
                    self.pax_list[num].onboard_time = self.simulation_step    # 设置乘客的上车时间——当车辆容量够时，在站点等待的乘客的上车时间是相同的——————————————————————————————————————————设置乘客上车时间
                    bus.onboard_list.append(num)    # 更新在车乘客的列表
                    board_cost += bus.board_period   # 上车时间每人3秒
                    pax_leave_stop.append(num)     # 更新从站点离开的乘客列表
            # 将登上车离开站点的乘客移除站点的等车列表
            for num in pax_leave_stop:
                self.busstop_list[stop.id].waiting_list.remove(num)    # 更新在站点等待的乘客列表

        return alight_cost,board_cost     # 返回该站点下车乘客花费的总时间、上车乘客花费的总时间

    def sim(self):
        # update bus state

        ## dispatch bus  更新已经发车的车辆列表，包括增加新发车的车辆以及移除完成行程的车辆
        for bus_id, bus in self.bus_list.items():
            # 这个if用来让公交发车，设置公交车的当前速度，并将该公交车添加进已经发车的公交列表中
            if bus.is_dispatch==0 and bus.dispatch_time<=self.simulation_step:   # self.simulation_step:24149  is_dispatch=0表示公交车还未发出
                bus.is_dispatch=1    # 当公交车的发车时间小于等于仿真时间步，设置公交车的属性is_dispatch=1，表示发车了
                if bus.is_virtual!=1:
                    bus.current_speed = bus.speed * np.random.randint(60., 120.) / 100.  # 随机数不包括120  #    bus.speed=0.005555555555555556km/s
                else:
                    bus.current_speed = bus.speed*0.8
                self.dispatch_buslist[bus_id]=bus    # 用来存储已经发车的公交列表？

            # 这个if条件是用来判断公交车剩余未经过的站点是否小于等于0，就是是否运行完了一个行程
            if bus.is_dispatch==1 and len(self.dispatch_buslist[bus_id].left_stop)<=0:   # 初始时，self.dispatch_buslist列表是空的，依赖于上面的if来填充列表
                bus.is_dispatch = -1   # 表示公交车的一个行程已经运行完了
                self.dispatch_buslist.pop(bus_id,None)  # 从 self.dispatch_buslist 中移除指定的 bus_id 对应的元素，并返回该元素的值。如果 bus_id 不存在于 self.dispatch_buslist 中，则返回 None，这儿没有接收返回值，但是在字典中删除了对应元素

        # 对于发车的公交，如果它后面的公交和前面的公交的行程已经结束，那么该公交车没有前、后车辆
        for bus_id,bus in self.dispatch_buslist.items():    # is_dispatch=0 未发车，=1 发车， =-1车辆的一个行程结束
            if bus.backward_bus!=None and self.bus_list[bus.backward_bus].is_dispatch==-1:     # 车辆列表中  当前  发出的最后一辆车（不是最后一辆车）
                bus.backward_bus=None
            if bus.forward_bus!=None and self.bus_list[bus.forward_bus].is_dispatch==-1:     # 当前车辆的（不是第一辆车）  前一辆车  已经运行完一个行程
                bus.forward_bus=None


        ## bus dynamic
        for bus_id, bus in self.dispatch_buslist.items():    # 遍历每一个已经发车的车辆
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
                # 记录公交站的属性：车辆到达站点的时间
                if bus.route_id in self.busstop_list[curr_stop.id].arr_log:    # arr_log是公交站的属性，记录每个公交线路上，公交车到达该站的时间
                    self.busstop_list[curr_stop.id].arr_log[bus.route_id].append(self.simulation_step)        #([bus.id, self.simulation_step])
                else:   # 为公交站的属性arr_log添加元素，表示某条公交线路上某个公交到达该站点的时间
                    self.busstop_list[curr_stop.id].arr_log[bus.route_id] =[self.simulation_step]# [[bus.id, self.simulation_step]]
                # 公交到站时，计算在该站点，服务乘客上下车的时间
                board_cost,alight_cost = self.serve(bus,curr_stop)   # 公交到站时，上下客共需要花费多长时间
                # 更新公交车辆的状态
                bus.arr=1     # 表示公交车到站了    当公交离开站点时，arr=0
                # 更新公交车的服务时间，在上车时间和下车时间中取最大值。 ？？为什么+1？？
                bus.serve_remain = max(board_cost,alight_cost)+1.    # 表示公交车在站点的服务时间（但为什么要加1）

                bus.stay[curr_stop.id] = 1     # 记录公交车在相应站点停车，停了为1
                # 公交在站点服务乘客花费的时间
                bus.cost[curr_stop.id] = bus.serve_remain    # 记录公交在每个站点的服务时间
                bus.pass_stop.append(curr_stop.id)   # 公交车已经经过的站点列表+1
                bus.left_stop = bus.left_stop[1:]    # 公交车未经过的站点列表-1

                # 判断是否进行驻站控制（条件：1、公交到达站点；2、到达的站点不是公交第一个到达的站点，也就是公交车不是刚从第一站发车；3、车辆不是线路上第一辆车
                ## if determine holding once arriving            # 公交到站了，且该站不是该车辆第一个到达的公交站，且该公交不是该线路上第一个发出的车辆（公交车到达的第一个站点和线路上的第一辆公交车不进行驻站控制）
                if self.hold_once_arr==1 and len(bus.pass_stop)>1 and self.dispatch_times[bus.route_id].index(bus.dispatch_time)>0 :#and len(self.dispatch_buslist)>2 and len(bus.pass_stop)>2 and len(bus.left_stop)>1 and bus.forward_bus!=None:
                    # 记录每个时间步公交到站的情况，包含{当前站点，车辆id,在车乘客数量}
                    if self.simulation_step in self.arrivals:    # self.arrivals是一个字典，记录公交到站时的时间步对应的站点id,公交id，以及在车乘客数量
                        self.arrivals[self.simulation_step].append([curr_stop.id, bus_id, len(bus.onboard_list)])
                    else:
                        self.arrivals[self.simulation_step] = [[curr_stop.id, bus_id, len(bus.onboard_list)]]   # 用于存储公交驻站控制的时间步对应的站点id，公交id，公交车上的乘客数量
                    # 公交驻站时间  秒
                    bus.hold_remain = self.control(bus, curr_stop,type=self.control_type)   # 返回的应该是公交的驻站时间   self.control_type=2强化学习控制，0是无控制

                    # 记录公交到达过该站点
                    if bus.hold_remain > 0:
                        bus.stay[curr_stop.id] = 1
                    # 公交的驻站时间小于10min时需略驻站控制
                    if bus.hold_remain<10:    # 控制时间小于10则忽略？
                        bus.hold_remain = 0

                    # 记录公家在每个站点驻站时长
                    bus.hold_cost[curr_stop.id] = bus.hold_remain    # 用来记录公交在每个站点的驻站控制的时长
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
                    if bus.forward_bus in self.dispatch_buslist and bus.speed+bus.loc[-1]>=self.dispatch_buslist[bus.forward_bus].loc[-1]:
                        bus.stop()  # 也就是将要发生超车时，让公交车停车，禁止超车
                        bus.current_speed = bus.speed * np.random.randint(60, 120) / 100.
                        if bus.b==0:    # 初始时bus.b=0
                            self.bunching_times+=1
                            bus.b=1
                    else:
                        bus.b = 0
                        bus.dep(bus.current_speed)  # 公交开始按照给定的速度运动，更新公交车的实时位置
                        for p in bus.onboard_list:
                            self.pax_list[p].onroad_cost+=1   # 车上乘客的在车时间成本+1

                        # 记录每个站点公家离开的时间
                        if len(bus.pass_stop)>0:
                            if bus.route_id in self.busstop_list:   # 这儿永远都不可能满足要求呀
                                self.busstop_list[bus.pass_stop[-1]].dep_log[bus.route_id].append([bus.id, self.simulation_step])
                            else: # 记录每辆公交车在每个站点的发车时间
                                self.busstop_list[bus.pass_stop[-1]].dep_log[bus.route_id] = [[bus.id, self.simulation_step]]

        self.simulation_step+=1   # 仿真时间步+1
        Flag =False
        for bus_id, bus in self.bus_list.items():
            if bus.is_dispatch!=-1:       # 公交的行程结束时为-1，否则为0或1（发车）
                Flag = True               # 当公交列表中全部车辆的行程都结束，Flag的值才会未Flase,否则Flag为True
        return Flag
    # 确定控制方法种类
    def control(self,bus,bus_stop,type=0):
        # 无驻站控制
        if type==0:
            return 0
        if type==1:
            fh, bh = self.cal_headway(bus)
            if bus.forward_bus==None:
                return 0
            else:
                return max(0, 58 + 0.05 * (abs(bus.dispatch_time-self.bus_list[bus.forward_bus].dispatch_time) - fh))#max(0, 58 + 0.05 * ( (self.mfh - fh)))#
        # 强化学习进行驻站控制
        if type==2:   # 类型2应该指的是使用强化学习进行驻站控制
            return self.rl_control(bus,bus_stop)     # 返回的是驻站时间

        return 0
    # 强化学习进行驻站控制
    def rl_control(self, bus, bus_stop):
        # retrieve historical state
        current_interval = self.simulation_step
        state = []   # 第一项是公交车在车乘客占公交容量的百分比
        for record in self.arrivals[current_interval]:   # self.arrivals记录的是相应的时间步下，公交到站的情况；公交站点、公交车、以及在车乘客数量的信息列表（在公交车到站时，将相应信息记录到改字典中）
            # record是一条记录，是一个列表，分别是：公交到达的站点，车辆id,在车人数
            bus_stop_id_ = record[0]
            bus_id_ = record[1]
            onboard = record[2]
            if bus_id_ == bus.id:
                state = [onboard / bus.capacity]    # 获取车上乘客的占比
                break   # 退除for循环
        # 获取公交前后向车头时距
        fh, bh = self.cal_headway(bus)         # 返回前向车头时距和后向车头时距
        var, mean = self.route_info(bus)       # 返回了后向车头时距的标方差和均值
        # 将前向车头时距和后向车头时距添加仅车辆的状态列表中
        state += [min(fh / 600., 2.), min(bh / 600., 2.)]    # 为什么要这样处理？    600s=10min    车头时距的大小不大于2min?

        self.state_record.append(state)
        if self.share_scale == 0:
            action = np.array(self.agents[bus.id].choose_action(np.array(state).reshape(-1, )))
        # 根据智能体的状态输出智能体的动作
        if self.share_scale == 1:    # 默认设置值为1，应该表示参数是共享的
            action = np.array(self.agents[bus.route_id].choose_action(np.array(state).reshape(-1, )))
        # mark记录了状态动作对
        mark = list(np.array(state + list(action)).reshape(-1, ))    # 状态+动作
        self.bus_list[bus.id].his[self.simulation_step] = mark  # his是用来存储对应时间步下的状态和动作的字典

        if len(self.GM.temp_memory[bus.id]['a']) > 0:    # temp_memory是一个字典，里面存储了s,a,fp,r
            # organize fingerprint: consider impact of other agent between two consecutive control of the ego agent
            # 考虑自我智能体在两次连续驻站控制之间其他代理的影响
            stop_dist = [0.]
            bus_dist = [0.]
            # 自身的贡献  用0补齐， ？[0.] + [bus.id]是什么？
            fp = [self.GM.temp_memory[bus.id]['s'][-1] +self.GM.temp_memory[bus.id]['a'][-1].tolist()+ stop_dist + bus_dist + [0.] + [bus.id]]    # stop_dist + bus_dist + [0.]这儿的0是为了确保相同的特征长度
            temp = bus.last_vist_interval         # 该公交最新一次到站的时间
            # 下面就是为了判断，在公交车连续两次到站的时间间隔内，其他到站的公交
            # 即公交两次到站的时间间隔内
            while temp <= current_interval:   # 为了寻找这个在这个公交上一次驻站控制到这一次进行驻站控制期间，其他进行驻站控制的公交
                if temp in self.arrivals:       #  self.arrivals记录相应时间步，进行驻站控制的车辆、公交站、公交的在车人数
                    for record in self.arrivals[temp]:
                        bus_stop_id_ = record[0]
                        bus_id_ = record[1]
                        onboard = record[2]
                        if bus_id_ == bus.id:  # 需要考虑的是其他智能体，当是自身智能体时，推出当前循环，进行下一次循环
                            continue
                        # 只关心两次驻站期间，与之相邻的前后公交车的驻站情况
                        if (bus_id_ == bus.forward_bus or bus_id_ == bus.backward_bus) or (self.all == 1):   # self.all=0,所以前面的条件必须满足其一
                            curr_bus = self.dispatch_times[bus.route_id].index(bus.dispatch_time)   # 获取当前车辆发车时间在线路发车时间列表中的索引
                            neigh_bus = self.dispatch_times[bus.route_id].index(self.bus_list[bus_id_].dispatch_time)  # 获取相邻车辆发车时间的索引
                            # 辆车之间的距离，索引差/车辆数
                            bus_dist = [(curr_bus - neigh_bus) / len(self.bus_list)]  # 发车时间索引之差/公交数量(两控制点之间公交车的数量)
                            stop_dist = [(bus.stop_list.index(bus.pass_stop[-2]) - bus.stop_list.index(bus_stop_id_)) / len(self.busstop_list)]   # 当前公交上一次到站的站点索引与相邻公交主站控制的索引之差/公交站数量(两控制点之间公交站的数量)
                            fp.append(self.bus_list[bus_id_].his[temp] + stop_dist + bus_dist + [abs(temp - current_interval)] + [bus_id_])  # 边际贡献：相邻公交站驻站是的状态动作+bus_dist+stop_dist+当前时间步与该公交站驻站时间步之差+相邻公交的索引

                temp += 1

            reward1 = (-var / mean / mean) * (1 - self.weight) * 5     # 5是什么？    # 论文公式二的第一项（多乘了5）
            reward2 = (-abs(self.GM.temp_memory[bus.id]['a'][-1])) * self.weight    # 论文公式二的第二项
            reward = reward1 + reward2     # 论文的公式二

            self.reward_record.append(reward)
            self.reward_signal[bus.id].append(reward)
            self.reward_signalp1[bus.id].append(reward1)
            self.reward_signalp2[bus.id].append(reward2)

            self.GM.temp_memory[bus.id]['r'].append(reward)
            self.GM.temp_memory[bus.id]['fp'].append(fp)
        # 将公交的状态动作存储在经验回放池中
        ## update temporal memory with current state and action and mark   公交车在当前站点执行动作的奖励是当公交到达下一个站点时才获得
        self.GM.temp_memory[bus.id]['s'].append(state)
        self.GM.temp_memory[bus.id]['a'].append(action)

        # 存储到经验回放池
        if len(self.GM.temp_memory[bus.id]['s']) > 2:  # 此时，fp和r的长度还为大于2，比s的长度小1
            s = self.GM.temp_memory[bus.id]['s'][-3]   # 相当于智能体的初始状态   在[-1]的状态下，[-3]状态下采取动作得到了奖励
            ns = self.GM.temp_memory[bus.id]['s'][-2]  # 智能体初始状态执行完动作之后进入的下一状态
            fp = self.GM.temp_memory[bus.id]['fp'][-2]  # 智能体上一状态的情况
            nfp = self.GM.temp_memory[bus.id]['fp'][-1] # 智能体进入下一状态情况
            a = self.GM.temp_memory[bus.id]['a'][-3]   # 智能体初始状态执行的动作
            r = self.GM.temp_memory[bus.id]['r'][-2]   # 智能体初始状态执行完动作之后获得的奖励
            self.GM.remember(s, fp, a, r, ns, nfp, bus.id)   # 经验回放池
        self.action_record.append(action)
        # 将动作裁剪到[0,3]之间
        action = np.clip(abs(action), 0., 3.)
        self.bus_list[bus.id].last_vist_interval = self.simulation_step        # 更新该公交的上一个驻站控制时间为当前时间步
        return 180. * action   # 最终返回的是驻站时间

    def cal_headway(self,bus):

        if bus.forward_bus!=None and bus.forward_bus in self.dispatch_buslist : # 该公交车前面有车，且该车在已经发车的列表中
            fh = abs(bus.loc[-1]-self.bus_list[bus.forward_bus].loc[-1])/bus.c_speed       # bus.c_speed是什么速度——大小是20km/h

        else:
            fh = abs(bus.loc[-1] - 0.) / bus.c_speed

        if bus.backward_bus!=None and bus.backward_bus in self.dispatch_buslist:
            bh = abs(bus.loc[-1]-self.bus_list[bus.backward_bus].loc[-1])/bus.c_speed

        else:
            bh = abs(bus.loc[-1]-0.)/bus.c_speed
            if self.dispatch_times[bus.route_id].index(bus.dispatch_time)==len(self.dispatch_times[bus.route_id])-1:   # 表示该车是该线路上最后一个发出的车辆
                bh=0.

        return fh, bh

    def route_info(self,bus):

        fh = [500 for _ in range(3)]   # fh = [500,500,500]
        bh = [500 for _ in range(3)]   # bh = [500,500,500]
        for bus_id, bus_ in self.dispatch_buslist.items():   # 遍历 所有 已经发出的公交车列表
            if bus_.route_id == bus.route_id:
                if bus_.forward_bus != None and bus_.forward_bus in self.dispatch_buslist:   # 公交前面有车，且公交前面的车的位于已经发出车辆的列表中
                    fh.append(abs(bus_.loc[-1] - self.bus_list[bus_.forward_bus].loc[-1]) / bus_.speed)  # 将公交的实际车头时距放入fh列表中
                if bus_.backward_bus != None and  bus_.backward_bus in self.dispatch_buslist:  # 公交后面有车，且公交后面的车的位于已经发出车辆的列表中
                    bh.append(abs(bus_.loc[-1] - self.bus_list[bus_.backward_bus].loc[-1]) / bus_.speed)

        if len(bh)<2:   # 不可能满足条件，长度至少也为3
            return 999999,999999

        return np.var(bh), np.mean(bh)      # 这儿其实是对已经发车的车辆列表中车辆的后向车头时距求均值和方差（忽略路线上最后一辆发出车辆的后向车头时距）

    def learn(self):
        ploss_set = []
        qloss_set = []

        # self.share_scale == 1
        if self.share_scale == 0:   # 1
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
        if self.share_scale == 1:
            for rid,r in self.route_list.items():
                b = np.random.randint(0,len(r.bus_list))   # 生成介于[0,59)之间的整数
                bus_id = r.bus_list[b]  # 获取相应的公交id
                while len(self.GM.memory[bus_id])<=0  :    # 在进行驻站控制的时候会存储数据，如果长度小于0，证明没有进行驻站控制，重新选择一个公交车
                    b = np.random.randint(0, len(r.bus_list))
                    bus_id = r.bus_list[b]

                ploss, qloss = self.agents[rid].learn(self.GM.memory[bus_id])
                try:
                    self.qloss[bus_id].append(np.mean(qloss))   # 当self.qloss字典中没有相应的[bus_id]对应的键，则执行except下的代码
                except:
                    self.qloss[bus_id] = [np.mean(qloss)]
                ploss_set.append(ploss)
                qloss_set.append(qloss)


        if len(ploss_set) > 0 and len(self.reward_signal) > 0:
            return np.mean(ploss_set), np.mean(qloss_set),True
        else:
            return _, _, False


