import xml.etree.ElementTree as ET
import numpy as np

class Bus():
    def __init__(self,id,route_id,stop_list,dispatch_time,speed_type,block_id,dir, round=1 ):
        '''
        :param id: bus unique id
        :param route_id: route id bus serving
        :param stop_dist: stop location list bus travelling
        :param stop_list: stop list bus travelling
        :param dispatch_time: the time when bus begin its trip
        :param schedule: planned departure time at each stop
        :param pass_stop: the stops bus has passed
        :param left_stop: the stops left to go
        :param time_step: record the simulation time step when this bus is alive
        :param trajectory: record the travel location
        :param onboard_list: passenger id list onboard
        :param forward_bus: bus id of forward bus
        :param backward_bus: bus id of backward bus
        :param current_stop_duration: how long the bus has been stopping at current stop
        :param speed: bus cruising speed
        :param arr: arr= 0: on road ; arr=1: on arr
        :param board_period:  how long for a passenger board
        :param alight_period: how long for a passenger alight
        :param capacity: the maximum number of passengers allowing in this bus
        :param his: (dictionary) record the bus activity for RL {simulation time step: [state,action]}
        :param is_dispatch: is_dispatch=0: bus has not dispatched
        '''

        self.id = id
        self.route_id = route_id       # 公交运行的线路id
        self.schedule = {}             # 每个站点的计划出发时间
        self.dispatch_time = dispatch_time    # 公交开始它的行程的时间
        self.trajectory = [0]          # 记录公交的运行位置
        self.stop_dist = {}        # 距离每个公交站的距离

        self.stop_list = stop_list
        self.pass_stop = []             # 公交已经经过的站点
        self.left_stop = stop_list      # 公交还未经过的站点
        self.time_step=[dispatch_time]
        self.onboard_list = []          # 在车上的乘客id列表
        self.alight_period = 1.8        # 乘客下车时间
        self.board_period = 3           # 乘客上车时间

        self.arr = 0                    # arr= 0: on road ; arr=1: on arr
        self.capacity = 80

        self.speed = 11 # km/s
        self.c_speed = 11
        self.current_stop_duration = 0    # 公交已经在当前站停靠的时间
        self.current_speed = 0.

        self.forward_bus = None      # 位于前面的公交的id
        self.backward_bus = None
        self.b = 0
        self.his={}         # 用于存储对应时间步下的状态和动作
        self.last_vist_interval = -1
        self.stops_record = [-1]

        '''增加一个公交的到站时间'''
        self.arr_time = None
        self.arr_time_list = []
        '''增加一个存储公交到站后还未进行上车车过程的在车乘客列表'''
        self.previous_onboard_list = []
        '''增加一个公交运行的总路程'''
        self.travel_sum = 0
        self.speed_type = speed_type
        '''新增'''
        self.leave_stop_time = None
        self.serve_over_time = []

    def set(self):
        self.serve_remain = 0
        self.hold_remain = 0
        self.currentstop =None
        self.is_hold = 0
        self.is_finish = 0
        self.is_dispatch = 0
        self.cost = {}
        self.hold_cost = {}
        self.stay ={}
        self.pax_boarded = {}
        self.trip_info = []

        for stop in self.stop_list:
            self.cost[stop] =  0
            self.stay[stop] = 0
            self.hold_cost[stop] = []
            self.pax_boarded[stop] = 0

        self.loc = [0]
        self.occp = [0]   # 公交车的占用率
        self.dwell=0

        self.hold_info = {}

        self.is_virtual = 0

    def move(self,d ):
        self.occp.append(len(self.onboard_list)/self.capacity)
        # 判断公交车位置是否大于2Π
        self.travel_sum += d
        if d+self.loc[-1] >= 2 * np.pi:
            self.loc.append(0)
        else:
            self.loc.append(d+self.loc[-1])
        self.time_step.append(self.time_step[-1]+1)
        self.stops_record.append(-1)
    def stop(self,):
        self.occp.append(len(self.onboard_list) / self.capacity)   # 更新公交的占用率
        self.loc.append(self.loc[-1])          # 记录公交车的实时位置
        self.time_step.append(self.time_step[-1]+1)
        self.stops_record.append(self.pass_stop[-1])
    def dep(self, d=0):      # d是公交车的当前速度
        try:
            self.move(d )
            self.arr = 0
            self.is_hold=0

        except:
            print(self.route_id)

    #  return a list of passengers alight
    def pax_alight(self,alight_rate=0.3):
        pax = []
        # 表示车上的所有乘客都要下车
        for i in range(len(self.onboard_list)):
            if np.random.binomial(1, alight_rate)>0:    # 生成服从二项分布的随机数,生成的二项分布的随机数都是大于0的
                pax.append(self.onboard_list[i])

        return pax

    def pax_alight_fix(self,stop,pax_list):    # 用来获取要下车的乘客列表
        pax = []

        for i in range(len(self.onboard_list)):

            if pax_list[self.onboard_list[i]].dest==stop.id:   # 当公交到站时，如果车上乘客的目的地是当前公交站
                pax.append(self.onboard_list[i])
        return pax



