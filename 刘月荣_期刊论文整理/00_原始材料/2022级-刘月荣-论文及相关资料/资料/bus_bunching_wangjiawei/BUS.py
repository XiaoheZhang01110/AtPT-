import numpy as np

class bus():
    '''
    id:                bus id
    is_serving:        whehter bus is serving a stop 1:stop for service 0:on route -1:holding -2: not emit
    omega:             angle speed
    capacity:          bus capacity
    car_size:          car size for visualization
    track_radius:      radius of bus corridor   # 公交走廊的半径
    color:             car color for visualization
    step:              simulation step increase with 1
    slack time:        time cost at each station
    emit_time:         time for departure
    is_emit:           emit flag     # 出发标志
    serve_list:        record served  passgenger number for each stop of a bus through the simulation  # 通过仿真记录公交车每个站点的服务乘客编号
    alight_list:       record numbers of passenger to alight in each stop in real time    # 实时记录每个站点下车的乘客人数
    serve_level:       bus serve level
    dispatch_loc:      dispatch location   # 调度位置
    loc:               bus current loc (reset when arriving at origin station in favor of visualization)
    travel_sum:        overall distance the bus has traveled   公交行驶的总距离
    arrival_schedule:  schedule arrval time on a stop  计划到达公交站点的时间
    arrival_bias:      bias between schedule arrival time and actual arrival time  计划到达时间与实际到达时间的偏差
    alight_rate:       alighting rate   下车率
    onboard_list:      record number of passenger on board boarding from each stop in real time  实时记录从没给站点上车的乘客人数
    slack_time:         slack time to force bus keep pace with the schedule <= hold time
    slack_time_sum:     sum of slack time
    hold_time:         holding time to force bus keep pace with the schedule
    hold_time_sum:     sum of holding time
    trajectory:        record bus trajectory
    hold_stop:         record at which station the bus is holding
    state:             state observation from the perspective of bus
    action:            control for each stop
    reward:            minimize mean of headway and variance of headway
    is_close           the bus is closed or not. 0:open 1:close
    ass_dispatch_loc:  assist locate the bus in trajectory construction
    hold_action:       record the where and when the holding function
    is_serving_rl:     flag for RL train sample collect
    special_state:     0 for normal ,1 for catching its leading bus
    serve_stop:        record which stop the bus is serving
    trip_record        record travel point for each od-pair trip
    trip_cost        record travel cost for each od-pair trip
    '''
    def __init__(self,w,capacity,radius,car_size=6,state_dim=6, action_dim=1,alight_rate=0.55  ,color_='blue',dispatch_loc=0,id=1,stop_nums=1,emit_time=0):
        self.id = id
        self.is_serving = 0  #  公交车是否服务于一个站点:stop for service 0:on route -1:holding -2: not emit
        self.omega = w    # 角速度
        self.capacity =capacity
        self.car_size = car_size
        self.track_radius = radius
        self.color = color_
        self.step = 0   # 仿真时间步按1增加
        self.slack_time = 0    # ？
        self.slack_time_sum = 0
        self.hold_time = 0
        self.hold_time_sum = 0
        self.emit_time = emit_time  # 出发时间
        self.is_emit = False
        self.serve_list = [0 for i in range(stop_nums)] # 通过仿真记录公交车每个站点的服务乘客编号
        self.alight_list = [0 for i in range(stop_nums)] # 实时记录每个站点下车的乘客人数
        self.serve_level = []  # 公交服务水平
        self.dispatch_loc = dispatch_loc # starting station
        self.loc = dispatch_loc   # 当前位置
        self.hold_stop = None   # 驻站控制的站点
        self.hold_stop_temp=-2
        self.travel_sum = 0   # 公交已经行驶的总距离
        self.arrival_schedule = 0
        self.alight_rate = alight_rate
        self.arrival_bias = []
        self.onboard_list = [0 for i in range(stop_nums)]
        self.trajectory = []
        self.state = []   # 公交视角下的状态观测
        self.action = [0 for i in range(action_dim)]
        self.reward = 0
        self.is_close = 0  # 公交车是否关闭
        self.loc_set=[]
        self.ass_dispatch_loc=dispatch_loc  # 帮助子轨迹中定位公交的位置
        self.hold_action=[]  # 记录控制动作的时间和地点
        self.hold_action_w = []
        self.is_serving_rl = 0  # 使用强化学习进行训练样本采样的标签
        self.special_state = 0
        self.serve_stop = None
        self.stop_visit=[0 for i in range(12)]
        self.hold_time_list=[0 for i in range(stop_nums)]
        cx = self.track_radius*np.cos(dispatch_loc)  #？
        cy = self.track_radius*np.sin(dispatch_loc)
        self.trip_record = [[[] for j in range(6) ] for i in range(12)] # 12 origin*6interval
        self.trip_cost = [[[] for j in range(6)  ] for i in range(12)]

    def move(self,):
        self.isserving = 0
        self.step += 1   # 调用一次move,时间步加1
        self.trajectory.append(self.travel_sum+self.dispatch_loc)
        self.loc_set.append(self.loc)
        self.loc = self.ass_dispatch_loc+self.omega*self.step # add random noise to bus speed
        self.travel_sum+=self.omega
        self.hold_action.append(0.)   # 控制动作的地点
        self.hold_action_w.append( 0.)   # 控制动作的时间
        if self.loc>=np.pi*2: # when vehicle arrive at the stop 1, loc returns to 0   # 相当于公交车回到了起点（第一站），开始了循环
            self.loc=0
            self.ass_dispatch_loc=0
            self.step=0
            self.pass_stop = []

    def stop(self):
        self.trajectory.append(self.travel_sum+self.dispatch_loc)

        self.isserving = 1
        self.slack_time_sum += 1.
        if self.is_serving==-1 and self.hold_stop!=self.hold_stop_temp and self.hold_time>30: # only calculate hold time if the bus is first held
            self.hold_action.append(self.loc)          # 控制动作的地点
            self.hold_action_w.append(self.hold_time)   # 控制动作的时间
            # print(self.hold_time)
            self.hold_time_sum += self.hold_time # 0.01=update step
            # print(self.hold_time)
            self.hold_stop_temp = self.hold_stop
            self.hold_time_list[self.hold_stop]+=self.hold_time  # 记录每个站点驻站控制的时间列表
            self.is_serving_rl = -1   # 使用强化学习进行训练样本采样的标签
        else:
            if len(self.loc_set)>len(self.hold_action):  # 公交位置列表的长度 > 大于控制动作的长度
                self.hold_action.append(0.)
                self.hold_action_w.append(0)
        self.loc_set.append(self.loc)
        if self.loc>=np.pi*2:
            self.loc=0
            self.ass_dispatch_loc=0
            self.step = 0
            self.pass_stop = []


