import xml.etree.ElementTree as ET
import numpy as np
from random import seed
from random import gauss,randint

class Bus_stop():
    def __init__(self,id,lat,lon  ):
        '''

        :param id: bus stop unique id
        :param lat:  bus stop latitude in real-world
        :param lon:  bus stop longitude in real-world
        :param routes: bus stop serving routes set
        :param waiting_list:  waitting passenger list in this stop
        :param dyna_arr_rate:  dynamic passenger arrival rate for this stop
        :param arr_bus_load:  record arrving bus load
        :param arr_log:  (dictionay) record bus arrival time with respect to each route (route id is key)
        :param uni_arr_log: (list) record bus arrival time
        :param dep_log: (dictionay) record bus departure time with respect to each route (route id is key)
        :param uni_dep_log: (list) record bus departure time

        '''

        self.id = id
        self.lat = lat
        self.lon = lon
        self.loc = 0.
        self.next_stop = None
        self.routes = []
        self.waiting_list=[]
        self.dyna_arr_rate = []   # 记录24小时内，每小时站点的乘客到达率（乘客数/s），每个列表的长度为24
        self.dyna_arr_rate_sp ={}
        self.arr_bus_load =[]     # 记录到达公交站的车辆上的人数？
        self.arr_log = {}         # 记录每条线路上公交车到站的时间，线路为键
        self.uni_arr_log = []
        self.dep_log = {}    # 记录线路上每个公交离开站点的时间
        self.uni_dep_log = []
        self.pax = {}
        self.dest = {}
        self.served = 0

    # return a list of passengers arrival time
    def pax_gen(self,bus,sim_step=0):

        pax = []
        base=0
        interval = 0
        self.arr_bus_load.append(len(bus.onboard_list))
        if bus!=None:
            if len(self.arr_log[bus.route_id])>1:
                interval = (self.arr_log[bus.route_id][-1] -self.arr_log[bus.route_id][-2] )
                sample = (np.random.poisson(self.rate*self.dyna_arr_rate[int(sim_step/3600)%24]+0.0001,int(interval)  ))
                base=self.arr_log[bus.route_id][-2]
                for i in range(sample.shape[0]):
                    if sample[i]>0:
                        pax+=[base+i for t in range(sample[i])]
            else:
                # assume passenger will gather in less than 15min before the first bus began.
                sample = (np.random.poisson(self.rate*self.dyna_arr_rate[int(sim_step/3600)%24]+0.0001,900 ))
                base = self.arr_log[bus.route_id][-1]
                for i in range(sample.shape[0]):
                    if sample[i]>0:
                        pax+=[base-i for t in range(sample[i])]
        else:
            for k,v in self.arr_log.items():
                interval = sim_step - v[-1]
                sample = (np.random.poisson(self.rate*self.dyna_arr_rate[int(sim_step/3600)%24],int(interval) ))
                base = v[-1][1]
                for i in range(sample.shape[0]):
                    if sample[i] > 0:
                        pax += [base + i for t in range(sample[i])]

        return pax

    def set_rate(self,r ):
        self.rate = r # pax/sec

    # 生成乘客采用此方法
    def pax_gen_od(self,bus,sim_step=0,dest_id=None):

        base=0
        interval = 0

        if bus!=None:
            if len(self.arr_log[bus.route_id])>1:   # 表示该站点并不是第一次有车辆到站，
                # 连续两辆车到达该公交站的时间间隔
                interval = (self.arr_log[bus.route_id][-1] -self.arr_log[bus.route_id][-2] )    # 同一路线上两辆车到达同一站点的时间间隔，也就是说乘客的聚集时间是该辆车到站时间与上一辆车到站时间的差值
                sample = (np.random.poisson(self.rate*(self.dest[dest_id][int(sim_step/3600)%24])+0.0001,int(interval)))   # np.random.poisson()函数的完整形式是np.random.poisson(lam, size=None)，其中lam是泊松分布的参数λ(单位时间内事件发生的平均次数)，size是生成随机数样本的大小
                base=self.arr_log[bus.route_id][-2]      # 公交站上一次公交车到站的时间步
                pax = []    # 记录的是乘客到达公交站的时间，长度为乘客数量
                for i in range(sample.shape[0]):
                    if sample[i]>0:
                        pax+=[base+i for t in range(sample[i])]
            else:   # 表示该站点第一次有车辆到站，乘客最早在车辆到达前15分钟聚集       # np.random.poisson(lam=2, size=(3, 4))  lam，表示泊松分布的均值或参数λ;size，表示输出随机数的形状
                # assume passenger will gather in less than 15min before the first bus began.    # sim_step代表仿真的时间步长
                sample = (np.random.poisson(self.rate*(self.dest[dest_id][int(sim_step/3600)%24])+0.0001,900 ))    # 生成长度为900的一维数据,900s=15min,表示乘客最早在车辆到达前15min中开始聚集

                base = self.arr_log[bus.route_id][-1]  # 该线路上最后一次到站的时间,也就是当前公交车的到站时间
                pax = []    # 列表中的元素是用来表示乘客生成的时间，列表长度表示乘客数量
                for i in range(sample.shape[0]):
                    if sample[i]>0:
                        pax+=[base-i for t in range(sample[i])]   # pax列表的长度，等于sample中元素的和
        else:
            for k,v in self.arr_log.items():    # 应该是没有车辆到站时
                interval = sim_step - v[-1]
                sample = (np.random.poisson(self.rate*self.dest[dest_id][int(sim_step/3600)%24],int(interval) ))

                base = v[-1][1]
                for i in range(sample.shape[0]):
                    if sample[i] > 0:
                        pax += [base + i for t in range(sample[i])]

        return pax


