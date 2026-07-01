import xml.etree.ElementTree as ET
import numpy as np

class Passenger():
    def __init__(self,id,origin,plan_board_time=None,arr_time=None):
        '''
        :param id: passegner unique id
        :param origin:  stop id where the passenger begins her trip
        :param plan_board_time:  passenge onboard time in reality if real-world data used
        :param arr_time:  passenger arrival time
        :param onboard_time: passenger onboard time in simulation
        :param alight_time: passenger alight time in simulation
        :param dest: stop id where the passenger finishes her trip
        :param realcost: passenger travel cost for her trip
        :param onroad_cost: passenger travel cost on road (after boarding)
        :param trip: passenger predefined travelling stop if real-world data used
        :param trip_left: stop left to go if real-world data used
        :param took_bus: : bus id where passenger is current taking
        :param miss: : record how many times the passenger can not take the arriving bus
        '''
        self.id = id
        self.origin = origin      # 初始的所在的公交站的id
        self.arr_time = arr_time
        self.onboard_time = 999999999
        self.alight_time = 999999999
        self.plan_board_time = plan_board_time    # 计划上车时间
        self.dest = None          # 完成行程时所在的站点
        self.realcost = 0     # 乘客行程的旅行费用
        self.trip = []
        self.trip_left = self.trip[:]

        self.took_bus = ''
        self.state = 0 # 0-waitting 1-on bus 2 waitting for transfer
        self.onroad_cost=0
        self.miss=0           # 记录乘客不能上车的次数