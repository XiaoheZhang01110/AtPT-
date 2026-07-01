import numpy as np
from collections import deque
import random

class Memory():
    def __init__(self,members):
        self.memory = {}

        # temp memory to store last-step state and action because of no immediate feedback
        self.temp_memory = {}      # 用于存储上一个时间步的状态和动作，由于没有立即的反馈
        self.experience = 0
        for m in members:
            self.memory[m] = deque(maxlen=2000)   # 双向队列，队列最大的长度为2000
            self.temp_memory[m]={'s':[],'a':[],'a_log_prob':[],'fp':[],'r':[] }



    def remember(self, state,fp, action, reward, next_state,next_fp,member_id):
        self.experience+=1
        self.memory[member_id].append((state,fp, action, reward, next_state,next_fp))