import numpy as np
from collections import deque
import random


class Memory():
    def __init__(self,members):

        # temp memory to store last-step state and action because of no immediate feedback
        self.temp_memory = {}      # 用于存储上一个时间步的状态和动作，由于没有立即的反馈
        self.experience = 0
        for m in members:
            self.temp_memory[m]={'s':[],'a':[],'a_log_prob':[],'fp':[],'r':[]}

c = [1,2,3,4,5,6,7,8,9,10]
A = Memory(c)
# A.temp_memory[]['s'].append([1,2,4])
# A.temp_memory['a'].append([11])
for i in c:
    A.temp_memory[i]['s'].append([1,2,3,4,5])
    A.temp_memory[i]['a'].append(i)
    A.temp_memory[i]['a_log_prob'].append(i)
    A.temp_memory[i]['fp'].append(i)
    A.temp_memory[i]['r'].append(i)

print(A.temp_memory)
print('*****************************')
'''
import time
start_time1 = time.time()  # 记录开始时间
generator = ((key, A.temp_memory[key]) for key in A.temp_memory)
for key, value in generator:
    # 处理键值对
    print(value)
end_time1 = time.time()
print("代码执行时间1:",end_time1 - start_time1)

start_time2 = time.time()  # 记录开始时间
for key in A.temp_memory:
    print(A.temp_memory[key])
end_time2 = time.time()
print("代码执行时间2:",end_time2 - start_time2)
'''

import threading

def calculate_sum(sub_dict, result_list):
    values = sub_dict["key1"]
    result = sum(values)
    result_list.append(result)

my_dict = {
    "sub_dict1": {
        "key1": [1.2, 2.3, 3.4],
        "key2": [4.5, 5.6, 6.7],
        "key3": [7.8, 8.9, 9.0]
    },
    "sub_dict2": {
        "key1": [0.1, 1.1, 2.2],
        "key2": [3.3, 4.4, 5.5],
        "key3": [6.6, 7.7, 8.8]
    },
    "sub_dict3": {
        "key1": [9.9, 10.0, 11.1],
        "key2": [12.2, 13.3, 14.4],
        "key3": [15.5, 16.6, 17.7]
    }
}
#
# result_list = []
# threads = []
#
# # 创建线程并启动
# for sub_dict in my_dict.values():
#     thread = threading.Thread(target=calculate_sum, args=(sub_dict, result_list))
#     thread.start()
#     threads.append(thread)
#
# # 等待所有线程执行完毕
# for thread in threads:
#     thread.join()
#
# # 打印结果列表
# print(result_list)

import threading
from concurrent.futures import ThreadPoolExecutor

def process_sub_dictionary(sub_dict):
    # 在这里处理子字典的操作
    for key, value in sub_dict.items():
        # 处理子字典的键值对
        print(f"Key: {key}, Value: {value}")

def process_dictionary(dictionary):
    with ThreadPoolExecutor(max_workers=16) as executor:
        for sub_dict in dictionary.values():
            executor.submit(process_sub_dictionary, sub_dict)

# 要遍历的包含59个子字典的字典
my_dictionary = {
    'sub_dict1': {'key1': 'value1'},
    'sub_dict2': {'key2': 'value2'},
    # ... 其他子字典
    'sub_dict59': {'key59': 'value59'}
}

# 处理字典的操作
process_dictionary(my_dictionary)


import pandas as pd
import matplotlib.pyplot as plt

# 创建一个示例 DataFrame
data = {'A': [10, 20, 30, 40, 50],
        'B': [15, 25, 35, 45, 55]}
df = pd.DataFrame(data)

# 绘制柱形图
df.plot(kind='bar')
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('Bar Chart from DataFrame')
plt.show()