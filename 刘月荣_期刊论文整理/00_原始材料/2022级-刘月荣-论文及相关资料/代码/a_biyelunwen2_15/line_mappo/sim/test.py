import matplotlib.pyplot as plt
import numpy as np

# 定义列表数据
list1 = [0.1 / 60, 0.5 / 60, 0.1 / 60, 0.6/ 60, 0.4 / 60, 0.1 / 60, 0.5 / 60, 0.7 / 60, 0.2 / 60, 0.1 / 60,
             0.4 / 60, 0.3 / 60, 0.2 / 60, 1.0 / 60, 1.5 / 60, 0.8 / 60, 1.4 / 60, 2.0 / 60, 1.7 / 60, 1.3 / 60,
             2.1 / 60, 1.7 / 60, 1.1 / 60, 1.5 / 60, 1.7 / 60, 1.5 / 60, 0.5 / 60, 0.8 / 60, 0.2 / 60, 0.3 / 60,
             0.6 / 60, 0.5 / 60, 0.3 / 60, 0.2 / 60, 0.1 / 60, 1.0 / 60, 0.3 / 60, 0.2 / 60, 0.0]

# 设置柱子的位置
x = np.arange(len(list1))

# 设置柱子的宽度
width = 0.8

# 创建画布和子图，设置大小
fig, ax = plt.subplots(figsize=(12, 6))

# 定义颜色列表
colors = ['skyblue' if val < 0.02 else 'lightcoral' for val in list1]

# 绘制柱形图，添加颜色
rects = ax.bar(x, list1, width, color=colors)

# 添加数据标签，调整字体大小
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.3f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10)

autolabel(rects)

# 添加标签和标题，调整字体大小
ax.set_ylabel('Values', fontsize=12)
ax.set_title('Bar chart of list1', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels([str(i + 1) for i in range(len(list1))], fontsize=10)

# 调整刻度范围，让图表更紧凑
ax.set_ylim(0, max(list1) * 1.1)

# 调整刻度标签的字体大小
ax.tick_params(axis='both', which='major', labelsize=10)

# 去除顶部和右侧边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 显示图表
plt.show()