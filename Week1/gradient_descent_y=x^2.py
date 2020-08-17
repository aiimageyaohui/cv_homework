import numpy as np
import matplotlib.pyplot as plt
import random
plot_x = np.linspace(-10,10,200)
plot_y = plot_x**2

plt.plot(plot_x,plot_y)
plt.show()

x_min = random.randrange(-10,10)
print(f"优化起点{x_min}")

def dJ(x):
    return 2*x

def J(x):
    try:
        return x**2
    except:
        return float('inf')

learning_rate = 0.1
error = 1e-8
history_x = [x_min]
while True:
    gradient = dJ(x_min)
    last_x = x_min
    x_min = x_min - learning_rate*gradient
    history_x.append(x_min)
    if (abs(J(last_x)-J(x_min))<error):
        break


plt.plot(plot_x,plot_y)
plt.plot(np.array(history_x),J(np.array(history_x)),color='r',marker='*')   #绘制x的轨迹
plt.show()
print(f"函数极小值点{x_min}")