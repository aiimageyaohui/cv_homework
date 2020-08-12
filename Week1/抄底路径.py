# -*- coding: utf-8 -*-
# @Time    : 2020/8/12 16:39
# @Author  : AiVision_YaoHui
# @FileName: 抄底路径.py
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

matplotlib.use("TkAgg")

# 定义待优化函数
def f(x):
    return (5 * x[0] + 3 * x[1] - 1) ** 2 + (-3 * x[0] - 4 * x[1] + 1) ** 2


def plot_position(x,y,z,fig,tracex,tracey,tracez):
    plt.ion()
    plt.clf()
    x_ = np.arange(-20, 20, 1)
    y_ = np.arange(-20, 20, 1)
    print(x)
    ax = Axes3D(fig)
    X, Y = np.meshgrid(x_, y_)
    Z = f([X, Y])
    plt.xlabel('x')
    plt.ylabel('y')
    tracex.append(float(x))
    tracey.append(float(y))
    tracez.append(float(z))
    # print(tracex)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow',alpha=0.5)
    # ax.plot_wireframe(X, Y, Z,alpha=0.1)
    # 画出optimizer优化出的位置点
    ax.scatter(x, y, z, c='r', s=10)  # 参数s控制点的大小
    ax.plot3D(tracex, tracey, tracez, c='r')
    plt.show()
    plt.pause(0.001)
# 给定任意初始化点
x = torch.tensor([20., 20.], requires_grad=True)
optimizer = torch.optim.Adam([x, ])
Z_ = 0
fig = plt.figure()
tracex = []
tracey = []
tracez = []
for step in range(30001):
    if step:
        optimizer.zero_grad()
        L.backward(retain_graph=True)
        optimizer.step()
    L = f(x)

    if step % 1000 == 0:
        print('step:{} , x = {} , value = {}'.format(step, x.tolist(), L))
        Z_ = L
        plot_position(x[0].detach().numpy(), x[1].detach().numpy(), Z_.detach().numpy(),fig,tracex,tracey,tracez)