import numpy as np
import matplotlib
import queue
import wx
matplotlib.use('WXAgg')
from matplotlib.animation import FuncAnimation
from collections import deque
import matplotlib.pyplot as plt
import time
import csv
import cv2
import random
import sounddevice as sd
import sys

ALPHA1 = 0
ALPHA = 100000

A = np.array([[1,1],[0,1]])
b = np.array([[0],[1]])
c = np.array([[1],[0]])

P0 = np.array([[ALPHA,0],[0,ALPHA]])
P01 = np.array([[ALPHA1,0],[0,ALPHA1]])

x0 = np.array([[0],[0]])
N = 8
M = 1  # 実験の回数

X_hat_log = deque([])
X_hat_A_log = deque([])
X = deque([])
X1_hat_log = deque([])
X1_hat_A_log = deque([])
X1 = deque([])
for k in range(M):

    x_hat_log = deque([])
    x_hat_A_log = deque([])
    x_log = deque([])
    x_hat_log.append(x0)
    x_hat_A_log.append(x0)
    x_log.append(x0)

    x = x0
    x_hat = x
    P = P0
    v = np.random.normal(0, 10, (N,1))
    w = np.random.normal(0, 20, (N,1))
    for i in range(N):
        xlog = x_hat

        x = A @ x + v[i]
        y = c.T @ x + w[i]

        x_hat_ = A @ xlog
        P_ = A @ P @ A.T + 10 * b @ b.T

        g = 1/(c.T @ P_ @ c + 20) * P_ @ c
        x_hat = x_hat_ + g @ (y - c.T @ x_hat_)
        P = (np.array([[1,0],[0,1]]) - g @ c.T) @ P_

        x_hat_log.append(x_hat_)
        x_hat_A_log.append(x_hat)
        x_log.append(x)

    X.append(x_log)
    X_hat_log.append(x_hat_log)
    X_hat_A_log.append(x_hat_A_log)


    x_hat_log = deque([])
    x_hat_A_log = deque([])
    x_log = deque([])
    x_hat_log.append(x0)
    x_hat_A_log.append(x0)
    x_log.append(x0)

    x = x0
    x_hat = x
    P = P01
    for i in range(N):
        xlog = x_hat

        x = A @ x + v[i]
        y = c.T @ x + w[i]

        x_hat_ = A @ xlog
        P_ = A @ P @ A.T + 10 * b @ b.T

        g = 1/(c.T @ P_ @ c + 20) * P_ @ c
        x_hat = x_hat_ + g @ (y - c.T @ x_hat_)
        P = (np.array([[1,0],[0,1]]) - g @ c.T) @ P_

        x_hat_log.append(x_hat_)
        x_hat_A_log.append(x_hat)
        x_log.append(x)

    X1.append(x_log)
    X1_hat_log.append(x_hat_log)
    X1_hat_A_log.append(x_hat_A_log)




fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)

for k in range(M):
    ax1.set_title("pre alpha = " + str(ALPHA))
    ax1.set_xlim([-1*max(abs(np.min(X_hat_A_log)), np.max(X_hat_A_log)), max(abs(np.min(X_hat_A_log)), np.max(X_hat_A_log))])
    ax1.set_ylim([-100,100])
    ax1.plot([x[0] for x in X_hat_log[k]], [x[1] for x in X_hat_log[k]])

    ax1.plot([x[0] for x in X[k]], [x[1] for x in X[k]], linewidth=5, alpha=0.5) #, markersize=np.linalg.det(P))

    ax1.scatter(X_hat_log[k][N][0][0], X_hat_log[k][N][1][0], s=100)
    ax1.scatter(0, 0, s=500, marker="+")

for k in range(M):
    ax2.set_title("post alpha = " + str(ALPHA))
    ax2.set_xlim([-1*max(abs(np.min(X_hat_A_log)), np.max(X_hat_A_log)), max(abs(np.min(X_hat_A_log)), np.max(X_hat_A_log))])
    ax2.set_ylim([-100, 100])
    ax2.plot([x[0] for x in X_hat_A_log[k]], [x[1] for x in X_hat_A_log[k]])

    ax2.plot([x[0] for x in X[k]], [x[1] for x in X[k]], linewidth=5, alpha=0.5) #, markersize=np.linalg.det(P))

    ax2.scatter(X_hat_A_log[k][N][0][0], X_hat_A_log[k][N][1][0], s=100)
    ax2.scatter(0,0, s=500, marker="+")


for k in range(M):
    ax3.set_title("pre alpha = " + str(ALPHA1))
    ax3.set_xlim([-1*max(abs(np.min(X_hat_A_log)), np.max(X_hat_A_log)), max(abs(np.min(X_hat_A_log)), np.max(X_hat_A_log))])
    ax3.set_ylim([-100,100])
    ax3.plot([x[0] for x in X1_hat_log[k]], [x[1] for x in X1_hat_log[k]])

    ax3.plot([x[0] for x in X1[k]], [x[1] for x in X1[k]], linewidth=5, alpha=0.5) #, markersize=np.linalg.det(P))

    ax3.scatter(X1_hat_log[k][N][0][0], X1_hat_log[k][N][1][0], s=100)
    ax3.scatter(0, 0, s=500, marker="+")

for k in range(M):
    ax4.set_title("post alpha = " + str(ALPHA1))
    ax4.set_xlim([-1*max(abs(np.min(X_hat_A_log)), np.max(X_hat_A_log)), max(abs(np.min(X_hat_A_log)), np.max(X_hat_A_log))])
    ax4.set_ylim([-100, 100])
    ax4.plot([x[0] for x in X1_hat_A_log[k]], [x[1] for x in X1_hat_A_log[k]])

    ax4.plot([x[0] for x in X1[k]], [x[1] for x in X1[k]], linewidth=5, alpha=0.5) #, markersize=np.linalg.det(P))

    ax4.scatter(X1_hat_A_log[k][N][0][0], X1_hat_A_log[k][N][1][0], s=100)
    ax4.scatter(0,0, s=500, marker="+")




plt.show()










"""x_hat_log = deque([])
x_hat_A_log = deque([])
x = x0
P = P0
for i in range(N):
    xlog = x
    v = np.random.normal(0,10)
    w = np.random.normal(0,20)
    x = A @ x + v
    y = c.T @ x + w

    x_hat_ = A @ xlog
    P_ = A @ P @ A.T + 10 * b @ b.T

    g = 1/(c.T @ P_ @ c + 20) * P_ @ c
    x_hat = x_hat_ + g @ (y - c.T @ x_hat_)
    P = (np.array([[1,0],[0,1]]) - g @ c.T) @ P_

    x_hat_log.append(x_hat_)
    x_hat_A_log.append(x_hat)"""


