#!/usr/bin/env python
# -*- coding:utf-8 -*-
# -*- coding: utf-8 -*-
import numpy as np

import matplotlib.pyplot as plt
from scipy import linalg
from scipy.special import expit

from sklearn.preprocessing import LabelEncoder


def set_seed(seed=None):
    if seed is None:
        import time
        seed = int((time.time() * 10 ** 6) % 4294967295)
    try:
        np.random.seed(seed)
    except Exception as e:
        print("!!! WARNING !!!: Seed was not set correctly.")
        print("!!! Seed that we tried to use: " + str(seed))
        print("!!! Error message: " + str(e))
        seed = None
    return seed

def normalize(ori_data, flag='01'):
    data = ori_data.copy()
    if len(data.shape) > 1:  # 2-D
        N, K = data.shape
        minV = np.zeros(shape=K)
        maxV = np.zeros(shape=K)
        for i in range(N):
            minV[i] = np.min(data[i, :])
            maxV[i] = np.max(data[i, :])
            if np.abs(maxV[i] - minV[i]) > 0.00001:
                if flag == '01':  # normalize to [0, 1]
                    data[i, :] = (data[i, :] - minV[i]) / (maxV[i] - minV[i])
                else:
                    data[i, :] = 2 * (data[i, :] - minV[i]) / (maxV[i] - minV[i]) - 1
        return data, maxV, minV
    else:  # 1D
        minV = np.min(data)
        maxV = np.max(data)
        if np.abs(maxV - minV) > 0.00001:
            if flag == '01':  # normalize to [0, 1]
                data = (data - minV) / (maxV - minV)
            else:
                data = 2 * (data - minV) / (maxV - minV) - 1
        return data, maxV, minV


def re_normalize(ori_data, maxV, minV, flag='01'):
    data = ori_data.copy()
    if len(data.shape) > 1:  # 2-D
        Nc, K = data.shape
        for i in range(Nc):
            if np.abs(maxV[i] - minV[i]) > 0.00001:
                if flag == '01':  # normalize to [0, 1]
                    data[i, :] = data[i, :] * (maxV[i] - minV[i]) + minV[i]
                else:
                    data[i, :] = (data[i, :] + 1) * (maxV[i] - minV[i]) / 2 + minV[i]
    else:  # 1-D
        if np.abs(maxV - minV) > 0.00001:
            if flag == '01':  # normalize to [0, 1]
                data = data * (maxV - minV) + minV
            else:
                data = (data + 1) * (maxV - minV) / 2 + minV
    return data

import pandas as pd

# dataset 1: PM2.5.csv parameters optimized
pollution = pd.read_csv(r'./dataset/pollution.csv', delimiter=',').values
# pollution = pollution[:, 1:]  # .astype(np.float)
encoder = LabelEncoder()
pollution[:, 5] = encoder.fit_transform(pollution[:, 5])
# float ensure all data is float
pollution = pollution[:, 1:].astype('float32')
dataset0 = np.array(pollution[:,:], dtype=np.float32)
data0 = np.zeros(shape=(dataset0.shape[1], dataset0.shape[0]))
for i in range(8):
    data0[i, :] = dataset0[:, i]
data, maxV, minV = normalize(data0,'-01')

dataset_name = 't_last2'

# generate the ESNP reservoir for Beijing PM2.5 (ESNP)
print(data0.shape)
seed = 42
# Size 43800
initLen = 2000 # 2000
trainLen = 35800 # 35800
testLen = 8000 # 8000
inSize = outSize = 8 # input/output dimension
α = 0.4 # 0.4
β = 0.7 # beta 0.7
resSize = 52 # m 52
spectral_radius = 0.99 # rho 0.99
sigma = 0.38 #  delta 0.38
input_scaling = 1
reg = 1e-6 # lambda

# dataset 2: lorenz.csv
# lorenz = pd.read_csv(r'./dataset/lorenz.csv', header=None).values
# dataset0 = lorenz[:, :].astype(np.float32)
# data0 = np.zeros(shape=(dataset0.shape[1], dataset0.shape[0]))
# for i in range(3):
#     data0[i, :] = dataset0[:, i]
# data, maxV, minV = normalize(data0, '-01')
#
# # size: 3000
# dataset_name = 't_last2'
# seed = 42
# initLen = 500 # 500
# trainLen = 2500 #2500
# testLen = 500 # 500
# #generate the ESNP reservoir
# inSize = outSize = 3 # input/output dimension
# α = 0.025 #0.025
# β = 0.33
# resSize = 147 #147
# spectral_radius = 0.99
# sigma = 0.3 # 0.3
# input_scaling = 1
# reg = 1e-6 # -6


# dataset 3: rossler.csv opt with new init len
# rossler = pd.read_csv(r'./dataset/rossler.csv',header=None).values
# dataset0 = rossler[:, :].astype(np.float32)
# data0 = np.zeros(shape=(dataset0.shape[1], dataset0.shape[0]))
# for i in range(3):
#     data0[i, :] = dataset0[:, i]
#
# data, maxV, minV = normalize(data0,'-01')
# dataset_name = 't_last2'
# seed = 42
# #len_total 5500
# initLen = 321 #321
# trainLen = 4580 # 4580
# testLen = 920 # 920
#
# # generate the ESNP reservoir
# inSize = outSize = 3 # input/output dimension
# α = 0.33 # 0.33
# β = 0.0011 # 0.0009
# resSize = 57 # 57
# spectral_radius = 0.9 # 0.9
# sigma = 0.35 # 0.35
# input_scaling = 1
# reg = 1e-7 # -7

mode = 'prediction'
import math
min_rmse= math.inf
for mm in np.arange(0,1):
    RMSE_V = 0
    MAE_V = 0
    MSE_V = 0
    RMSE_std=[]
    MSE_std=[]
    MAE_std=[]
    for i in range(30):
        set_seed(seed)
        Win = (np.random.rand(resSize, 1 + inSize) - 0.5) * input_scaling
        W = np.random.rand(resSize, resSize) - 0.5
        Z = np.random.binomial(1, 1 - sigma, (resSize, resSize))  # Zero-one matrix with density 1-alpha (sparsity alpha)
        W = np.multiply(W, Z)  # Element-wise multiply to get desired sparsity
        rhoW = max(abs(linalg.eig(W)[0]))
        W *= spectral_radius / rhoW

        U = np.zeros((1 + inSize + resSize, trainLen - initLen))
        Yt = data[:, initLen + 1:trainLen + 1]  # Multivariable

        u = np.zeros((resSize, 1))

        for t in range(trainLen):
            x = data[:, t]  # Multivariable
            # ESNP update equation
            # Try new state equations
            #u = α * u + np.dot(W, np.tanh(np.dot(Win, np.r_[np.ones(1),x]).reshape(resSize,1) + np.dot(β, u))) # Multivariable
            #rand_term = np.random.randint(0, 2)
            if t%2 == 0:
                u = α * u + np.dot(W,  ((1 - β) * np.tanh(
                     np.dot(Win, np.r_[np.ones(1), x]).reshape(resSize, 1)) + β*np.tanh(u))) #Multivariable
            else:
                u = α * u + np.dot(W, np.tanh(
                    np.dot(Win, np.r_[np.ones(1), x]).reshape(resSize, 1) + np.dot(β, u)))  # Multivariable
            if t >= initLen:
                U[:, t - initLen] = np.r_[(np.ones(1), x, u.reshape(resSize,))]  # Multivariable

        U_T = U.T
        if reg is not None:
            # use ridge regression
            Wout = np.dot(np.dot(Yt, U_T), linalg.inv(np.dot(U, U_T) + \
                                                      reg * np.eye(1 + inSize + resSize)))
        else:
            # use pseudo inverse
            Wout = np.dot(Yt, linalg.pinv(U))

        Y = np.zeros((outSize, testLen))
        x = data[:, trainLen]  # Multivariable
        for t in range(testLen):
            #state equation -- new
            #u = α * u + np.dot(W, np.tanh(np.dot(Win, np.r_[np.ones(1), x]).reshape(resSize, 1) + np.dot(β, u)))  # Multivariable
            #Using expit (sigmoid function)
            #rand_term = np.random.randint(0,2)
            if t%2 == 0:
                u = α * u + np.dot(W,  ( (1 - β) * np.tanh(
                    np.dot(Win, np.r_[np.ones(1), x]).reshape(resSize, 1)) + β * np.tanh(u)))  # Multivariable
            else:
                u = α * u + np.dot(W, np.tanh(
                    np.dot(Win, np.r_[np.ones(1), x]).reshape(resSize, 1) + np.dot(β, u)))  # Multivariable
            #Output of ESNP system
            y = np.dot(Wout, np.r_[np.ones(1), x, u.reshape(resSize,)])  # Multivariable
            Y[:, t] = y
            if mode == 'generative':
                x = y
            elif mode == 'prediction':
                x = data[:, trainLen + t]  # Multivariable

        # compute MSE,RMSE,MAE for the first errorLen time steps
        errorLen = testLen
        np.savetxt('./predict/' + dataset_name + str(i) + '.csv', Y[0, 0:errorLen], delimiter=',')
        # Y = re_normalize(Y, maxV, minV, '-01')  # Multivariable
        mse = sum(np.square(data[0, trainLen+4:trainLen+errorLen+1] - Y[0, 4:errorLen])) / errorLen
        # mse = sum(np.square(data0[0,trainLen + 1:trainLen + errorLen + 1] - Y[0,0:errorLen])) / errorLen
        MSE_std.append(mse)
        from sklearn.metrics import mean_squared_error
        from math import sqrt

        rmse = sqrt(mean_squared_error(data[0, trainLen+4:trainLen+errorLen+1], Y[0, 4:errorLen]))
        # rmse = sqrt(mean_squared_error(data0[0,trainLen + 1:trainLen + errorLen + 1], Y[0,0:errorLen]))
        RMSE_std.append(rmse)
        from sklearn.metrics import median_absolute_error

        mae = median_absolute_error(data[0, trainLen+4:trainLen+errorLen+1], Y[0, 4:errorLen])
        #mae = median_absolute_error(data0[0, trainLen + 1:trainLen + errorLen + 1], Y[0, 0:errorLen])
        MAE_std.append(mae)
        # mae = median_absolute_error(data_org, data_pred)
        MAE_V += mae
        MSE_V += mse
        RMSE_V += rmse
    print('******************************************')
    print('resSize='+str(resSize))
    print('RMSE = ' + str(RMSE_V / 30)+'±'+str(np.std(RMSE_std)))
    print('MAE = ' + str(MAE_V / 30)+'±'+str(np.std(MAE_std)))
    print('MSE = ' + str(MSE_V / 30)+'±'+str(np.std(MSE_std)))
    if (RMSE_V / 30)<min_rmse:
        min_rmse = RMSE_V / 30
        resSize_N = resSize
print('resSize_N='+str(resSize_N))

#Y = re_normalize(Y,maxV,minV,'-01')

fig4 = plt.figure()
ax41 = fig4.add_subplot(111)
time = range(testLen)
ax41.plot(time, data[0, trainLen:trainLen + testLen + 1], 'r-', label='the original data')
ax41.plot(time, Y[0, 0:testLen], 'g--', label='the predicted data')
#ax41.plot(time, Y[0,0:testLen], 'g--', label='the predicted data')
ax41.set_ylabel("Magnitude")
ax41.set_xlabel('Time')
ax41.set_title('Beijing PM2.5')
fig5 = plt.figure()
ax5 = fig5.add_subplot(111)

ax5.plot(time[4:], np.abs((data[0, trainLen:trainLen+testLen+1]-Y[0, 0:testLen])[4:]), 'r')
# ax5.plot(time, np.abs(data0[0,trainLen + 1:trainLen + testLen + 1] - Y[0,0:testLen]), 'r')
ax5.set_ylabel("Magnitude")
ax5.set_xlabel('Time')
ax5.set_title('Beijing PM2.5')


ax41.legend()
plt.tight_layout()
plt.show()
