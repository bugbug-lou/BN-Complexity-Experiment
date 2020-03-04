import numpy as np
import datetime
import random
from matplotlib import pyplot as plt
import multiprocessing
import torch
from torch.autograd import Variable
from torch import optim
from lempel_ziv_complexity import lempel_ziv_complexity
import collections
import argparse

def array_to_string(x):
    y = ''
    for l in x:
        y += str(int(l))
    return y

def Output(x):
    a = x.size()[0]
    y = torch.zeros(a)
    for i in range(a):
        if x[i, 0] > x[i, 1]:
            y[i] = 0
        else:
            y[i] = 1
    return y

def get_max_freq(x):
    T = collections.Counter(x)
    Y = np.array(list(T.values()), dtype=np.longfloat)
    a = np.max(Y)
    for f in list(T.keys()):
        if T[f] == a:
            return f

def get_LVComplexity(x):
    return lempel_ziv_complexity(array_to_string(x))

def train(model, loss, optimizer, inputs, labels):
        model.train()
        inputs = Variable(inputs, requires_grad=False)
        labels = Variable(labels, requires_grad=False)
        # reset gradient
        optimizer.zero_grad()
        # forward loop
        logits = model.forward(inputs)
        output = loss.forward(logits, labels)
        # backward
        output.backward()
        optimizer.step()
        return output.item()

def get_error(model, inputs, labels, m):
        model.eval()
        inputs = Variable(inputs, requires_grad=False)
        labels = Variable(labels, requires_grad=False)
        logits = model.forward(inputs)
        predicts = Output(logits)
        k = predicts - labels
        a = torch.sum(torch.abs(k))
        return a/m

def predict(model, inputs):
        model.eval()
        inputs = Variable(inputs, requires_grad=False)
        logits = model.forward(inputs)
        return logits

## some parameters
n = 10  ## dimension of input data, user-defined
m = 2 ** n  ## number of data points
m_2 = 2 ** (n - 1)
m_3 = 2 ** (n - 2)
layer_num = 3  ## number of layers of the neural network, user-defined
neu = 40  ## neurons per layer
mod_num = 20 ## numbers of models used for each example
mean = 0.0 ## mean of initialization
scale = 1.0 ## var of initialization


## choose target, need to choose targets of different LVC
epochs, dims, datas, targets, XTrains, YTrains, TLVS= [], [], [], [], [], [], []
epochs.append(int(3))
epochs.append(int(10))
for i in range(2,9):
    epochs.append(int(36))


## target of LVC: 7
dim = 4
t = torch.zeros(2 ** dim)
data = np.zeros([2 ** dim, dim], dtype=np.float32)
XTrain = torch.zeros([2 ** (dim-1), dim])
YTrain = torch.zeros(2 ** (dim-1))
for i in range(2 ** dim):
    bin = np.binary_repr(i, dim)
    a = np.array(list(bin), dtype=int)
    data[i, :] = a
data = torch.from_numpy(data)

for i in range(2 ** dim):
    if i%2 ==0:
        t[i] = 1
    else:
        t[i] = 0

for i in range(2 ** dim):
    if i < 2 ** (dim-1):
        YTrain[i] = t[i]
        XTrain[i,:] = data[i,:]

t = t.long()
YTrain = YTrain.long()
targets.append(t)
datas.append(data)
dims.append(dim)
XTrains.append(XTrain)
YTrains.append(YTrain)
TLVS.append(int(7))

## target of LVC: 31
dim = 8
t = torch.zeros(2 ** dim)
data = np.zeros([2 ** dim, dim], dtype=np.float32)
XTrain = torch.zeros([2 ** (dim-1), dim])
YTrain = torch.zeros(2 ** (dim-1))
for i in range(2 ** dim):
    bin = np.binary_repr(i, dim)
    a = np.array(list(bin), dtype=int)
    data[i, :] = a
data = torch.from_numpy(data)
for i in range(2 ** dim):
    if i%2 ==0:
        t[i] = 1
    else:
        t[i] = 0

for i in range(2 ** dim):
    if i < 2 ** (dim-1):
        YTrain[i] = t[i]
        XTrain[i,:] = data[i,:]

t = t.long()
YTrain = YTrain.long()
targets.append(t)
datas.append(data)
dims.append(dim)
XTrains.append(XTrain)
YTrains.append(YTrain)
TLVS.append(int(31))


## target of LVC: 48
dim = n
t = torch.zeros(2 ** dim)
data = np.zeros([2 ** dim, dim], dtype=np.float32)
XTrain = torch.zeros([2 ** (dim-1), dim])
YTrain = torch.zeros(2 ** (dim-1))
for i in range(2 ** dim):
    bin = np.binary_repr(i, dim)
    a = np.array(list(bin), dtype=int)
    data[i, :] = a
data = torch.from_numpy(data)
for i in range(2 ** dim):
    if i%m_3 ==0:
        t[i] = 1
    else:
        t[i] = 0
for i in range(2 ** dim):
    if i < 2 ** (dim-1):
        YTrain[i] = t[i]
        XTrain[i,:] = data[i,:]

t = t.long()
YTrain = YTrain.long()
targets.append(t)
datas.append(data)
dims.append(dim)
XTrains.append(XTrain)
YTrains.append(YTrain)
TLVS.append(int(48))


## target of LVC: 65
data = np.zeros([2 ** dim, dim], dtype=np.float32)
XTrain = torch.zeros([2 ** (dim-1), dim])
YTrain = torch.zeros(2 ** (dim-1))
for i in range(2 ** dim):
    bin = np.binary_repr(i, dim)
    a = np.array(list(bin), dtype=int)
    data[i, :] = a
data = torch.from_numpy(data)
for i in range(m_3):
    t[i] = 1
    t[i + m_2] = 1
    t[i + m_3] = 0
    t[i + m_3 + m_2] = 0
for i in range(2 ** dim):
    if i < 2 ** (dim-1):
        YTrain[i] = t[i]
        XTrain[i,:] = data[i,:]

t = t.long()
YTrain = YTrain.long()
targets.append(t)
datas.append(data)
dims.append(dim)
XTrains.append(XTrain)
YTrains.append(YTrain)
TLVS.append(int(65))

## target of LVC: 86
data = np.zeros([2 ** dim, dim], dtype=np.float32)
XTrain = torch.zeros([2 ** (dim-1), dim])
YTrain = torch.zeros(2 ** (dim-1))
for i in range(2 ** dim):
    bin = np.binary_repr(i, dim)
    a = np.array(list(bin), dtype=int)
    data[i, :] = a
data = torch.from_numpy(data)
for i in range(m):
    if i%4 == 0 or i%4 == 3:
        t[i] = 1
    else:
        t[i] = 0
for i in range(2 ** dim):
    if i < 2 ** (dim-1):
        YTrain[i] = t[i]
        XTrain[i,:] = data[i,:]

t = t.long()
YTrain = YTrain.long()
targets.append(t)
datas.append(data)
dims.append(dim)
XTrains.append(XTrain)
YTrains.append(YTrain)
TLVS.append(int(86))

## target of LVC: 108
data = np.zeros([2 ** dim, dim], dtype=np.float32)
XTrain = torch.zeros([2 ** (dim-1), dim])
YTrain = torch.zeros(2 ** (dim-1))
for i in range(2 ** dim):
    bin = np.binary_repr(i, dim)
    a = np.array(list(bin), dtype=int)
    data[i, :] = a
data = torch.from_numpy(data)
for i in range(m):
    if i%7 == 0 or i%7 == 1 or i%7 == 3 or i%7 == 6:
        t[i] = 1
    else:
        t[i] = 0
for i in range(2 ** dim):
    if i < 2 ** (dim-1):
        YTrain[i] = t[i]
        XTrain[i,:] = data[i,:]

t = t.long()
YTrain = YTrain.long()
targets.append(t)
datas.append(data)
dims.append(dim)
XTrains.append(XTrain)
YTrains.append(YTrain)
TLVS.append(int(108))

## targe of LVC：124
data = np.zeros([2 ** dim, dim], dtype=np.float32)
XTrain = torch.zeros([2 ** (dim-1), dim])
YTrain = torch.zeros(2 ** (dim-1))
for i in range(2 ** dim):
    bin = np.binary_repr(i, dim)
    a = np.array(list(bin), dtype=int)
    data[i, :] = a
data = torch.from_numpy(data)
for i in range(m):
    if i%11 == 0 or i%11 == 1 or i%11 == 4 or i%11 == 9 or i%11 == 5:
        t[i] = 1
    else:
        t[i] = 0
for i in range(2 ** dim):
    if i < 2 ** (dim-1):
        YTrain[i] = t[i]
        XTrain[i,:] = data[i,:]

t = t.long()
YTrain = YTrain.long()
targets.append(t)
datas.append(data)
dims.append(dim)
XTrains.append(XTrain)
YTrains.append(YTrain)
TLVS.append(int(124))

## targe of LVC：142
data = np.zeros([2 ** dim, dim], dtype=np.float32)
XTrain = torch.zeros([2 ** (dim-1), dim])
YTrain = torch.zeros(2 ** (dim-1))
for i in range(2 ** dim):
    bin = np.binary_repr(i, dim)
    a = np.array(list(bin), dtype=int)
    data[i, :] = a
data = torch.from_numpy(data)
for i in range(m):
    if i%17 == 0 or i%17 == 1 or i%17 == 4 or i%17 == 9 or i%17 == 16 or i%17 == 8 or i%17 == 2 or i%17 == 15:
        t[i] = 1
    else:
        t[i] = 0
for i in range(2 ** dim):
    if i < 2 ** (dim-1):
        YTrain[i] = t[i]
        XTrain[i,:] = data[i,:]

t = t.long()
YTrain = YTrain.long()
targets.append(t)
datas.append(data)
dims.append(dim)
XTrains.append(XTrain)
YTrains.append(YTrain)
TLVS.append(int(142))

## targe of LVC：176
data = np.zeros([2 ** dim, dim], dtype=np.float32)
XTrain = torch.zeros([2 ** (dim-1), dim])
YTrain = torch.zeros(2 ** (dim-1))
for i in range(2 ** dim):
    bin = np.binary_repr(i, dim)
    a = np.array(list(bin), dtype=int)
    data[i, :] = a
data = torch.from_numpy(data)
p = 0.5 ##probabilty for choosing 0
zero_ones = np.array([0,1]) ## things to choose from
dis = np.array([p, 1-p]) ## probability input for choose function
Ori = np.zeros(m)
for i in range(m):
    Ori[i] = np.random.choice(zero_ones,p = dis)
t = torch.from_numpy(Ori)
for i in range(2 ** dim):
    if i < 2 ** (dim-1):
        YTrain[i] = t[i]
        XTrain[i,:] = data[i,:]

t = t.long()
YTrain = YTrain.long()
targets.append(t)
datas.append(data)
dims.append(dim)
XTrains.append(XTrain)
YTrains.append(YTrain)
TLVS.append(int(176))

## outputs
LVC_outputs, LVC_output_BNs, GE_outputs, GE_output_BNs, LVC_output_UEs, GE_output_UEs = [], [], [], [], [], []


## define loss function
loss = torch.nn.CrossEntropyLoss(size_average=True)

## train BN models and non-BN models based on different targets
for MC in range(9):
    print('MC sample ' + str(MC) + ' complete!')
    n = dims[MC]
    LVC = torch.zeros(mod_num)
    LVC_BN = torch.zeros(mod_num)
    GE = torch.zeros(mod_num)
    GE_BN = torch.zeros(mod_num)
    LVC_UE = torch.zeros(mod_num)
    GE_UE = torch.zeros(mod_num)
    for i in range(mod_num):
        model1 = torch.nn.Sequential()  # model without batch normalization
        model2 = torch.nn.Sequential()  # model with batch normalization

        # add some layers for model 1, this is without BN
        model1.add_module('FC1', torch.nn.Linear(n, neu))
        model1.add_module('relu1', torch.nn.ReLU())
        model1.add_module('FC2', torch.nn.Linear(neu, neu))
        model1.add_module('relu2', torch.nn.ReLU())
        model1.add_module('FC3', torch.nn.Linear(neu, 2))
        with torch.no_grad():
            torch.nn.init.normal_(model1.FC1.weight, mean=mean, std=scale)
            torch.nn.init.normal_(model1.FC2.weight, mean=mean, std=scale)
            torch.nn.init.normal_(model1.FC3.weight, mean=mean, std=scale)

        # add some layers for model 2, this is with BN
        model2.add_module('FC1', torch.nn.Linear(n, neu))
        model2.add_module('bn1', torch.nn.BatchNorm1d(neu))
        model2.add_module('relu1', torch.nn.ReLU())
        model2.add_module('FC2', torch.nn.Linear(neu, neu))
        model2.add_module('bn2', torch.nn.BatchNorm1d(neu))
        model2.add_module('relu2', torch.nn.ReLU())
        model2.add_module('FC3', torch.nn.Linear(neu, 2))
        with torch.no_grad():
            torch.nn.init.normal_(model2.FC1.weight, mean=mean, std=scale)
            torch.nn.init.normal_(model2.FC2.weight, mean=mean, std=scale)
            torch.nn.init.normal_(model2.FC3.weight, mean=mean, std=scale)

        # define optimizer
        optimizer1 = optim.Adam(model1.parameters(), lr=0.01)
        optimizer2 = optim.Adam(model2.parameters(), lr=0.01)

        # train
        epoch = 0
        while epoch < epochs[MC]:
            train(model1, loss, optimizer1, XTrains[MC], YTrains[MC])
            train(model2, loss, optimizer2, XTrains[MC], YTrains[MC])
            epoch = epoch + 1

        # prediction
        Aggregate1 = predict(model1, datas[MC])
        Aggregate2 = predict(model2, datas[MC])
        Output_1 = Output(Aggregate1)
        Output_2 = Output(Aggregate2)
        GE[i] = get_error(model1, datas[MC], targets[MC], 2 ** n)
        GE_BN[i] = get_error(model2, datas[MC], targets[MC], 2 ** n)
        LVC[i] = lempel_ziv_complexity(array_to_string(Output_1))
        LVC_BN[i] = lempel_ziv_complexity(array_to_string(Output_2))

        # randomly choose a 0-1 string as unbiased estimator
        dis = np.array([0.5, 0.5])  ## probability input for choose function
        Ori = np.zeros(2 ** n)
        for j in range(2 ** n):
            Ori[j] = np.random.choice(zero_ones, p=dis)
        predicts = torch.from_numpy(Ori)
        k = predicts - targets[MC]
        a = torch.sum(torch.abs(k))
        GE_UE[i] = a / 2 ** n
        LVC_UE[i] = get_LVComplexity(predicts)

    LVC_outputs.append(LVC), LVC_output_BNs.append(LVC_BN), GE_outputs.append(GE), GE_output_BNs.append(GE_BN)
    LVC_output_UEs.append(LVC_UE), GE_output_UEs.append(GE_UE)





# produce a plot concerning output function complexities
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
ax = (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9)
for h in range(9):
    ax[h].scatter(LVC_outputs[h], GE_outputs[h], label='Non BN', c='green', alpha=0.5)
    ax[h].scatter(LVC_output_BNs[h], GE_output_BNs[h], label='BN', c='red', alpha=0.5)
    ax[h].scatter(LVC_output_UEs[h], GE_output_UEs[h], label='Unbiased Estimator', c='blue', alpha=0.5)
    ax[h].legend(loc="upper right")
    ax[h].set_xlabel('Complexity')
    ax[h].set_ylabel('Error Rates')

fig.show()