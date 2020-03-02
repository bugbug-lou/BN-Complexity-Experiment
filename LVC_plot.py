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

def get_error(model, inputs, labels):
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

n = 7  ## dimension of input data, user-defined
m = 2 ** n  ## number of data points
k = 2 ** m
m_2 = 2 ** (n - 1)
m_3 = 2 ** (n - 2)
layer_num = 3  ## number of layers of the neural network, user-defined
neu = 40  ## neurons per layer
epochs = 30 ## training time
mean = 0.0 ## mean of initialization
scale = 1.0 ## var of initialization

## initialize data sets as binary strings
data = np.zeros([m, n])
for i in range(m):
    bin = np.binary_repr(i, n)
    a = np.array(list(bin), dtype=int)
    data[i, :] = a

## Target Function: we choose one that is of intermediate Lempel_Ziv complexity
target = np.zeros(m)
for i in range(m_2):
    target[i] = 1
    target[i + m_2] = 0
data = torch.from_numpy(data)
target = torch.from_numpy(target)
target = target.long()

## generate training set and inference set
XTrain = torch.zeros(m_2, n)
XTest = torch.zeros(m_2, n)
YTrain = torch.zeros(m_2)
YTest = torch.zeros(m_2)
for i in range(m_3):
    XTrain[i, :] = data[i, :]
    XTrain[i + m_3, :] = data[i + m_2, :]
    YTrain[i] = target[i]
    YTrain[i + m_3] = target[i + 3 * m_3]
    XTest[i, :] = data[i + m_3, :]
    XTest[i, :] = data[i + 3 * m_3, :]
    YTest[i] = target[i + m_3]
    YTest[i + m_3] = target[i + 3 * m_3]
Complexity_train = lempel_ziv_complexity(array_to_string(YTrain))
Complexity = lempel_ziv_complexity(array_to_string(target))
YTrain = YTrain.long()
YTest = YTest.long()
error_nonBN = torch.zeros(epochs)
error_BN = torch.zeros(epochs)
non_BN_mean_complexity = torch.zeros(epochs)
BN_mean_complexity = torch.zeros(epochs)

# initialize MC number of models
MC_num = 100  ## number of models
model1s, optimizer1s = [], []
model2s, optimizer2s = [], []

## define loss function
loss = torch.nn.CrossEntropyLoss(size_average=True)

for MC in range(MC_num):
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
        model2.FC1.weight = torch.nn.Parameter(model1.FC1.weight.clone().detach())
        model2.FC2.weight = torch.nn.Parameter(model1.FC2.weight.clone().detach())
        model2.FC3.weight = torch.nn.Parameter(model1.FC3.weight.clone().detach())
    # define optimizer
    optimizer1 = optim.Adam(model1.parameters(), lr=0.01)
    optimizer2 = optim.Adam(model2.parameters(), lr=0.01)

    model1s.append(model1)
    model2s.append(model2)
    optimizer1s.append(optimizer1)
    optimizer2s.append(optimizer2)



for epoch in range(epochs):
    if epoch % 5 == 0:
        print(f'{datetime.datetime.now()} Epoch {epoch} complete!')
    ## the main program
    L_nonBN = torch.zeros(MC_num)
    L_BN = torch.zeros(MC_num)
    Complexity_agg_BN = torch.zeros(MC_num)
    Complexity_agg = torch.zeros(MC_num)
    for MC in range(MC_num):
        train(model1s[MC], loss, optimizer1s[MC], XTrain, YTrain)
        train(model2s[MC], loss, optimizer2s[MC], XTrain, YTrain)
        data = data.float()
        Aggregate1 = predict(model1s[MC], data)
        Aggregate2 = predict(model2s[MC], data)
        Output_1 = Output(Aggregate1)
        Output_2 = Output(Aggregate2)
        L_nonBN[MC] = get_error(model1s[MC], data, target)
        L_BN[MC] = get_error(model2s[MC], data, target)
        a = lempel_ziv_complexity(array_to_string(Output_1))
        b = lempel_ziv_complexity(array_to_string(Output_2))
        Complexity_agg[MC] = a
        Complexity_agg_BN[MC] = b

    error_nonBN[epoch] = torch.mean(L_nonBN)
    error_BN[epoch] = torch.mean(L_BN)
    non_BN_mean_complexity[epoch] = torch.mean(Complexity_agg)
    BN_mean_complexity[epoch] = torch.mean(Complexity_agg_BN)

# plot
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
M = min(min(BN_mean_complexity), min(non_BN_mean_complexity))
Ma = max(max(BN_mean_complexity), max(non_BN_mean_complexity))
T = min(min(error_BN), min(error_nonBN))
Ta = max(max(error_BN), max(error_nonBN))
X = np.arange(epochs)
ax1.plot(X, error_nonBN, label="Error, NN")
ax1.plot(X, error_BN, label="Error, NN with batch norm")
ax1.legend(loc="upper right")
ax1.set_xlabel('epochs')
ax1.set_ylabel('error rates')
ax2.plot(X, non_BN_mean_complexity, label="Mean complexity, NN")
ax2.plot(X, BN_mean_complexity, label="Mean complexity, NN with batch norm")
ax2.legend(loc="upper right")
ax2.set_xlabel('epochs')
ax2.set_ylabel('mean complexity')
plt.savefig('lvc_lc.png')
plt.show()

