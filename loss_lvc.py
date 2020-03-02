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


def output_anal(x):
    a = x.size()[0]
    y = torch.zeros(a)
    for i in range(a):
        if x[i, 0] > x[i, 1]:
            y[i] = 0
        else:
            y[i] = 1
    return y


def get_freq(x):
    T = collections.Counter(x)
    Y = np.array(list(T.values()), dtype=np.longfloat)
    Y = Y / MC_sample
    Y = np.sort(Y)
    Y = Y[::-1]
    return Y


def get_max_freq(x):
    T = collections.Counter(x)
    Y = np.array(list(T.values()), dtype=np.longfloat)
    a = np.max(Y)
    for f in list(T.keys()):
        if T[f] == a:
            return f


def get_LVComplexity(x):
    return lempel_ziv_complexity(array_to_string(x))

testtime = 30
Loss_nonBN = torch.zeros(testtime)
Loss_BN = torch.zeros(testtime)
non_BN_mean_complexity = torch.zeros(testtime)
BN_mean_complexity = torch.zeros(testtime)

for t in range(testtime):
    if t % (testtime / 5) == 0:
        print(f'{datetime.datetime.now()} No.{t} Complete!')
    ## parameters
    epochs = t  ## training time
    MC_sample = 100  ## number of sampling
    n = 7  ## dimension of input data, user-defined
    m = 2 ** n  ## number of data points
    k = 2 ** m
    m_2 = 2 ** (n - 1)
    m_3 = 2 ** (n - 2)
    layer_num = 3  ## number of layers of the neural network, user-defined
    neu = 40  ## neurons per layer

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

    ## define loss function
    loss = torch.nn.CrossEntropyLoss(size_average=True)


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


    def get_cost(model, loss, inputs, labels):
        model.eval()
        inputs = Variable(inputs, requires_grad=False)
        labels = Variable(labels, requires_grad=False)
        logits = model.forward(inputs)
        output = loss.forward(logits, labels)
        return output.item()


    def predict(model, inputs):
        model.eval()
        inputs = Variable(inputs, requires_grad=False)

        logits = model.forward(inputs)
        return logits


    ## the main program
    L_nonBN = torch.zeros(MC_sample)
    L_BN = torch.zeros(MC_sample)
    Complexity_agg_BN = torch.zeros(MC_sample)
    Complexity_agg = torch.zeros(MC_sample)

    h = 0
    while (h < MC_sample):
        ## initialize model
        model1 = torch.nn.Sequential()  ## model without batch normalization
        model2 = torch.nn.Sequential()  ## model with batch normalization

        ## add some layers for model 1, this is without BN
        model1.add_module('FC1', torch.nn.Linear(n, neu))
        model1.add_module('relu1', torch.nn.ReLU())
        model1.add_module('FC2', torch.nn.Linear(neu, neu))
        model1.add_module('relu2', torch.nn.ReLU())
        model1.add_module('FC3', torch.nn.Linear(neu, neu))
        model1.add_module('relu2', torch.nn.ReLU())
        model1.add_module('FC4', torch.nn.Linear(neu, neu))
        model1.add_module('relu2', torch.nn.ReLU())
        model1.add_module('FC5', torch.nn.Linear(neu, 2))

        ##model1.add_module('FC4', torch.nn.Linear(neu,1))
        #   #model1.add_module('relu4', torch.nn.ReLU())

        ## add some layers for model 2, this is with BN
        model2.add_module('FC1', torch.nn.Linear(n, neu))
        model2.add_module('bn1', torch.nn.BatchNorm1d(neu))
        model2.add_module('relu1', torch.nn.ReLU())
        model2.add_module('FC2', torch.nn.Linear(neu, neu))
        model2.add_module('bn2', torch.nn.BatchNorm1d(neu))
        model2.add_module('relu2', torch.nn.ReLU())
        model2.add_module('FC3', torch.nn.Linear(neu, 2))

        ## define optimizer
        optimizer1 = optim.SGD(model1.parameters(), lr=0.01, momentum=0.9)
        optimizer2 = optim.SGD(model2.parameters(), lr=0.01, momentum=0.9)

        for epoc in range(epochs):
            train(model1, loss, optimizer1, XTrain, YTrain)
            train(model2, loss, optimizer2, XTrain, YTrain)

        data = data.float()
        Aggregate1 = predict(model1, data)
        Aggregate2 = predict(model2, data)
        Output_1 = output_anal(Aggregate1)
        Output_2 = output_anal(Aggregate2)
        L_nonBN[h] = get_cost(model1, loss, data, target)
        L_BN[h] = get_cost(model2, loss, data, target)
        a = lempel_ziv_complexity(array_to_string(Output_1))
        b = lempel_ziv_complexity(array_to_string(Output_2))
        Complexity_agg[h] = a
        Complexity_agg_BN[h] = b
        diff_lvc[h] = a - b
        h = h + 1

    Loss_nonBN[t] = torch.mean(L_nonBN)
    Loss_BN[t] = torch.mean(L_BN)
    non_BN_mean_complexity[t] = torch.mean(Complexity_agg)
    BN_mean_complexity[t] = torch.mean(Complexity_agg_BN)
    t = t+1

# plot
M = min(min(BN_mean_complexity),min(non_BN_mean_complexity))
Ma = max(max(BN_mean_complexity),max(non_BN_mean_complexity))
X = np.arange(testtime)
#plt.plot(X,Loss_nonBN, label="Loss, no BatchNorm")
#plt.plot(X,Loss_BN label="Loss, BatchNorm")
plt.plot(X,non_BN_mean_complexity, label="Mean output complexity, NN")
plt.plot(X,BN_mean_complexity,label="Mean output complexity, NN with BatchNorm")
plt.xlabel('Epocs')
plt.ylabel('mean complexity')
plt.legend(loc="upper right")
plt.xlim(1, testtime)
plt.ylim(M, Ma)
plt.show()


