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
    ones = torch.ones(len(x))
    zeros = torch.zeros(len(x))
    if torch.all(torch.eq(x, ones)) or torch.all(torch.eq(x, ones)):
        return np.log2(len(x))
    else:
        with torch.no_grad():
            a = N_w(x)
            y = np.asarray(x)
            y = y[::-1]
            b = N_w(y)
        return np.log2(len(x)) * (a + b)/2

def N_w(S):
    # get number of words in dictionary
    i = 0
    C = 1
    u = 1
    v = 1
    vmax = v
    while u + v < len(S):
        if S[i + v] == S[u + v]:
            v = v + 1
        else:
            vmax = max(v, vmax)
            i = i + 1
            if i == u:
                C = C + 1
                u = u + vmax
                v = 1
                i = 0
                vamx = v
            else:
                v = 1
    if v != 1:
        C = C + 1
    return C

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

def get_error(model, inputs, labels, d):
        model.eval()
        inputs = Variable(inputs, requires_grad=False)
        labels = Variable(labels, requires_grad=False)
        logits = model.forward(inputs)
        predicts = Output(logits)
        k = predicts - labels
        a = torch.sum(torch.abs(k))
        return a/d

def predict(model, inputs):
        model.eval()
        inputs = Variable(inputs, requires_grad=False)
        logits = model.forward(inputs)
        return logits

n = 7  ## dimension of input data, user-defined
m = 2 ** n  ## number of data points
m_2 = 2 ** (n - 1)
m_3 = 2 ** (n - 2)
layer_num = 3  ## number of layers of the neural network, user-defined
neu = 40  ## neurons per layer
epochs = 50 ## training time
mean = 0.0 ## mean of initialization
scale = 1.0 ## var of initialization

## data: 7 * 128
data = np.zeros([2 ** n, n], dtype=np.float32)
for i in range(2 ** n):
    bin = np.binary_repr(i, n)
    a = np.array(list(bin), dtype=int)
    data[i, :] = a
data = torch.from_numpy(data)

## generate training set and inference set
XTrain = torch.zeros(m_2, n)
XTest = torch.zeros(m_2, n)
for i in range(m_3):
    XTrain[i, :] = data[i, :]
    XTrain[i + m_3, :] = data[i + m_2, :]
    XTest[i, :] = data[i + m_3, :]
    XTest[i, :] = data[i + 3 * m_3, :]

## choose target, need to choose targets of different LVC
targets, YTrains, YTests, TLVS= [], [], [], []


## target of LVC: 7
t = torch.ones(2 ** n)
YTrain = torch.zeros(2 ** (n-1))
YTest = torch.zeros(2 ** (n-1))
for i in range(2 ** (n - 1)):
    YTrain[i] = t[i]
    YTest[i] = t[i + m_2]
t = t.long()
YTrain = YTrain.long()
YTest = YTest.long()
targets.append(t)
YTrains.append(YTrain.clone())
YTests.append(YTest.clone())
TLVS.append(int(7))

## target of LVC: 28
for i in range(m_3):
    t[i] = 1
    t[i + m_2] = 1
    t[i + m_3] = 0
    t[i + m_3 + m_2] = 0
for i in range(2 ** (n - 1)):
    YTrain[i] = t[i]
    YTest[i] = t[i + m_2]
t = t.long()
YTrain = YTrain.long()
YTest = YTest.long()
targets.append(t)
YTrains.append(YTrain.clone())
YTests.append(YTest.clone())
TLVS.append(int(28))

## targe of LVC：49
for i in range(m):
    if i%17 == 0 or i%17 == 1 or i%17 == 4 or i%17 == 9 or i%17 == 16 or i%17 == 8 or i%17 == 2 or i%17 == 15:
        t[i] = 1
    else:
        t[i] = 0
for i in range(2 ** (n - 1)):
    YTrain[i] = t[i]
    YTest[i] = t[i + m_2]
t = t.long()
YTrain = YTrain.long()
YTest = YTest.long()
targets.append(t)
YTrains.append(YTrain.clone())
YTests.append(YTest.clone())
TLVS.append(int(49))

## targe of LVC：63
for i in range(m):
    if i%41 == 0 or i%41 == 1 or i%41 == 4 or i%41 == 9 or i%41 == 16 or i%41 == 25 or i%41 == 36 or i%41 == 6:
        t[i] = 1
    else:
        t[i] = 0
for i in range(2 ** (n - 1)):
    YTrain[i] = t[i]
    YTest[i] = t[i + m_2]
t = t.long()
YTrain = YTrain.long()
YTest = YTest.long()
targets.append(t)
YTrains.append(YTrain.clone())
YTests.append(YTest.clone())
TLVS.append(int(get_LVComplexity(t)))


## targe of LVC：70
for i in range(m):
    if i%53 == 0 or i%53 == 1 or i%53 == 4 or i%53 == 9 or i%53 == 16 or i%53 == 25 or i%53 == 36 or i%53 == 49 or i%53 == 11 or i%53 == 28:
        t[i] = 1
    else:
        t[i] = 0
for i in range(2 ** (n - 1)):
    YTrain[i] = t[i]
    YTest[i] = t[i + m_2]
t = t.long()
YTrain = YTrain.long()
YTest = YTest.long()
targets.append(t)
YTrains.append(YTrain.clone())
YTests.append(YTest.clone())
TLVS.append(int(get_LVComplexity(t)))

## targe of LVC：84
for i in range(m):
    if i%89 == 0 or i%89 == 1 or i%89 == 4 or i%89 == 9 or i%89 == 16 or i%89 == 25 or i%89 == 36 or i%89 == 49 or i%89 == 64 or i%89 == 81 or i%89 == 11or i%89 == 32:
        t[i] = 1
    else:
        t[i] = 0
for i in range(2 ** (n - 1)):
    YTrain[i] = t[i]
    YTest[i] = t[i + m_2]
t = t.long()
YTrain = YTrain.long()
YTest = YTest.long()
targets.append(t)
YTrains.append(YTrain.clone())
YTests.append(YTest.clone())
TLVS.append(int(get_LVComplexity(t)))

## targe of LVC：108
t = torch.tensor([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0.,
        0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
        1., 1., 1., 1., 0., 0., 0., 0., 0., 1., 1., 0., 1., 0., 1., 0., 0., 0.,
        0., 1., 0., 0., 0., 1., 1., 0., 0., 1., 1., 0., 1., 0., 0., 0., 1., 1.,
        1., 1., 0., 1., 1., 0., 1., 1., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.,
        0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
        0., 0., 0., 0., 1., 0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 1., 0.,
        1., 0.])
for i in range(2 ** (n - 1)):
    YTrain[i] = t[i]
    YTest[i] = t[i + m_2]
t = t.long()
YTrain = YTrain.long()
YTest = YTest.long()
targets.append(t)
YTrains.append(YTrain.clone())
YTests.append(YTest.clone())
TLVS.append(int(get_LVComplexity(t)))


## targe of LVC：126
t = torch.tensor([1., 1., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 0., 1., 0., 0., 1., 0.,
        1., 1., 0., 1., 0., 0., 0., 1., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
        0., 1., 0., 0., 1., 1., 1., 1., 0., 0., 1., 0., 1., 0., 1., 0., 1., 1.,
        1., 1., 1., 1., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 1., 0., 1.,
        1., 0., 0., 1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 0., 0., 0., 1., 1.,
        0., 0., 1., 0., 1., 1., 1., 0., 0., 1., 1., 0., 1., 1., 1., 0., 0., 0.,
        0., 1., 0., 1., 1., 1., 0., 0., 1., 0., 0., 1., 0., 1., 0., 1., 1., 0.,
        1., 1.])
for i in range(2 ** (n - 1)):
    YTrain[i] = t[i]
    YTest[i] = t[i + m_2]
t = t.long()
YTrain = YTrain.long()
YTest = YTest.long()
targets.append(t)
YTrains.append(YTrain.clone())
YTests.append(YTest.clone())
TLVS.append(int(get_LVComplexity(t)))

## targe of LVC：143
t = torch.tensor([0., 0., 1., 1., 0., 1., 0., 1., 1., 0., 0., 1., 1., 0., 1., 0., 1., 0.,
        0., 1., 0., 0., 1., 1., 1., 0., 0., 1., 1., 1., 1., 0., 1., 1., 0., 0.,
        0., 1., 0., 0., 1., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0.,
        0., 0., 0., 1., 0., 0., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 0., 1.,
        0., 0., 0., 0., 1., 0., 0., 1., 0., 1., 1., 0., 0., 1., 0., 1., 0., 1.,
        1., 0., 1., 0., 1., 1., 1., 0., 1., 0., 0., 1., 1., 1., 0., 1., 1., 0.,
        0., 1., 1., 0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 1., 0., 1., 1., 1.,
        0., 1.])
for i in range(2 ** (n - 1)):
    YTrain[i] = t[i]
    YTest[i] = t[i + m_2]
t = t.long()
YTrain = YTrain.long()
YTest = YTest.long()
targets.append(t)
YTrains.append(YTrain.clone())
YTests.append(YTest.clone())
TLVS.append(int(get_LVComplexity(t)))


## define loss function
loss = torch.nn.CrossEntropyLoss(size_average=True)

def process(process_key, return_dict, MC_num):
    model1s, optimizer1s = [], []
    model2s, optimizer2s = [], []
    error_nonBN = torch.zeros(epochs)
    error_BN = torch.zeros(epochs)
    non_BN_mean_complexity = torch.zeros(epochs)
    BN_mean_complexity = torch.zeros(epochs)
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
        optimizer1 = optim.Adam(model1.parameters(), lr=0.02)
        optimizer2 = optim.Adam(model2.parameters(), lr=0.02)

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
            train(model1s[MC], loss, optimizer1s[MC], XTrain, YTrains[process_key])
            train(model2s[MC], loss, optimizer2s[MC], XTrain, YTrains[process_key])
            Aggregate1 = predict(model1s[MC], data)
            Aggregate2 = predict(model2s[MC], data)
            Output_1 = Output(Aggregate1)
            Output_2 = Output(Aggregate2)
            L_nonBN[MC] = get_error(model1s[MC], XTrain, YTrains[process_key], m_2)
            L_BN[MC] = get_error(model2s[MC], XTrain, YTrains[process_key], m_2)
            a = get_LVComplexity(Output_1)
            b = get_LVComplexity(Output_2)
            Complexity_agg[MC] = a
            Complexity_agg_BN[MC] = b

        error_nonBN[epoch] = torch.mean(L_nonBN)
        error_BN[epoch] = torch.mean(L_BN)
        non_BN_mean_complexity[epoch] = torch.mean(Complexity_agg)
        BN_mean_complexity[epoch] = torch.mean(Complexity_agg_BN)

    return_dict[process_key] = (error_nonBN, error_withBN, non_BN_mean_complexity, BN_mean_complexity)



if __name__ == '__main__':
    num_jobs = 10   # number of differnet targets we have chosen
    MC_num = int(100)
    total_times = MC_num * num_jobs
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    for i in range(num_jobs):
        p = multiprocessing.Process(target=process, args=(i, return_dict, MC_num))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()

    error, error_BN, cplx, cplx_BN = None, None, None, None

    for pair in return_dict.values():
        if error is None and error_BN is None and cplx is None and cplx_BN is None:
            error, error_BN, cplx, cplx_BN = np.asarray(pair[0]), np.asarray(pair[1]), \
                                             np.asarray(pair[2]), np.asarray(pair[3])
        else:
            error = np.concatenate((error, np.asarray(pair[0])), axis=None)
            error_BN = np.concatenate((error_BN, np.asarray(pair[1])), axis=None)
            cplx = np.concatenate((cplx, np.asarray(pair[2])), axis=None)
            cplx_BN = np.concatenate((cplx_BN, np.asarray(pair[3])), axis=None)


    # plot
    X = np.arange(1, epochs + 1, 1)
    fig, ax = plt.subplots(nrows=10, ncols=2, figsize=(15, 15),constrained_layout=True)
    for h in range(10):
        ax[h,0].plot(X, error[h], label='NN')
        ax[h,0].plot(X, error_BN[h], label='NN + BN')
        ax[h, 1].plot(X,cplx[h], label='NN')
        ax[h, 1].plot(X, cplx_BN[h], label='NN + BN')
        ax[h, 0].legend(loc="upper right")
        ax[h, 1].legend(loc="upper right")
        ax[h, 0].set_xlabel(f'Generalization/Test Error; Target Complexity: {TLVS[h]}')
        ax[h, 1].set_xlabel(f'Output Complexity; Target Complexity: {TLVS[h]}')
        ax[h, 0].set_ylabel('error')
        ax[h, 1].set_ylabel('complexity')
    fig.show()

























