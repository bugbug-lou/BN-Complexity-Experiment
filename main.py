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
        y+=str(int(l))
    return y

def output_anal(x):
    a = x.size()[0]
    y = torch.zeros(a)
    for i in range(a):
        if x[i,0] > x[i,1]:
            y[i] = 0
        else:
            y[i] = 1
    return y

def get_freq(x):
    T = collections.Counter(x)
    Y = np.array(list(T.values()),dtype = np.longfloat)
    Y = Y/times
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


## parameters
epocs = 36 ## training time
times = 1000 ## number of sampling
n = 7  ## dimension of input data, user-defined
m = 2 ** n  ## number of data points
k = 2 ** m
m_2 = 2 ** (n-1)
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
Ran = list(range(m))
random.shuffle(Ran)
XTrain = torch.zeros(m_2, n)
XTest = torch.zeros(m_2, n)
YTrain = torch.zeros(m_2)
YTest = torch.zeros(m_2)
for i in range(m_2):
    XTrain[i, :] = data[Ran[i], :]
    YTrain[i] = target[Ran[i]]
    XTest[i, :] = data[Ran[i + m_2], :]
    YTest[i] = target[Ran[i + m_2]]
Complexity_train = lempel_ziv_complexity(array_to_string(YTrain))
YTrain = YTrain.long()
YTest = YTest.long()



## define loss function
loss = torch.nn.CrossEntropyLoss(size_average = True)


def train(model, loss, optimizer, inputs, labels):
    model.train()
    inputs = Variable(inputs, requires_grad = False)
    labels = Variable(labels, requires_grad = False)

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
def sampling_process(process_key, return_dict, times):
    P_f = list(range(times))
    P_f_BN = list(range(times))
    Complexity_agg_BN = torch.zeros(times)
    Complexity_agg = torch.zeros(times)
    diff_lvc = torch.zeros(times)
    h = 0
    while(h<times):
        if h%(times/10) == 0:
            print('complete %d', h)
        ## initialize model
        model1 = torch.nn.Sequential()  ## model without batch normalization
        model2 = torch.nn.Sequential()  ## model with batch normalization

        ## add some layers for model 1, this is without BN
        model1.add_module('FC1', torch.nn.Linear(n, neu))
        model1.add_module('relu1', torch.nn.ReLU())
        model1.add_module('FC2', torch.nn.Linear(neu, neu))
        model1.add_module('relu2', torch.nn.ReLU())
        model1.add_module('FC3', torch.nn.Linear(neu, 2))

        ##model1.add_module('FC4', torch.nn.Linear(neu,1))
        ##model1.add_module('relu4', torch.nn.ReLU())

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

        ## train
        for epoc in range(epocs):
            train(model1, loss, optimizer1, XTrain, YTrain)
            train(model2, loss, optimizer2, XTrain, YTrain)
        ## time for output!
        data = data.float()
        Aggregate1 = predict(model1, data)
        Aggregate2 = predict(model2, data)
        Output_1 = output_anal(Aggregate1)
        Output_2 = output_anal(Aggregate2)
        P_f[h] = array_to_string(Output_1)
        P_f_BN[h] = array_to_string(Output_2)
        a = lempel_ziv_complexity(array_to_string(Output_1))
        b = lempel_ziv_complexity(array_to_string(Output_2))
        Complexity_agg[h] = a
        Complexity_agg_BN[h] = b
        diff_lvc[h] = a-b
        h = h + 1

    return_dict[process_key] = (P_f, P_f_BN, diff_lvc, Complexity_agg, Complexity_agg_BN)

if __name__ == '__main__':
    num_jobs = 3
    times = int(3000)
    total_times = times*num_jobs
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    for i in range(num_jobs):
        p = multiprocessing.Process(target=sampling_process, args=(i, return_dict, times))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()


    P_fs, P_f_BNs, diff_lvcs, Complexity_aggs, Complexity_agg_BN = None, None, None, None, None

    for pair in return_dict.values():
        if P_fs is None and P_f_BNs is None and diff_lvcs is None and Complexity_aggs is None and Complexity_agg_BNs is None:
            P_fs, P_f_BNs, diff_lvcs, Complexity_aggs, Complexity_agg_BNs = pair[0], pair[1], pair[2], pair[3], pair[4]
        else:
            P_fs = torch.cat((P_fs, pair[0]), axis=None)
            P_f_BNs = torch.cat((P_f_BNs, pair[1]), axis=None)
            diff_lvcs = torch.cat((diff_lvcs, pair[2]), axis=None)
            Complexity_aggs = torch.cat((Complexity_aggs, pair[3]), axis=None)
            Complexity_agg_BNs = torch.cat((Complexity_agg_BNs, pair[4]), axis=None)


    # organize output
    Y = get_freq(P_fs)
    Z = get_freq(P_f_BNs)
    A = Complexity_aggs / 1.75
    B = Complexity_agg_BNs / 1.75
    print(A-B)

    # plot
    Min = min(min(Y), min(Z))
    t = max(len(Y), len(Z))
    X = np.arange(t)
    U = (np.log(2) * (2**7) * X) ** (-1)
    plt.plot(Y, label="no BatchNorm")
    plt.plot(Z, label="BatchNorm")
    plt.plot(X, U, label="Zipf's Law")
    plt.xlabel('Rank')
    plt.ylabel('Probability')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc="upper right")
    plt.xlim(1, t)
    plt.ylim(Min, 1)
    plt.show()















