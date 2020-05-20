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
import pickle
import os.path

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
        return np.log2(len(x)) * (a + b) / 2


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
    return a / d


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
epochs = 50  ## training time
mean = 0.0  ## mean of initialization
scale = 1.0  ## var of initialization

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
for i in range(m_2):
    XTrain[i, :] = data[i, :]
    XTest[i, :] = data[i + m_2, :]

## choose target, need to choose targets of different LVC
targets, YTrains, YTests, TLVS = [], [], [], []

## target of LVC: 7
t = torch.ones(2 ** n)
YTrain = torch.zeros(2 ** (n - 1))
YTest = torch.zeros(2 ** (n - 1))
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
t = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1, 1, 1])
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
t = torch.tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
                  0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0.,
                  0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0.,
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

## targe of LVC：70
t = torch.tensor([0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0.,
                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
                  0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0.,
                  0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                  0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1.,
                  0., 0.])
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
t = torch.tensor([0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1., 1., 0.,
                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0.,
                  0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
                  0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
                  0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
                  0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 1., 0., 0., 0., 0.,
                  1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                  0., 0.])
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
# initialize MC number of models
MC_num = 500  ## number of models
total_MC = 9
error_nonDPs, error_DPs, nonDP_mcom, DP_mcom, sds, sd_BNs = [], [], [], [], [], []

## define loss function
loss = torch.nn.CrossEntropyLoss(size_average=True)


def process(i, MC_num=MC_num):
    model1s, optimizer1s = [], []
    model2s, optimizer2s = [], []
    error_nonDP = torch.zeros(epochs)
    error_DP = torch.zeros(epochs)
    non_DP_mean_complexity = torch.zeros(epochs)
    DP_mean_complexity = torch.zeros(epochs)
    sd = np.zeros(epochs)
    sd_BN = np.zeros(epochs)
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
        optimizer1 = optim.Adam(model1.parameters(), lr=0.1)
        optimizer2 = optim.Adam(model2.parameters(), lr=0.1)

        model1s.append(model1)
        model2s.append(model2)
        optimizer1s.append(optimizer1)
        optimizer2s.append(optimizer2)

    for epoch in range(epochs):
        if epoch % 5 == 0:
            print(f'{datetime.datetime.now()} Epoch {epoch} complete!')
        ## the main program
        L_nonDP = torch.zeros(MC_num)
        L_DP = torch.zeros(MC_num)
        Complexity_agg_DP = torch.zeros(MC_num)
        Complexity_agg = torch.zeros(MC_num)
        for MC in range(MC_num):
            train(model1s[MC], loss, optimizer1s[MC], XTrain, YTrains[i])
            train(model2s[MC], loss, optimizer2s[MC], XTrain, YTrains[i])
            Aggregate1 = predict(model1s[MC], data)
            Aggregate2 = predict(model2s[MC], data)
            Output_1 = Output(Aggregate1)
            Output_2 = Output(Aggregate2)
            L_nonDP[MC] = get_error(model1s[MC], XTrain, YTrains[i], m_2)
            L_DP[MC] = get_error(model2s[MC], XTrain, YTrains[i], m_2)
            a = get_LVComplexity(Output_1)
            b = get_LVComplexity(Output_2)
            Complexity_agg[MC] = a
            Complexity_agg_DP[MC] = b

        error_nonDP[epoch] = torch.mean(L_nonDP)
        error_DP[epoch] = torch.mean(L_DP)
        non_DP_mean_complexity[epoch] = torch.mean(Complexity_agg)
        DP_mean_complexity[epoch] = torch.mean(Complexity_agg_DP)
        a = np.asarray(Complexity_agg)
        b = np.asarray(Complexity_agg_DP)
        sd[epoch] = np.sqrt(np.var(a))
        sd_BN[epoch] = np.sqrt(np.var(b))
        

    return (error_nonDP, error_DP, non_DP_mean_complexity, DP_mean_complexity, sd, sd_BN)


pool = multiprocessing.Pool(14)
tasks = range(total_MC)
result = []
with tqdm.tqdm(total=total_MC, mininterval=5, bar_format='{elapsed}{l_bar}{bar}{r_bar}') as t:
    for i, x in enumerate(pool.imap(process, tasks)):
        t.update()
        result.append(x)
pool.close()
pool.join()

for output in result:
    k = output
    error_nonDPs.append(k[0])
    error_DPs.append(k[1])
    nonDP_mcom.append(k[2])
    DP_mcom.append(k[3])
    sds.append(k[4])
    sd_BNs.append(k[5])

k = 'D:/pickles'
file_path1 = os.path.join(k, 'error_nonDPs.pkl')
outfile1 = open(file_path1, 'wb')
pickle.dump(error_nonDPs, outfile1)
outfile1.close()

file_path2 = os.path.join(k, 'error_DPs.pkl')
outfile2 = open(file_path2, 'wb')
pickle.dump(error_DPs, outfile2)
outfile2.close()

file_path3 = os.path.join(k, 'nonDP_mcom.pkl')
outfile3 = open(file_path3, 'wb')
pickle.dump(nonDP_mcom, outfile3)
outfile3.close()

file_path4 = os.path.join(k, 'DP_mcom.pkl')
outfile4 = open(file_path4, 'wb')
pickle.dump(DP_mcom, outfile4)
outfile4.close()

file_path5 = os.path.join(k, 'sds.pkl')
outfile5 = open(file_path5, 'wb')
pickle.dump(sds, outfile5)
outfile5.close()

file_path6 = os.path.join(k, 'sd_BNs.pkl')
outfile6 = open(file_path6, 'wb')
pickle.dump(sd_BNs, outfile6)
outfile6.close()


# plot
fig, ax = plt.subplots(nrows=9, ncols=2, figsize=(15, 15), constrained_layout=True)
X = np.arange(epochs)
for h in range(9):
    a = np.asarray(nonDP_mcom[h]) + sds[h]
    b = np.asarray(nonDP_mcom[h]) - sds[h]
    c = np.asarray(DP_mcom[h]) + sd_BNs[h]
    d = np.asarray(DP_mcom[h]) - sd_BNs[h]
    ax[h, 0].plot(X, error_nonDPs[h], label="Error, NN")
    ax[h, 0].plot(X, error_DPs[h], label="Error, NN+BN")
    ax[h, 1].plot(X, )
    ax[h, 1].fill_between(X, a, b, color='blue', alpha=0.4, label="Mean complexity, NN")
    ax[h, 1].fill_between(X, c, d, color='red', alpha=0.4, label="Mean complexity, NN+BN")
    ax[h, 0].legend(loc="upper right")
    ax[h, 1].legend(loc="upper right")
    ax[h, 0].set_xlabel('Training Curve' + '' + 'tcomplexity =' + '' + str(TLVS[h]))
    ax[h, 1].set_xlabel('Mean Complexity' + '' + 'tcomplexity =' + '' + str(TLVS[h]))
    ax[h, 0].set_ylabel('Training Error')
    ax[h, 1].set_ylabel('Complexity')
fig.show()

