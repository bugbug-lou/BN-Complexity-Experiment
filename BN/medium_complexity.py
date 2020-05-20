# randomly generate YTrain, investigate which complexity range is more, then investigate which target should be ...
# target complexity's relationship with YTrain?

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
import tqdm
import time
import scipy.stats as stats
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


## some parameters
n = 7  ## dimension of input data, user-defined
m = 2 ** n  ## number of data points
m_2 = 2 ** (n - 1)
m_3 = 2 ** (n - 2)
predict_threshold = 0.001  ## training accuracy threshold
neu = 40
mean = 0.0  ## mean of initialization
scale = 1.0  # STD of initialization

## data: 7 * 128
data = np.zeros([m, n], dtype=np.float32)
for i in range(m):
    bin = np.binary_repr(i, n)
    a = np.array(list(bin), dtype=int)
    data[i, :] = a
data = torch.from_numpy(data)

## Training set:
XTrain = torch.zeros([m_2, n])
XTest = torch.zeros([m_2, n])
for i in range(m_2):
    XTrain[i, :] = data[i, :]
    XTest[i, :] = data[i + m_2, :]

# set probability:
l = 5
total_MC = 10 ** (l)
ps = [0.05, 0.1, 0.15, 0.25, 0.4]
MCs = [(1 * total_MC) / 5, (1 * total_MC) / 5, (1 * total_MC) / 5, (1 * total_MC) / 5, (1 * total_MC) / 5]
dic1 = {}  # a dictionary for NN output
dic2 = {}  # a dictionary for NNdp output
dic3 = {}

## define loss function
loss = torch.nn.CrossEntropyLoss(size_average=True)

# generate targets
def process(MC):
    k = int((MC - MC % (total_MC/5)) / (total_MC/5))
    p = ps[k]
    zero_ones = np.array([0, 1])  ## things to choose from
    dis = np.array([p, 1 - p])  ## probability input for choose function
    t = np.zeros(m_2)
    for k in range(m_2):
        t[k] = np.random.choice(zero_ones, p=dis)
    t = torch.from_numpy(t)
    t = t.long()
    t_c = get_LVComplexity(t)
    h = (t_c - t_c % 5) / 5
    if 2 <= h:
        YTrain = t
        # then train NN, BN models on the target
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
        model2.add_module('bn1', torch.nn.BatchNorm1d(neu, momentum=0.1))
        model2.add_module('relu1', torch.nn.ReLU())
        model2.add_module('FC2', torch.nn.Linear(neu, neu))
        model2.add_module('bn2', torch.nn.BatchNorm1d(neu, momentum=0.1))
        model2.add_module('relu2', torch.nn.ReLU())
        model2.add_module('FC3', torch.nn.Linear(neu, 2))

        with torch.no_grad():
            model2.FC1.weight = torch.nn.Parameter(model1.FC1.weight.clone().detach())
            model2.FC2.weight = torch.nn.Parameter(model1.FC2.weight.clone().detach())
            model2.FC3.weight = torch.nn.Parameter(model1.FC3.weight.clone().detach())

        # define optimizer
        optimizer1 = optim.Adam(model1.parameters(), lr=0.1)
        optimizer2 = optim.Adam(model2.parameters(), lr=0.1)

        # train until convergence
        pr1 = 1
        pr2 = 1
        timeout = time.time() + 25
        while pr1 > predict_threshold and time.time() < timeout:
            train(model1, loss, optimizer1, XTrain, YTrain)
            pr1 = get_error(model1, XTrain, YTrain, 2 ** (n - 1))
        while pr2 > predict_threshold and time.time() < timeout:
            train(model2, loss, optimizer2, XTrain, YTrain)
            pr2 = get_error(model2, XTrain, YTrain, 2 ** (n - 1))

        # prediction
        Aggregate1 = predict(model1, data)
        Aggregate2 = predict(model2, data)
        Output_NN = Output(Aggregate1)
        Output_BN = Output(Aggregate2)
        Out_NN = get_LVComplexity(Output_NN)
        Out_BN = get_LVComplexity(Output_BN)
        a = Out_NN
        b = Out_BN
        c = Out_NN - Out_BN

        del model1
        del model2

        return (h, a, b, c)
    else:
        return (h, 0, 0, 0)


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
    h, a, b, c = output
    if int(h) >= 2:
        if h in dic1.keys():
            dic1[h].append(a)
            dic2[h].append(b)
            dic3[h].append(c)
        else:
            dic1[h] = [a]
            dic2[h] = [b]
            dic3[h] = [c]


#alpha = 0.1 # alpha value for testing the Null Hypothesis: mean NN = mean BN
t_stats1, t_stats2 = [], []
p_val1, p_val2 = [], []
test_results = [] # output test results
# T-statistics Test:
for h in sorted(dic1.keys()):
    h = int(h)
    a = [0]*len(dic1[h])
    t1 = stats.ttest_ind(dic1[h], dic2[h], axis=0, equal_var=True)[0]
    t2 = stats.ttest_ind(dic3[h], a, axis=0, equal_var=True)[0]
    p1 = stats.ttest_ind(dic1[h], dic2[h], axis=0, equal_var=True)[1]
    p2 = stats.ttest_ind(dic3[h], a, axis=0, equal_var=True)[1]
    t_stats1.append(t1)
    t_stats2.append(t2)
    p_val1.append(p1)
    p_val2.append(p2)

# make confidence interval
a, b, c = [], [], []
z = 2.576 # z-score corresponding to ci
for h in sorted(dic1.keys()):
    count = 0
    k = np.asarray(dic1[h]) > np.asarray(dic2[h])
    for i in range(len(k)):
        if k[i] == True:
            count = count + 1
    l = 0
    s = np.asarray(dic1[h]) < np.asarray(dic2[h])
    for i in range(len(s)):
        if s[i] == True:
            l = l + 1
    p1 = count/len(k)
    p2 = l/len(s)
    a.append((p1 - z * np.sqrt(p1*(1-p1)/len(k)), p1 + z * np.sqrt(p1*(1-p1)/len(k))))
    b.append((p2 - z * np.sqrt(p2*(1-p2)/len(k)), p2 + z * np.sqrt(p2*(1-p2)/len(k))))
    if p2 - z * np.sqrt(p2*(1-p2)/len(k)) > p1 + z * np.sqrt(p1*(1-p1)/len(k)):
        c.append('BN Larger Convincing')
    else:
        if p1 - z * np.sqrt(p1*(1-p1)/len(k)) > p2 + z * np.sqrt(p2*(1-p2)/len(k)):
            c.append('NN Larger Convincing')
        else:
            c.append('ambiguous')

k = 'D:/pickles'
file_path1 = os.path.join(k, 'outfile1.pkl')
outfile1 = open(file_path1, 'wb')
pickle.dump(dic1, outfile1)
outfile1.close()

file_path2 = os.path.join(k, 'outfile2.pkl')
outfile2 = open(file_path2, 'wb')
pickle.dump(dic2, outfile2)
outfile2.close()

file_path3 = os.path.join(k, 'outfile3.pkl')
outfile3 = open(file_path3, 'wb')
pickle.dump(dic3, outfile3)
outfile3.close()

file_path4 = os.path.join(k, 'outfile4.pkl')
outfile4 = open(file_path4, 'wb')
pickle.dump(a, outfile4)
outfile4.close()

file_path5 = os.path.join(k, 'outfile5.pkl')
outfile5 = open(file_path5, 'wb')
pickle.dump(p_val1, outfile5)
outfile5.close()

file_path6 = os.path.join(k, 'outfile6.pkl')
outfile6 = open(file_path6, 'wb')
pickle.dump(p_val2, outfile6)
outfile6.close()

file_path7 = os.path.join(k, 'outfile7.pkl')
outfile7 = open(file_path7, 'wb')
pickle.dump(b, outfile7)
outfile7.close()

file_path8= os.path.join(k, 'outfile8.pkl')
outfile8 = open(file_path8, 'wb')
pickle.dump(t_stats1, outfile8)
outfile8.close()

file_path9= os.path.join(k, 'outfile9.pkl')
outfile9 = open(file_path9, 'wb')
pickle.dump(t_stats2, outfile9)
outfile9.close()

file_path0= os.path.join(k, 'outfile0.pkl')
outfile0 = open(file_path0, 'wb')
pickle.dump(t_stats2, outfile0)
outfile0.close()
