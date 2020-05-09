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
l = 4
total_MC = 10 ** (l)
ps = [0.05, 0.2, 0.3, 0.4, 0.5]
MCs = [(1 * total_MC) / 5, (1 * total_MC) / 5, (1 * total_MC) / 5, (1 * total_MC) / 5, (1 * total_MC) / 5]
dic1 = {}  # a dictionary for NN output
dic2 = {}  # a dictionary for NNdp output

## define loss function
loss = torch.nn.CrossEntropyLoss(size_average=True)

# generate targets
def process(MC):
    k = int((MC - MC % (total_MC/5)) / (total_MC/5))
    p = ps[k]
    zero_ones = np.array([0, 1])  ## things to choose from
    dis = np.array([p, 1 - p])  ## probability input for choose function
    t = np.zeros(m)
    for k in range(m):
        t[k] = np.random.choice(zero_ones, p=dis)
    t = torch.from_numpy(t)
    t = t.long()
    t_c = get_LVComplexity(t)
    h = (t_c - t_c % 10) / 10
    YTrain = torch.zeros(m_2)
    YTest = torch.zeros(m_2)
    for j in range(m_2):
        YTrain[j] = t[j]
        YTest[j] = t[j + m_2]
    YTrain = YTrain.long()
    YTest = YTest.long()

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
    while pr1 > predict_threshold:
        train(model1, loss, optimizer1, XTrain, YTrain)
        pr1 = get_error(model1, XTrain, YTrain, 2 ** (n - 1))
    while pr2 > predict_threshold:
        train(model2, loss, optimizer2, XTrain, YTrain)
        pr2 = get_error(model2, XTrain, YTrain, 2 ** (n - 1))

    # prediction
    Aggregate1 = predict(model1, data)
    Aggregate2 = predict(model2, data)
    Output_NN = Output(Aggregate1)
    Output_BN = Output(Aggregate2)
    Out_NN = get_LVComplexity(Output_NN)
    Out_BN = get_LVComplexity(Output_BN)
    a = (Out_NN - t_c) / t_c
    b = (Out_BN - t_c) / t_c

    del model1
    del model2

    return (h, a, b)


pool = multiprocessing.Pool(9)
tasks = range(total_MC)
result = []
with tqdm.tqdm(total=total_MC, mininterval=5, bar_format='{elapsed}{l_bar}{bar}{r_bar}') as t:
    for i, x in enumerate(pool.imap(process, tasks)):
        t.update()
        result.append(x)
pool.close()
pool.join()

for output in result:
    h, a, b = output
    if h <= 10:
        if h in dic1.keys():
            dic1[h].append(a)
        else:
            dic1[h] = [a]
        if h in dic2.keys():
            dic2[h].append(a)
        else:
            dic2[h] = [a]


# plot barcode based on the outputs
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = plt.subplots(nrows=2, ncols=5, figsize=(25, 15))
ax = (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10)
colors = ['lightblue', 'darkviolet']
labels = ['NN', 'NN+BN']
i = 0
for h in sorted(dic1.keys()):
    h = int(h)
    all_data = [dic1[h], dic2[h]]
    t = ax[i].boxplot(all_data, vert=True, patch_artist=True, labels=labels)
    for patch, color in zip(t['boxes'], colors):
        patch.set_facecolor(color)
    ax[i].axhline(y=0, color='black', linestyle='dashed')
    ax[i].set_title('complexity range' + ' ' + str(10 * h) + '-' + str(
        10 * (h + 1)))
    i = i + 1
fig.show()
