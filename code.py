# import
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

# functions
def array_to_string(x):
    y = ''
    for l in x:
        y += str(int(l))
    return y

def analyze(x):
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
    predicts = analyze(logits)
    k = predicts - labels
    a = torch.sum(torch.abs(k))
    return a / d


def predict(model, inputs):
    model.eval()
    inputs = Variable(inputs, requires_grad=False)
    logits = model.forward(inputs)
    return logits

# some parameters
n = 7 # dimension of input data, user-defined
m = 2 ** n  # number of data points
m_2 = 2 ** (n - 1)
m_3 = 2 ** (n - 2)
predict_threshold = 0.001 # training accuracy threshold
MC_num = 10 ** 3 # number of random initialization of models
neu = 40  # neurons per layer
mean = 0.0  # mean of initialization
scale = 10  # STD of initialization

# generate train data and test data
# data: 7 * 128
data = np.zeros([m, n], dtype=np.float32)
for i in range(m):
    bin = np.binary_repr(i, n)
    a = np.array(list(bin), dtype=int)
    data[i, :] = a
data = torch.from_numpy(data)

# Test Set and Train Set
XTrain = torch.zeros([m_2, n])
XTest = torch.zeros([m_2, n])
YTrain = torch.zeros(m_2)
YTest = torch.zeros(m_2)
for i in range(m_2):
    XTrain[i,:] = data[2*i,:]
    XTest[i, :] = data[2 * i + 1, :]

target = torch.zeros(m)
for i in range(m_3):
    target[i] = 1
    target[i + m_2] = 1
    target[i + m_3] = 0
    target[i + m_3 + m_2] = 0
for i in range(m_2):
    YTrain[i] = target[2 * i]
    YTest[i] = target[2 * i + 1]
YTrain = YTrain.long()
YTest = YTest.long()

# initialize 10^6 models and train to 100% accuracy
outputs = {}

# we use MSE Loss:
# loss = torch.nn.MSELoss(size_average=True)
loss = torch.nn.CrossEntropyLoss(size_average=True)

for i in range(MC_num):
    if i%100 == 0:
        print(str(i) + 'aha!')
    model1 = torch.nn.Sequential()
    model1.add_module('FC1', torch.nn.Linear(n, neu))
    model1.add_module('FC2', torch.nn.Linear(neu, 2))
    with torch.no_grad():
        torch.nn.init.normal_(model1.FC1.weight, mean=mean, std=scale)
        torch.nn.init.normal_(model1.FC2.weight, mean=mean, std=scale)

    model2 = torch.nn.Sequential()
    model2.add_module('FC1', torch.nn.Linear(n, neu))
    model2.add_module('bn1', torch.nn.BatchNorm1d(neu, momentum=0.1))
    model2.add_module('FC2', torch.nn.Linear(neu, 2))
    with torch.no_grad():
        torch.nn.init.normal_(model2.FC1.weight, mean=mean, std=scale)
        torch.nn.init.normal_(model2.FC2.weight, mean=mean, std=scale)

    # define optimizer
    optimizer1 = optim.SGD(model1.parameters(), lr=0.02)
    optimizer2 = optim.SGD(model2.parameters(), lr=0.02)

    # train until convergence:
    pr1 = pr2 = 1
    while pr1 > predict_threshold:
        train(model1, loss, optimizer1, XTrain, YTrain)
        pr1 = get_error(model1, XTrain, YTrain, m_2)
    while pr2 > predict_threshold:
        train(model2, loss, optimizer2, XTrain, YTrain)
        pr2 = get_error(model2, XTrain, YTrain, m_2)

    # record test set:
    a = array_to_string(analyze(predict(model1, XTest)))
    b = array_to_string(analyze(predict(model2, XTest)))
    c = get_error(model1, XTest, YTest, 1)
    d = get_error(model2, XTest, YTest, 1)
    if a in outputs.keys():
        outputs[a] = (outputs[a][0] + float(1/MC_num), outputs[a][1], c)
    else:
        outputs[a] = (1/MC_num, 0, c)
    if b in outputs.keys():
        outputs[b] = (outputs[b][0], outputs[b][1] + float(1/MC_num), d)
    else:
        outputs[b] = (0, float(1/MC_num), d)

    del model1
    del model2

# plot
color = ['darkgreen', 'yellowgreen', 'khaki', 'yellow', 'gold', 'orange', 'orangered', 'red', 'red', 'red', 'red']
Z = torch.arange(0,1,0.001)

for i in range(11):
    if i < 10:
        l = [x for (x, y, z) in list(outputs.values()) if z==i]
        p = [y for (x, y, z) in list(outputs.values()) if z==i]
        plt.scatter(p, l, color=color[i], alpha=1.0, label=str(i))
    else:
        l = [x for (x, y, z) in list(outputs.values()) if z >= 10]
        p = [y for (x, y, z) in list(outputs.values()) if z >= 10]
        plt.scatter(p, l, color=color[i], alpha=1.0, label='>= 10')

plt.plot(Z, Z, color = 'blue')
plt.xlim(1/MC_num, 1)
plt.ylim(1/MC_num, 1)
plt.xlabel('P(f) SGD C_E BN')
plt.ylabel('P(f) SGD C_E NN')
plt.legend(loc="upper right")
plt.xscale('log')
plt.yscale('log')
plt.show()






