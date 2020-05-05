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

## some parameters
n = 7 ## dimension of input data, user-defined
m = 2 ** n  ## number of data points
m_2 = 2 ** (n - 1)
m_3 = 2 ** (n - 2)
predict_threshold = 0.001 ## training accuracy threshold
layer_num = 3  ## number of layers of the neural network
neu = 40  ## neurons per layer
mod_num = 1000  ## numbers of models used for each example
mean = 0.0  ## mean of initialization
scale = 1.0  # STD of initialization

## data: 7 * 128
data = np.zeros([2 ** n, n], dtype=np.float32)
for i in range(2 ** n):
    bin = np.binary_repr(i, n)
    a = np.array(list(bin), dtype=int)
    data[i, :] = a
data = torch.from_numpy(data)

## Training set:
XTrain = torch.zeros([2 ** (n-1), n])
XTest = torch.zeros([2 ** (n-1), n])
for i in range(2 ** (n-1)):
    XTrain[i,:] = data[i,:]
    XTest[i, :] = data[i + m_2, :]

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

## outputs
LVC_outputs, LVC_output_BNs, GE_outputs, GE_output_BNs = [], [], [], []


## define loss function
loss = torch.nn.CrossEntropyLoss(size_average=True)

## train BN models and non-BN models based on different targets
total_MC = 9
def process(MC):
    # print('MC sample ' + str(MC) + f'/{total_MC} starts!')
    LVC = np.zeros(mod_num)
    LVC_BN = np.zeros(mod_num)
    GE = np.zeros(mod_num)
    GE_BN = np.zeros(mod_num)

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
        optimizer1 = optim.Adam(model1.parameters(), lr=0.2)
        optimizer2 = optim.Adam(model2.parameters(), lr=0.2)

        # train until convergence
        pr1 = 1
        pr2 = 1
        while pr1 > predict_threshold:
            train(model1, loss, optimizer1, XTrain, YTrains[MC])
            pr1 = get_error(model1, XTrain, YTrains[MC], 2 ** (n-1))
        while pr2 > predict_threshold:
            train(model2, loss, optimizer2, XTrain, YTrains[MC])
            pr2 = get_error(model2, XTrain, YTrains[MC], 2 ** (n-1))

        # prediction
        Aggregate1 = predict(model1, data)
        Aggregate2 = predict(model2, data)
        Output_1 = Output(Aggregate1)
        Output_2 = Output(Aggregate2)
        GE[i] = get_error(model1, XTest, YTests[MC], 2 ** (n-1))/2
        GE_BN[i] = get_error(model2, XTest, YTests[MC], 2 ** (n-1))/2
        LVC[i] = get_LVComplexity(Output_1)
        LVC_BN[i] = get_LVComplexity(Output_2)

        del model1
        del model2


    return (LVC, LVC_BN, GE, GE_BN)
    # LVC_outputs.append(LVC), LVC_output_BNs.append(LVC_BN), GE_outputs.append(GE), GE_output_BNs.append(GE_BN)

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
    LVC, LVC_BN, GE, GE_BN = output
    LVC_outputs.append(LVC)
    LVC_output_BNs.append(LVC_BN)
    GE_outputs.append(GE)
    GE_output_BNs.append(GE_BN)

# produce a plot concerning output function complexities
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
ax = (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9)
for h in range(9):
    ax[h].scatter(LVC_outputs[h], GE_outputs[h], label='NN', c='green', alpha=0.5)
    ax[h].scatter(LVC_output_DPs[h], GE_output_DPs[h], label='NN+DP', c='red', alpha=0.5)
    # ax[h].scatter(LVC_output_UEs[h], GE_output_UEs[h], label='Unbiased Estimator', c='blue', alpha=0.5)
    ax[h].legend(loc="upper right")
    ax[h].set_xlabel(f'Target Complexity: {TLVS[h]}')
    ax[h].set_ylabel('Generalization/Test Error')

fig.show()

# histoplot
fig, ax = plt.subplots(nrows=9, ncols=2, figsize=(15, 15),constrained_layout=True)
for h in range(9):
    
    ax[h,0].hist(GE_outputs[h], bins = 20, range = (0.0,1.0), facecolor='blue', alpha=0.75, label='NN')
    ax[h,0].hist(GE_output_DPs[h], bins = 20, range = (0.0,1.0), facecolor='red', alpha=0.75, label='NN + DP')
    ax[h, 1].hist(LVC_outputs[h],bins = 20, range = (0, 140), facecolor='blue', alpha=0.75, label='NN')
    ax[h, 1].hist(LVC_output_DPs[h], bins = 20, range = (0, 140), facecolor='red', alpha=0.75, label='NN + DP')
    ax[h, 1].axvline(TLVS[h], 0, mod_num, color = 'black', label = 'target complexity', linestyle = 'dashed')
    ax[h, 0].legend(loc="upper right")
    ax[h, 1].legend(loc="upper right")
    ax[h, 0].set_xlabel('Error rate histplot' + '' + 'tcomplexity =' + '' + str(TLVS[h]))
    ax[h, 1].set_xlabel('Complexity histplot' + '' + 'tcomplexity =' + '' + str(TLVS[h]))
    ax[h, 0].set_ylabel('Error Rates')
    ax[h, 1].set_ylabel('Complexity')
fig.show()
