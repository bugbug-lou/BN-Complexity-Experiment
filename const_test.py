## new experiment, this gives us a clue how complexity related, when 0-1 strings are largely generated randomly
Times= 5
haha = 100
Out = np.zeros(haha)
while (Times < haha):
    t = 0
    times1 = 30
    n = Times ##length determination
    p = 0.5 ##probabilty for choosing 0
    zero_ones = np.array([0,1]) ## things to choose from
    dis = np.array([p, 1-p]) ## probability input for choose function
    Ori = np.zeros(2*n)
    for k in range(2*n):
        Ori[k] = np.random.choice(zero_ones,p = dis)
    m = get_LVComplexity(Ori)
    Rand = list(range(2*n))
    LV = np.zeros(times1)
    while (t<times1):
        random.shuffle(Rand)
        Sect = np.zeros(n)
        for i in range(n):
            Sect[i] = Ori[Rand[i]]
        LV[t] = get_LVComplexity(Sect)
        t = t + 1
    i = get_max_freq(LV)
    Out[Times] = m/i
    Times = Times + 1
np.mean(Out)