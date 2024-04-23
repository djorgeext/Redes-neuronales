import numpy as np
import matplotlib.pyplot as plt

def signo(h):
    if h >= 0:
        s = 1
    else:
        s = -1
    return s

def perceptron_train(x, z, eta):
    W = np.random.random(int(np.size(x)/len(x))+1)
    for i in range(len(x)*2):
        for index, sample in enumerate(x):
            sample = np.append(sample, 1)
            y = signo(np.dot(sample, W))
            while y != z[index]:
                W += eta*(z[index]-y)*sample
                y = signo(np.dot(sample, W))
        E = sum([(z[index] - signo(np.dot(np.append(sample, 1), W)))**2 for index, sample in enumerate(x)])
    return E

C = {}
for N in range(5,25,5):
    C[N] = {}
    P = []
    Y = []
    Error = []
    for p in range(2,82,2):
        AP = 0
        Error_ = []
        for i in range(100):
            x = [np.where(np.random.randint(2, size=N) == 0, -1, 1) for n in range(p)]
            y = np.where(np.random.randint(2, size=p) == 0, -1, 1)
            eta = 0.25
            if perceptron_train(x, y, eta) == 0:
                AP += 1
            else:
                Error_.append(perceptron_train(x, y, eta))
        P.append(p)
        Y.append(AP/100)
        Error.append(np.mean(np.array(Error_)))
    C[N]['AP/REP'] = Y
    C[N]['P'] = P
    C[N]['Error'] = Error

fig, axes = plt.subplots(ncols=2, nrows=2, constrained_layout=True)

axes[0, 0].set_title('n = 5')
axes[0, 0].set_ylabel('Tasa de aprendizaje')
axes[0, 0].set_xlabel('Número de patrones enseñados')
axes[0, 0].plot(C[5]['P'], C[5]['AP/REP'], '.--k')
axes[0, 0].grid()

axes[0, 1].set_title('n = 10')
axes[0, 1].set_ylabel('Tasa de aprendizaje')
axes[0, 1].set_xlabel('Número de patrones enseñados')
axes[0, 1].plot(C[10]['P'], C[10]['AP/REP'], '.--k')
axes[0, 1].grid()

axes[1, 0].set_title('n = 15')
axes[1, 0].set_ylabel('Tasa de aprendizaje')
axes[1, 0].set_xlabel('Número de patrones enseñados')
axes[1, 0].plot(C[15]['P'], C[15]['AP/REP'], '.--k')
axes[1, 0].grid()

axes[1, 1].set_title('n = 20')
axes[1, 1].set_ylabel('Tasa de aprendizaje')
axes[1, 1].set_xlabel('Número de patrones enseñados')
axes[1, 1].plot(C[20]['P'], C[20]['AP/REP'], '.--k')
axes[1, 1].grid()

fig.suptitle('Tasa de aprendizaje vs Cantidad de patrones enseñados',fontweight ="bold")
plt.show()