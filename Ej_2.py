import numpy as np
import matplotlib.pyplot as plt

x=5
m=2
eta = 0.2
w = np.random.rand(x+1)
vector = np.linspace(-1.2, 1.2, 200)
# funcion para definir entradas y salidas
def entrada_salida(x,m):
    s = np.random.normal(0, 0.5, [x, m])
    yd = np.zeros(x)
    for i in range(x):
        if all(s[i, :] >= 0):
            yd[i] = 1
        else:
            yd[i] = -1
    return s, yd

def perceptron(inp, eta, yd, w):
    y = 0
    n_aciertos = 0
    while y != yd:
        h = w[0] + np.dot(inp, w[1:])
        if h >= 0:
            y = 1
        else:
            y = -1
        w[0] = w[0] + eta * (yd - y)
        w[1:] = w[1:] + eta * (yd - y) * inp
        n_aciertos += 1
    return w, n_aciertos

s, yd = entrada_salida(x)
for i in range(1000):
    c_ap = 0
    for i in range(x):
        w, temp = perceptron(s[i], eta, yd[i], w)
        c_ap += temp

print(c_ap)


# def andver (s, w):
#     for i in range(x):
#         h = w[0] + np.dot(s[i], w[1:])
#         if h >= 0:
#             y = 1
#         else:
#             y = -1
#         plt.plot(s[i][0], s[i][1], 'ro' if y == 1 else 'bo')
        
#         print('entradas ',s[i], '\n salidas ', y)
    
# andver(s, w)