import numpy as np
import matplotlib.pyplot as plt
# Datos de entrada
x = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
w = np.random.rand(3)  # Corrected the syntax error in w initialization
sigma = 0.1  # Learning rate
vector = np.linspace(-1.2, 1.2, 200)

def orps(s1, s2, w):
    yd = 1  # Initialize yd before the while loop
    y = 0  # Initialize y before the while loop
    while yd != y:
        if s1 == 1 or s2 == 1:
            yd = 1
        else:
            yd = -1
        h = s1 * w[1] + s2 * w[2] + w[0]
        if h >= 0:
            y = 1
        else:
            y = -1
        w[0] = w[0] + sigma * (yd - y)
        w[1] = w[1] + sigma * (yd - y) * s1
        w[2] = w[2] + sigma * (yd - y) * s2

    return w

for i in range(10000):
    for i in range(4):
        w = orps(x[i][0], x[i][1], w)


print('Pesos obtenidos ',w)

def orver (x, w):
    for i in range(4):
        h = x[i][0] * w[1] + x[i][1] * w[2] + w[0]
        if h >= 0:
            y = 1
        else:
            y = -1
        plt.plot(x[i][0], x[i][1], 'ro' if y == 1 else 'bo', markersize=15)
        
        print('entradas ',x[i], '\n salidas ', y)
    rdiscriminante = -w[0]/w[2] - w[1]/w[2]*vector
    plt.plot(vector, rdiscriminante, label='Discriminante', linewidth=4)
    plt.tick_params(axis='both', which='major', labelsize=20)

    plt.show()
orver(x, w)

