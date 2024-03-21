import numpy as np
import matplotlib.pyplot as plt
# Datos de entrada
x = np.array([[1, 1, 1, 1], [1, 1, 1, -1], [1, 1, -1, 1], [1, 1, -1, -1], [1, -1, 1, 1], [1, -1, 1, -1], [1, -1, -1, 1], [1, -1, -1, -1],
              [-1, 1, 1, 1], [-1, 1, 1, -1], [-1, 1, -1, 1], [-1, 1, -1, -1], [-1, -1, 1, 1], [-1, -1, 1, -1], [-1, -1, -1, 1], [-1, -1, -1, -1]])
w = np.random.rand(5)  # Corrected the syntax error in w initialization
sigma = 0.4  # Learning rate

def andps(s1, s2, s3, s4, w):
    yd = 1  # Initialize yd before the while loop
    y = 0  # Initialize y before the while loop
    while yd != y:
        if s1 == 1 and s2 == 1 and s3 == 1 and s4 == 1:
            yd = 1
        else:
            yd = -1
        h = s1 * w[1] + s2 * w[2] + s3*w[3] + s4*w[4] + w[0]
        if h >= 0:
            y = 1
        else:
            y = -1
        w[0] = w[0] + sigma * (yd - y)
        w[1] = w[1] + sigma * (yd - y) * s1
        w[2] = w[2] + sigma * (yd - y) * s2
        w[3] = w[3] + sigma * (yd - y) * s3
        w[4] = w[4] + sigma * (yd - y) * s4

    return w

for i in range(10000):
    for i in range(16):
        w = andps(x[i][0], x[i][1], x[i][2], x[i][3], w)


print('Pesos obtenidos ',w)

def andver (x, w):
    for i in range(16):
        h = x[i][0] * w[1] + x[i][1] * w[2] + x[i][2] * w[3] + x[i][3] * w[4] + w[0]
        if h >= 0:
            y = 1
        else:
            y = -1
        print('entradas ',x[i], '\n salidas ', y)
     
andver(x, w)
