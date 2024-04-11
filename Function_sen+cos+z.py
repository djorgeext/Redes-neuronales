# Multicap perceptron to compute XOR
import numpy as np
import matplotlib.pyplot as plt

# Activation function tanh
def tanh(x):
    return np.tanh(x)

# Derivative of the tanh activation function
def tanh_derivative(x):
    return 1.0 - x**2

# function sigmoid
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Derivative of the sigmoid activation function
def sigmoid_derivative(x):
    return x*(1-x)

# Input data
x = np.sin(np.random.uniform(0, 2*np.pi, 1000))
y = np.cos(np.random.uniform(0, 2*np.pi, 1000))
z = np.random.uniform(-1, 1, 1000)

s = np.array([x, y, z]).T
yd = np.array([np.sin(x) + np.cos(y) + z]).T

# Weights initialization
w1 = np.random.rand(3, 8)
w2 = np.random.rand(8, 4)
w3 = np.random.rand(4, 1)
# Bias initialization
b1 = np.random.rand(1, 8)
b2 = np.random.rand(1, 4)
b3 = np.random.rand(1, 1)

# Learning rate
eta = 0.1

# Number of epochs
epochs = 1000

# Error storage
errors = []

# Training
def train(inp,w1,w2,w3,sald,eta,b1,b2,b3):
    # Forward propagation
    z1 = np.dot(inp, w1) + b1
    a1 = tanh(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = tanh(z2)
    z3 = np.dot(a2, w3) + b3
    a3 = tanh(z3)

    # Error calculation
    error = 0.5*np.sum((sald - a3)**2)
    errors.append(error)

    # Backward propagation
    delta3 = (sald - a3)
    delta2 = delta3.dot(w3.T) * tanh_derivative(a2)
    delta1 = delta2.dot(w2.T) * tanh_derivative(a1)

    # Weights update
    w3 += a2.T.dot(delta3) * eta
    w2 += a1.T.dot(delta2) * eta
    w1 += inp.T.dot(delta1) * eta

# Training loop
for i in range(epochs):
    permutation_indices = np.random.permutation(len(yd))
    # Aplicar la misma permutación a ambos arreglos
    s = s[permutation_indices]
    yd = yd[permutation_indices]
    for j in range(100):
        train(s[j:j+10], w1, w2, w3, yd[j:j+10], eta, b1, b2, b3)
        
# Plotting the error
errors = np.array(errors).flatten()
plt.plot(errors)
plt.show()

# Test

""" for i in range(len(s)):
    z1 = np.dot(s[i].reshape(1, 3), w1) + b1
    a1 = tanh(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = tanh(z2)
    z3 = np.dot(a2, w3) + b3
    a3 = tanh(z3)
    
    
    print('Input:', s[i], 'Output:', a3) """
