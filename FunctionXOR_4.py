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
s = np.array([[-1, -1, -1, -1],
              [-1, -1, -1, 1],
              [-1, -1, 1, -1],
              [-1, -1, 1, 1],
              [-1, 1, -1, -1],
              [-1, 1, -1, 1],
              [-1, 1, 1, -1],
              [-1, 1, 1, 1],
              [1, -1, -1, -1],
              [1, -1, -1, 1],
              [1, -1, 1, -1],
              [1, -1, 1, 1],
              [1, 1, -1, -1],
              [1, 1, -1, 1],
              [1, 1, 1, -1],
              [1, 1, 1, 1]])

# # Output data
yd = np.array([[-1], [1], [1], [-1], [1], [-1], [-1], [1], [1], [-1], [-1], [1], [-1], [1], [1], [-1]])

# Weights initialization
w1 = np.random.rand(4, 16)
w2 = np.random.rand(16, 1)
#w3 = np.random.rand(8, 1)
# Bias initialization
b1 = np.random.rand(1, 16)
b2 = np.random.rand(1, 1)
#b3 = np.random.rand(1, 1)

# Learning rate
eta = 0.01

# Number of epochs
epochs = 30000

# Error storage
errors = []

# Training
def train(inp,w1,w2,sald,eta,b1,b2):
    # Forward propagation
    z1 = np.dot(inp, w1) + b1
    a1 = tanh(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = tanh(z2)
    
    # Error calculation
    error = 0.5*np.sum((sald - a2)**2)
    errors.append(error)

    # Backward propagation
    delta2 = (sald - a2) * tanh_derivative(a2)
    #delta2 = delta3.dot(w3.T) * tanh_derivative(a2)
    delta1 = delta2.dot(w2.T) * tanh_derivative(a1)

    # Weights update
    #w3 += a2.T.dot(delta3) * eta
    w2 += a1.T.dot(delta2) * eta
    w1 += inp.T.dot(delta1) * eta

# Training loop
for i in range(epochs):
    train(s, w1, w2, yd, eta, b1, b2)
        
# Plotting the error
errors = np.array(errors).flatten()
plt.plot(errors)
plt.show()

# Test

for i in range(len(s)):
    z1 = np.dot(s[i].reshape(1, 4), w1) + b1
    a1 = tanh(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = tanh(z2)
    #z3 = np.dot(a2, w3) + b3
    #a3 = tanh(z3)
    
    print('Input:', s[i], 'Output:', a2)
