# Multicap perceptron to compute XOR
import numpy as np
import matplotlib.pyplot as plt

# Activation function ReLU
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x <= 0, 0, 1)

# Activation function tanh
def tanh(x):
    return np.tanh(x)


# Derivative of the tanh activation function
def tanh_derivative(x):
    return 1.0 - x ** 2


# function sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivative of the sigmoid activation function
def sigmoid_derivative(x):
    return x * (1 - x)


# Input data
x = np.sin(np.random.uniform(0, 2 * np.pi, 1000))
y = np.cos(np.random.uniform(0, 2 * np.pi, 1000))
z = np.random.uniform(-1, 1, 1000)

s = np.array([x, y, z]).T
yd = np.array([np.sin(x) + np.cos(y) + z]).T
yd = yd / 3

# Weights initialization
w1 = np.random.rand(3, 8)
w2 = np.random.rand(8, 6)
w3 = np.random.rand(6, 1)
# Bias initialization
b1 = np.random.rand(1, 8)
b2 = np.random.rand(1, 6)
b3 = np.random.rand(1, 1)

# Learning rate
eta = 0.1

# Number of epochs
epochs = 1000
packages_size = 2
packages_iterations = len(s) // packages_size

# Error storage
errors = []


# Training
def train(inp, w1, w2, w3, sald, eta, b1, b2, b3, errors):
    # Forward propagation
    z1 = np.dot(inp, w1) + b1
    a1 = tanh(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = tanh(z2)
    z3 = np.dot(a2, w3) + b3
    a3 = tanh(z3)

    # Error calculation
    error = 0.5 * np.sum((sald - a3) ** 2)
    errors.append(error)

    # Backward propagation
    delta3 = (sald - a3) * tanh_derivative(a3)
    delta2 = delta3.dot(w3.T) * tanh_derivative(a2)
    delta1 = delta2.dot(w2.T) * tanh_derivative(a1)

    # Weights update
    w3 += a2.T.dot(delta3) * eta
    w2 += a1.T.dot(delta2) * eta
    w1 += inp.T.dot(delta1) * eta
    return w1, w2, w3, b1, b2, b3, errors


# Training loop
for i in range(epochs):
    permutation_indices = np.random.permutation(len(yd))
    # Aplicar la misma permutación a ambos arreglos
    s = s[permutation_indices]
    yd = yd[permutation_indices]
    for j in range(packages_iterations - 1):
        s_batch = s[j * packages_size:(j + 1) * packages_size]
        yd_batch = yd[j * packages_size:(j + 1) * packages_size]
        train(s_batch, w1, w2, w3, yd_batch, eta, b1, b2, b3, errors)

# Plotting the error
errors = np.array(errors).flatten()
#plt.plot(errors)
#plt.show()

# Test
xtest = np.sin(np.random.uniform(0, 2 * np.pi, 200))
ytest = np.cos(np.random.uniform(0, 2 * np.pi, 200))
ztest = np.random.uniform(-1, 1, 200)

stest = np.array([xtest, ytest, ztest]).T
ydtest = np.array([np.sin(xtest) + np.cos(ytest) + ztest]).T
ytest = []

for i in range(200):
    z1 = np.dot(stest[i].reshape(1, 3), w1) + b1
    a1 = tanh(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = tanh(z2)
    z3 = np.dot(a2, w3) + b3
    a3 = tanh(z3)
    ytest.append(a3)

ytest = np.array(ytest).flatten()
ytest = ytest * 3
plt.plot(ydtest, ytest, 'r.')
plt.show()

print(ydtest[100])
print("\n", ytest[100])
