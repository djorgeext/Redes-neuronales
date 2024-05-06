# Multicap perceptron to compute XOR
import numpy as np
import matplotlib.pyplot as plt

# Activation function tanh
def tanh(x):
    return np.tanh(x)

# Derivative of the tanh activation function
def tanh_derivative(x):
    return 1.0 - x**2

# function activation sigmoid
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Derivative of the sigmoid activation function
def sigmoid_derivative(x):
    return x*(1-x)

# Input data
s = np.array([[-1, -1],[-1, 1],[1, -1],[1, 1]])
yd = np.array([[-1], [1], [1], [-1]])

# Weights initialization
w1 = np.random.rand(2, 4)
w2 = np.random.rand(4, 1)

# Bias initialization
b1 = np.random.rand(1, 4)
b2 = np.random.rand(1, 1)

# Learning rate
eta = 0.1

# Number of epochs
epochs = 1000

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
    delta1 = delta2.dot(w2.T) * tanh_derivative(a1)

    # Weights update
    w2 += a1.T.dot(delta2) * eta
    w1 += inp.T.dot(delta1) * eta

    # Bias update
    b2 += np.sum(delta2) * eta
    b1 += np.sum(delta1) * eta

    return error, w1, w2, b1, b2

# Training loop
for i in range(epochs):
    train(s, w1, w2, yd, eta, b1, b2)
        
# Plotting the error
"""errors = np.array(errors).flatten()
plt.plot(errors, linewidth=4)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Error', fontsize=20)
plt.title('Error during training', fontsize=20)
plt.show()"""

# Test

""" for i in range(len(s)):
    z1 = np.dot(s[i].reshape(1, 2), w1) + b1
    a1 = tanh(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = tanh(z2)
    print('Input:', s[i], 'Output:', a2) """

# Test varying two weights
error_mesh = np.zeros((500, 500))
w_aux = np.linspace(-10*w1[0][0], 10*w1[0][0], 500)
w_aux2 = np.linspace(-5*w2[3], 5*w2[3], 500)
for i in range(len(w_aux)):
    for j in range(len(w_aux2)):
        w1[0][0] = w_aux[i]
        w2[3] = w_aux2[j]
        z1 = np.dot(s, w1) + b1  # especificar el subindice para s[] y yd[] en caso de que se quiera ver el error para cada uno de los 4 pares de entrada
        a1 = tanh(z1)
        z2 = np.dot(a1, w2) + b2
        a2 = tanh(z2)
        error_mesh[i][j] = 0.5*np.sum((yd - a2)**2)
# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

 # Create meshgrid for x and y values
x = w_aux
y = w_aux2
X, Y = np.meshgrid(x, y)

# Plot the surface
ax.plot_surface(X, Y, error_mesh, cmap='viridis', alpha=1)

# Set labels for axes
ax.set_xlabel('Primer peso, primera capa', fontsize=20)
ax.set_ylabel('Cuarto peso, segunda capa', fontsize=20)
ax.set_zlabel('Error', fontsize=20)

# Show the plot
plt.show()