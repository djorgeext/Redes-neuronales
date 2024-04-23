import numpy as np
import matplotlib.pyplot as plt

# XOR input and output
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

# Sigmoid function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# tanh function and its derivative
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - x ** 2

# Genetic algorithm parameters
population_size = 100
mutation_rate = 0.08
num_generations = 1000

# Perceptron parameters
input_size = 2
hidden_size = 4
output_size = 1

# Initialize population
population = np.random.uniform(low=-1, high=1, size=(population_size, hidden_size * (input_size + output_size)))

# Genetic algorithm loop
for generation in range(num_generations):
    # Evaluate fitness of population
    fitness = np.zeros(population_size)
    for i in range(population_size):
        weights = population[i]
        hidden_weights = weights[:hidden_size * input_size].reshape(hidden_size, input_size)
        output_weights = weights[hidden_size * input_size:].reshape(output_size, hidden_size)

        hidden_layer_input = np.dot(X, hidden_weights.T)
        hidden_layer_output = tanh(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, output_weights.T)
        output_layer_output = tanh(output_layer_input)

        fitness[i] = -np.sum(np.square(Y - output_layer_output))

    # Select parents
    parents = population[np.argsort(fitness)[-10:]]

    # Crossover
    offspring = np.zeros((population_size, hidden_size * (input_size + output_size)))
    for i in range(population_size):
        crossover_point = np.random.randint(low=0, high=hidden_size * (input_size + output_size))
        offspring[i, :crossover_point] = parents[0, :crossover_point]
        offspring[i, crossover_point:] = parents[1, crossover_point:]

    # Mutation
    mutations = np.random.uniform(low=-1, high=1, size=(population_size, hidden_size * (input_size + output_size))) < mutation_rate
    offspring[mutations] += np.random.uniform(low=-0.1, high=0.1, size=(population_size, hidden_size * (input_size + output_size)))[mutations]

    # Update population
    population = offspring

# Print best individual
best_individual = population[np.argmax(fitness)]
hidden_weights = best_individual[:hidden_size * input_size].reshape(hidden_size, input_size)
output_weights = best_individual[hidden_size * input_size:].reshape(output_size, hidden_size)

hidden_layer_input = np.dot(X, hidden_weights.T)
hidden_layer_output = tanh(hidden_layer_input)

output_layer_input = np.dot(hidden_layer_output, output_weights.T)
output_layer_output = tanh(output_layer_input)

print(output_layer_output)
plt.plot(fitness)
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.show()