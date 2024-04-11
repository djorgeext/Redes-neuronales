import numpy as np

# Función de activación (tangente hiperbólica)
def activation_function(x):
    return np.tanh(x)

# Derivada de la función de activación (para el entrenamiento)
def activation_derivative(x):
    return 1 - np.tanh(x)**2

# Función para entrenar el perceptrón
def train_perceptron(inputs, outputs, learning_rate=0.3, epochs=1000):
    num_inputs = len(inputs[0])
    weights = np.random.rand(num_inputs + 1)  # +1 para el sesgo
    errors = []

    for _ in range(epochs):
        total_error = 0
        for input_row, desired_output in zip(inputs, outputs):
            input_with_bias = np.concatenate(([1], input_row))  # Agregar el sesgo
            prediction = activation_function(np.dot(input_with_bias, weights))
            error = desired_output - prediction
            total_error += abs(error)
            delta = learning_rate * error * activation_derivative(np.dot(input_with_bias, weights))
            weights += delta * input_with_bias
        errors.append(total_error)
        if total_error == 0:
            break

    return weights, errors

# Función para realizar la predicción con el perceptrón entrenado
def predict(inputs, weights):
    input_with_bias = np.concatenate(([1], inputs))  # Agregar el sesgo
    return activation_function(np.dot(input_with_bias, weights))

# Datos de entrada y salida para la operación XOR de 4 entradas
inputs = np.array([[0, 0, 0, 0],
                   [0, 0, 0, 1],
                   [0, 0, 1, 0],
                   [0, 0, 1, 1],
                   [0, 1, 0, 0],
                   [0, 1, 0, 1],
                   [0, 1, 1, 0],
                   [0, 1, 1, 1],
                   [1, 0, 0, 0],
                   [1, 0, 0, 1],
                   [1, 0, 1, 0],
                   [1, 0, 1, 1],
                   [1, 1, 0, 0],
                   [1, 1, 0, 1],
                   [1, 1, 1, 0],
                   [1, 1, 1, 1]])

outputs = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0])

# Entrenar el perceptrón
weights, errors = train_perceptron(inputs, outputs)

# Realizar predicciones
for input_row in inputs:
    prediction = predict(input_row, weights)
    print(f'Input: {input_row}, Prediction: {prediction}')

print('Final weights:', weights)
