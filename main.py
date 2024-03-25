import numpy as np
import os

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def train_and_save_model():
    inputs = np.array([[0,0,0], [0,1,0], [1,0,0], [1,1,0], [0,0,1], [0,1,1], [1,0,1], [1,1,1]])
    expected_output = np.array([[0], [0], [0], [1], [0], [1], [0], [1]])

    epochs = 20000
    initial_learning_rate = 0.05

    inputLayerNeurons, hiddenLayerNeurons1, hiddenLayerNeurons2, outputLayerNeurons = 3, 8, 6, 1

    hidden_weights1 = np.random.uniform(size=(inputLayerNeurons, hiddenLayerNeurons1))
    hidden_bias1 = np.zeros((1, hiddenLayerNeurons1))
    hidden_weights2 = np.random.uniform(size=(hiddenLayerNeurons1, hiddenLayerNeurons2))
    hidden_bias2 = np.zeros((1, hiddenLayerNeurons2))
    output_weights = np.random.uniform(size=(hiddenLayerNeurons2, outputLayerNeurons))
    output_bias = np.zeros((1, outputLayerNeurons))

    for epoch in range(epochs):
        hidden_layer_activation1 = relu(np.dot(inputs, hidden_weights1) + hidden_bias1)
        hidden_layer_activation2 = relu(np.dot(hidden_layer_activation1, hidden_weights2) + hidden_bias2)
        predicted_output = sigmoid(np.dot(hidden_layer_activation2, output_weights) + output_bias)

        error = expected_output - predicted_output
        d_predicted_output = error * sigmoid_derivative(predicted_output)
        
        error_hidden_layer2 = d_predicted_output.dot(output_weights.T)
        d_hidden_layer2 = error_hidden_layer2 * relu_derivative(hidden_layer_activation2)
        
        error_hidden_layer1 = d_hidden_layer2.dot(hidden_weights2.T)
        d_hidden_layer1 = error_hidden_layer1 * relu_derivative(hidden_layer_activation1)

        output_weights += hidden_layer_activation2.T.dot(d_predicted_output) * initial_learning_rate
        output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * initial_learning_rate
        hidden_weights2 += hidden_layer_activation1.T.dot(d_hidden_layer2) * initial_learning_rate
        hidden_bias2 += np.sum(d_hidden_layer2, axis=0, keepdims=True) * initial_learning_rate
        hidden_weights1 += inputs.T.dot(d_hidden_layer1) * initial_learning_rate
        hidden_bias1 += np.sum(d_hidden_layer1, axis=0, keepdims=True) * initial_learning_rate

    np.save('hidden_weights1.npy', hidden_weights1)
    np.save('hidden_bias1.npy', hidden_bias1)
    np.save('hidden_weights2.npy', hidden_weights2)
    np.save('hidden_bias2.npy', hidden_bias2)
    np.save('output_weights.npy', output_weights)
    np.save('output_bias.npy', output_bias)
    print("Model parameters have been saved.")

def load_model_and_predict(input_data):
    hidden_weights1 = np.load('hidden_weights1.npy')
    hidden_bias1 = np.load('hidden_bias1.npy')
    hidden_weights2 = np.load('hidden_weights2.npy')
    hidden_bias2 = np.load('hidden_bias2.npy')
    output_weights = np.load('output_weights.npy')
    output_bias = np.load('output_bias.npy')

    hidden_layer_activation1 = relu(np.dot(input_data, hidden_weights1) + hidden_bias1)
    hidden_layer_activation2 = relu(np.dot(hidden_layer_activation1, hidden_weights2) + hidden_bias2)
    predicted_output = sigmoid(np.dot(hidden_layer_activation2, output_weights) + output_bias)

    return (predicted_output > 0.5).astype(int)

def play_game():
    print("\nWelcome to the Number Guessing Game!\nEnter three numbers (0 or 1) separated by commas. The AI will predict if the output is 0 or 1 based on its training.")
    while True:
        user_input = input("Enter your numbers (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        try:
            input_vals = np.array([[int(num) for num in user_input.split(',')]])
            prediction = load_model_and_predict(input_vals)
            print(f"AI prediction: {prediction[0][0]}")
        except:
            print("Invalid input. Please enter three numbers separated by commas (e.g., 0,1,0).")

if __name__ == "__main__":
    if not all(os.path.exists(f) for f in ['hidden_weights1.npy', 'hidden_bias1.npy', 'hidden_weights2.npy', 'hidden_bias2.npy', 'output_weights.npy', 'output_bias.npy']):
        train_and_save_model()
    play_game()