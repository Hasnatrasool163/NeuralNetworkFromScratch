
import numpy as np
import os
import pickle

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, x * alpha)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

class DeepNeuralNetwork:
    def __init__(self, layers, learning_rate=0.01, lambda_reg=0.001):
        self.layers = layers
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.weights = [np.random.randn(layers[i], layers[i+1]) * np.sqrt(2. / layers[i]) for i in range(len(layers)-1)]
        self.biases = [np.zeros((1, layers[i+1])) for i in range(len(layers)-1)]

    def feedforward(self, X):
        activations = [X]
        for i in range(len(self.weights) - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            a = leaky_relu(z)
            activations.append(a)
        z = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        a = softmax(z)
        activations.append(a)
        return activations

    def compute_loss(self, Y, Y_hat):
        m = Y.shape[0]
        l2_penalty = sum(np.sum(np.square(w)) for w in self.weights) * (self.lambda_reg / (2 * m))
        cross_entropy = -np.sum(Y * np.log(Y_hat + 1e-8)) / m
        return cross_entropy + l2_penalty

    def backprop(self, X, Y, activations):
        m = X.shape[0]
        Y_hat = activations[-1]
        delta = Y_hat - Y
        dW = np.dot(activations[-2].T, delta) / m
        db = np.sum(delta, axis=0, keepdims=True) / m

        dW += (self.lambda_reg / m) * self.weights[-1]
        self.weights[-1] -= self.learning_rate * dW
        self.biases[-1] -= self.learning_rate * db

        for i in range(len(self.weights) - 2, -1, -1):
            delta = np.dot(delta, self.weights[i + 1].T) * leaky_relu_derivative(activations[i + 1])
            dW = np.dot(activations[i].T, delta) / m
            db = np.sum(delta, axis=0) / m

            dW += (self.lambda_reg / m) * self.weights[i]
            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db

    def train(self, X, Y, epochs):
        for epoch in range(epochs):
            activations = self.feedforward(X)
            loss = self.compute_loss(Y, activations[-1])
            self.backprop(X, Y, activations)

            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, loss: {loss}")

    def predict(self, X):
        activations = self.feedforward(X)
        return np.argmax(activations[-1], axis=1)

    def save_weights(self, filename='model_weights95.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump((self.weights, self.biases), f)

    def load_weights(self, filename='model_weights95.pkl'):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.weights, self.biases = pickle.load(f)
                print("Model weights loaded.")
def generate_data():
    # Adjust to generate data for 95 characters
    n = 95  # Number of characters
    X = np.eye(n)  # Input: Identity matrix representing one-hot encoded characters
    Y = np.eye(n)  # Output: Same as input
    return X, Y

def test_all_characters(nn, test_characters):
    print("Testing all characters...")
    for char in test_characters:
        idx = test_characters.index(char)
        prediction = nn.predict(np.eye(95)[idx].reshape(1, -1))
        print(f"Input: {char}, Predicted: {test_characters[prediction[0]]}")

def manual_test(nn, test_characters):
    while True:
        user_input = input("Enter a character to test (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        elif user_input in test_characters:
            idx = test_characters.index(user_input)
            prediction = nn.predict(np.eye(95)[idx].reshape(1, -1))
            print(f"Input: {user_input}, Predicted: {test_characters[prediction[0]]}")
        else:
            print("Character not in test range.")

def main():
    np.random.seed(42)
    layers = [95, 128, 128, 95]
    nn = DeepNeuralNetwork(layers, learning_rate=0.001, lambda_reg=0.01)
    
    if os.path.exists('model_weights95.pkl'):
        nn.load_weights('model_weights95.pkl')
    else:
        X, Y = generate_data()
        nn.train(X, Y, epochs=29000)
        nn.save_weights('model_weights95.pkl')

    test_characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789' + \
                      '!@#$%^&*()_+-=[]{}|;:\'",.<>/?`~'
                      
    mode = input("Enter 'test' to test all characters or 'manual' for manual testing: ")
    if mode.lower() == 'test':
        test_all_characters(nn, test_characters)
    elif mode.lower() == 'manual':
        manual_test(nn, test_characters)
    else:
        print("Invalid option selected.")

if __name__ == "__main__":
    main()
