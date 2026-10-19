"""
Project 005: Simple neural network from scratch

Overview:
  This project builds a production-minded simple neural network from scratch workflow inside the broader area of foundational machine learning. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import numpy as np

# Simple 2-layer Neural Network trained on XOR logic
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Inputs: [x1, x2], Output: XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

np.random.seed(42)
weights_input_hidden = np.random.uniform(size=(2, 4))
weights_hidden_output = np.random.uniform(size=(4, 1))
bias_hidden = np.random.uniform(size=(1, 4))
bias_output = np.random.uniform(size=(1, 1))

lr = 0.5
epochs = 3000

for epoch in range(epochs):
    # Forward Pass
    hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(final_input)

    # Backpropagation
    error = y - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    
    error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_output)

    # Gradient Descent Weight Updates
    weights_hidden_output += hidden_output.T.dot(d_predicted_output) * lr
    bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * lr
    weights_input_hidden += X.T.dot(d_hidden_layer) * lr
    bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * lr

print("=== Simple Neural Network from Scratch (XOR Problem) ===")
print("Final Predictions after 3000 epochs:")
for inp, target, pred in zip(X, y, predicted_output):
    print(f"  Input: {inp} | Expected: {target[0]} | Predicted: {pred[0]:.4f} (Class {int(round(pred[0]))})")
