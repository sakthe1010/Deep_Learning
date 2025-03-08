import numpy as np

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def tanh(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1 - np.tanh(z)**2

def identity(z):
    return z

def identity_derivative(z):
    return np.ones_like(z)

def softmax(z):
    z_stable = z - np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(z_stable)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

# Dictionaries mapping activation names to (forward_func, derivative_func)
ACTIVATIONS = {
    "relu": (relu, relu_derivative),
    "sigmoid": (sigmoid, sigmoid_derivative),
    "tanh": (tanh, tanh_derivative),
    "identity": (identity, identity_derivative),
    "none": (identity, identity_derivative),
    "softmax": (softmax, None), 
}
