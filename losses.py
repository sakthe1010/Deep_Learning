import numpy as np

def cross_entropy_loss(y_pred, y_true):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[1]

def cross_entropy_derivative(y_pred, y_true):
    return (y_pred - y_true) / y_true.shape[1]

def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true)**2)

def mse_derivative(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_true.shape[1]

# Dictionary containing the loss functions and their derivatives
LOSSES = {
    "cross_entropy": (cross_entropy_loss, cross_entropy_derivative),
    "mean_squared_error": (mse_loss, mse_derivative)
}
