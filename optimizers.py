import numpy as np

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, model, grad_W, grad_b):
        for i in range(len(model.weights)):
            model.weights[i] -= self.lr * grad_W[i]
            model.biases[i]  -= self.lr * grad_b[i]

class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.vel_W = None
        self.vel_b = None

    def update(self, model, grad_W, grad_b):
        if self.vel_W is None:
            self.vel_W = [np.zeros_like(w) for w in model.weights]
            self.vel_b = [np.zeros_like(b) for b in model.biases]

        for i in range(len(model.weights)):
            self.vel_W[i] = self.momentum * self.vel_W[i] + self.lr * grad_W[i]
            self.vel_b[i] = self.momentum * self.vel_b[i] + self.lr * grad_b[i]
            model.weights[i] -= self.vel_W[i]
            model.biases[i]  -= self.vel_b[i]

class Nesterov:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.vel_W = None
        self.vel_b = None

    def update(self, model, grad_W, grad_b):
        if self.vel_W is None:
            self.vel_W = [np.zeros_like(w) for w in model.weights]
            self.vel_b = [np.zeros_like(b) for b in model.biases]

        for i in range(len(model.weights)):
            # Save previous velocity
            prev_vel_W = self.vel_W[i].copy()
            prev_vel_b = self.vel_b[i].copy()

            self.vel_W[i] = self.momentum * self.vel_W[i] - self.lr * grad_W[i]
            self.vel_b[i] = self.momentum * self.vel_b[i] - self.lr * grad_b[i]

            model.weights[i] += -self.momentum *    prev_vel_W + (1 + self.momentum) * self.vel_W[i]
            model.biases[i]  += -self.momentum * prev_vel_b + (1 + self.momentum) * self.vel_b[i]

class RMSprop:
    def __init__(self, lr=0.01, decay=0.9):
        self.lr = lr
        self.decay = decay
        self.vel_W = None
        self.vel_b = None

    def update(self, model, grad_W, grad_b):
        if self.vel_W is None:
            self.vel_W = [np.zeros_like(w) for w in model.weights]
            self.vel_b = [np.zeros_like(b) for b in model.biases]

        for i in range(len(model.weights)):
            self.vel_W[i] = self.decay * self.vel_W[i] + (1 - self.decay) * grad_W[i]**2
            self.vel_b[i] = self.decay * self.vel_b[i] + (1 - self.decay) * grad_b[i]**2
            model.weights[i] -= self.lr * grad_W[i] / np.sqrt(self.vel_W[i] + 1e-8)
            model.biases[i]  -= self.lr * grad_b[i] / np.sqrt(self.vel_b[i] + 1e-8)


