import numpy as np

class SGD:
    def __init__(self, lr=0.1, **kwargs):
        self.lr = lr

    def update(self, model, grad_W, grad_b):
        for i in range(len(model.weights)):
            model.weights[i] -= self.lr * grad_W[i]
            model.biases[i]  -= self.lr * grad_b[i]

class Momentum:
    def __init__(self, lr=0.1, momentum=0.5, **kwargs):
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
    def __init__(self, lr=0.1, momentum=0.5, **kwargs):
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
    def __init__(self, lr=0.1, beta=0.5, **kwargs):
        self.lr = lr
        self.beta = beta
        self.vel_W = None
        self.vel_b = None

    def update(self, model, grad_W, grad_b):
        if self.vel_W is None:
            self.vel_W = [np.zeros_like(w) for w in model.weights]
            self.vel_b = [np.zeros_like(b) for b in model.biases]

        for i in range(len(model.weights)):
            self.vel_W[i] = self.beta * self.vel_W[i] + (1 - self.beta) * grad_W[i]**2
            self.vel_b[i] = self.beta * self.vel_b[i] + (1 - self.beta) * grad_b[i]**2
            model.weights[i] -= self.lr * grad_W[i] / np.sqrt(self.vel_W[i] + 1e-8)
            model.biases[i]  -= self.lr * grad_b[i] / np.sqrt(self.vel_b[i] + 1e-8)

class Adam:
    def __init__(self, lr=0.1, beta1=0.5, beta2=0.5, **kwargs):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.fm_W = None
        self.sm_W = None
        self.fm_b = None
        self.sm_b = None
        self.t = 0  # timestep

    def update(self, model, grad_W, grad_b):
        if self.fm_W is None:
            self.fm_W = [np.zeros_like(w) for w in model.weights]
            self.sm_W = [np.zeros_like(w) for w in model.weights]
            self.fm_b = [np.zeros_like(b) for b in model.biases]
            self.sm_b = [np.zeros_like(b) for b in model.biases]  

        self.t += 1
        for i in range(len(model.weights)):
            self.fm_W[i] = self.beta1 * self.fm_W[i] + (1 - self.beta1) * grad_W[i]
            self.sm_W[i] = self.beta2 * self.sm_W[i] + (1 - self.beta2) * grad_W[i]**2
            self.fm_b[i] = self.beta1 * self.fm_b[i] + (1 - self.beta1) * grad_b[i]
            self.sm_b[i] = self.beta2 * self.sm_b[i] + (1 - self.beta2) * grad_b[i]**2

            fm_hat_W = self.fm_W[i] / (1 - self.beta1**self.t)
            sm_hat_W = self.sm_W[i] / (1 - self.beta2**self.t)
            fm_hat_b = self.fm_b[i] / (1 - self.beta1**self.t)
            sm_hat_b = self.sm_b[i] / (1 - self.beta2**self.t)

            model.weights[i] -= self.lr * fm_hat_W / (np.sqrt(sm_hat_W) + 1e-8)
            model.biases[i]  -= self.lr * fm_hat_b / (np.sqrt(sm_hat_b) + 1e-8)

class Nadam:
    def __init__(self, lr=0.1, beta1=0.5, beta2=0.5, **kwargs):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.fm_W = None
        self.sm_W = None
        self.fm_b = None
        self.sm_b = None        
        self.t = 0  # timestep

    def update(self, model, grad_W, grad_b):
        if self.fm_W is None:
            self.fm_W = [np.zeros_like(w) for w in model.weights]
            self.sm_W = [np.zeros_like(w) for w in model.weights]
            self.fm_b = [np.zeros_like(b) for b in model.biases]
            self.sm_b = [np.zeros_like(b) for b in model.biases]    

        self.t += 1
        for i in range(len(model.weights)):
            self.fm_W[i] = self.beta1 * self.fm_W[i] + (1 - self.beta1) * grad_W[i]
            self.sm_W[i] = self.beta2 * self.sm_W[i] + (1 - self.beta2) * (grad_W[i] ** 2)
            self.fm_b[i] = self.beta1 * self.fm_b[i] + (1 - self.beta1) * grad_b[i]
            self.sm_b[i] = self.beta2 * self.sm_b[i] + (1 - self.beta2) * (grad_b[i] ** 2)

            fm_hat_W = self.fm_W[i] / (1 - self.beta1 ** self.t)
            sm_hat_W = self.sm_W[i] / (1 - self.beta2 ** self.t)
            fm_hat_b = self.fm_b[i] / (1 - self.beta1 ** self.t)
            sm_hat_b = self.sm_b[i] / (1 - self.beta2 ** self.t)

            update_W = (self.beta1 * fm_hat_W + (1 - self.beta1) * grad_W[i] / (1 - self.beta1 ** self.t))
            update_b = (self.beta1 * fm_hat_b + (1 - self.beta1) * grad_b[i] / (1 - self.beta1 ** self.t))

            model.weights[i] -= self.lr * update_W / (np.sqrt(sm_hat_W) + 1e-8)
            model.biases[i]  -= self.lr * update_b / (np.sqrt(sm_hat_b) + 1e-8)

OPTIMIZERS = {
    "sgd": SGD,
    "momentum": Momentum,
    "nesterov": Nesterov,
    "rmsprop": RMSprop,
    "adam": Adam,
    "nadam": Nadam
}