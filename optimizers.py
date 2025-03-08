class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, model, grads_W, grads_b):
        for i in range(len(model.weights)):
            model.weights[i] -= self.lr * grads_W[i]
            model.biases[i]  -= self.lr * grads_b[i]


