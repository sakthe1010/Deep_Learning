import numpy as np 

from activations import ACTIVATIONS
from losses import LOSSES
############################################ NEURAL NETWORK MODEL ############################################
class Layer():
    def __init__(self, size, activation='sigmoid'):
        self.size = size
        self.activation = activation
    
    def printLayer(self):
        print(self.size)

class Sequential():
    def __init__(self, weight_init='random'):
        self.weight_init = weight_init
        self.layers = []
        self.weights = []
        self.biases = []

    def add(self, layer: Layer):
        self.layers.append(layer)

    def create_parameters(self):
        for i in range(len(self.layers)-1):

            current_size = self.layers[i].size
            next_size = self.layers[i+1].size

            if self.weight_init == 'random':
                # Random initialization
                weight_matrix = np.random.randn(next_size, current_size)
            elif self.weight_init.lower() == 'xavier':
                # Xavier initialization
                weight_matrix = np.random.randn(next_size, current_size) * np.sqrt(1. / current_size)
            else:
                weight_matrix = np.random.randn(next_size, current_size)
            self.weights.append(weight_matrix)

            bias_vector = np.random.randn(next_size, 1)
            self.biases.append(bias_vector)

    def forward(self, x):
        pre_activations = []
        activations = [x]
        for i in range(len(self.layers)-1):
            z = np.dot(self.weights[i], x) + self.biases[i]
            pre_activations.append(z)
            activation_func, _ = ACTIVATIONS[self.layers[i+1].activation]
            a = activation_func(z)
            activations.append(a)
            x = a
        return pre_activations, activations
    
    def backward(self, y_true, pre_activations, activations, loss_name):
        grads_W = [None] * len(self.weights)
        grads_b = [None] * len(self.biases)

        output_activation = self.layers[-1].activation
        _, loss_deriv = LOSSES[loss_name]

        y_pred = activations[-1]
        dL_da = loss_deriv(y_pred, y_true)

        L = len(self.layers) - 1
        if output_activation == 'softmax' and loss_name == 'cross_entropy':
            delta = dL_da
        else:
            _, output_deriv = ACTIVATIONS[output_activation]
            if output_deriv is None:
                delta = dL_da
            else:
                delta = dL_da * output_deriv(pre_activations[-1])

        grads_W[-1] = np.dot(delta, activations[-2].T)
        grads_b[-1] = np.sum(delta, axis=1, keepdims=True)

        for l in range(L-1, 0, -1):
            activation_name = self.layers[l].activation
            _, activation_deriv = ACTIVATIONS[activation_name]

            delta = np.dot(self.weights[l].T, delta)
            delta *= activation_deriv(pre_activations[l-1])

            grads_W[l-1] = np.dot(delta, activations[l-1].T)
            grads_b[l-1] = np.sum(delta, axis=1, keepdims=True)

        return grads_W, grads_b

############################################ MAIN FUNCTION ############################################

if __name__ == "__main__":
    
    model = Sequential(weight_init='xavier')
    model.add(Layer(784, activation='identity'))   
    model.add(Layer(64, activation='relu'))          
    model.add(Layer(10, activation='softmax'))       
    model.create_parameters()

    X_dummy = np.random.rand(784, 1)
    y_dummy = np.zeros((10, 1))
    y_dummy[3] = 1

    pre_activations, activations = model.forward(X_dummy)

    grads_W, grads_b = model.backward(y_dummy, pre_activations, activations, loss_name='cross_entropy')
    
    for i, (gw, gb) in enumerate(zip(grads_W, grads_b)):
        print(f"Layer {i+1}: grad_W shape: {gw.shape}, grad_b shape: {gb.shape}")