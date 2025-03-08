import numpy as np 

from activations import ACTIVATIONS

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

            activations.append(x)
        return pre_activations, activations
    
    def backward(self, y, pre_activations, activations):
        grads_W = [None] * len(self.weights)
        grads_b = [None] * len(self.biases)
        L = len(self.layers) - 1  # Index of the last layer.
        
        aL = activations[-1]
        if self.layers[-1].activation == 'softmax':
            delta = aL - y
        else:
            _, derivative_func = ACTIVATIONS[self.layers[-1].activation]
            delta = (aL - y) * derivative_func(pre_activations[-1])

        grads_W[-1] = np.dot(delta, activations[-2].T)
        grads_b[-1] = np.sum(delta, axis=1, keepdims=True)
        
        for l in range(L-1, 0, -1):
            _, derivative_func = ACTIVATIONS[self.layers[l].activation]
            delta = np.dot(self.weights[l].T, delta) * derivative_func(pre_activations[l-1])
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
    output = model.forward(X_dummy)
    output.shape
    print(output)