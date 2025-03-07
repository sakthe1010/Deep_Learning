import numpy as np 

############################################ ACTIVATION FUNCTIONS ############################################
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def identity(x):
    return x

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x, axis=0)

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
        for i in range(len(self.layers)-1):
            x = np.dot(self.weights[i], x) + self.biases[i]

            if self.layers[i+1].activation == 'relu':
                x = relu(x)
            elif self.layers[i+1].activation == 'sigmoid':
                x = sigmoid(x)
            elif self.layers[i+1].activation == 'tanh':
                x = tanh(x)
            elif self.layers[i+1].activation == 'identity':
                x = identity(x)
            elif self.layers[i+1].activation == 'softmax':
                x = softmax(x)
        return x

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