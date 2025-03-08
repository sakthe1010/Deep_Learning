import argparse
import numpy as np
import wandb


from model import Layer, Sequential
from activations import ACTIVATIONS
from losses import LOSSES
from optimizers import OPTIMIZERS

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Neural Network with W&B logging")

    parser.add_argument('-wp', "--wandb_project", type=str, default="myprojectname", help="Project name used to track experiments in Weights & Biases dashboard.")
    parser.add_argument('-we',"--wandb_entity", type=str, default="myname", help="W&B Entity used to track experiments in the Weights & Biases dashboard.")
    parser.add_argument('-d',"--dataset", type=str, default="fashion_mnist", choices=["fashion_mnist", "mnist"], help="Which dataset to train on. If not available, fallback to dummy data.")
    parser.add_argument('-e',"--epochs", type=int, default=1, help="Number of epochs to train.")
    parser.add_argument('-b',"--batch_size", type=int, default=4, help="Batch size used to train neural network.")
    parser.add_argument('-l',"--loss", type=str, default="cross_entropy", choices=["mean_squared_error", "cross_entropy"], help="Loss function to use.")
    parser.add_argument('-o',"--optimizer", type=str, default="sgd", choices=["sgd", "momentum", "nesterov", "rmsprop", "adam", "nadam"], help="Optimizer to use.")
    parser.add_argument('-lr',"--learning_rate", type=float, default=0.1, help="Learning rate.")
    parser.add_argument('-m',"--momentum", type=float, default=0.5, help="Momentum used by momentum and nesterov.")
    parser.add_argument('-beta',"--beta", type=float, default=0.5, help="Beta used by rmsprop optimizer.")
    parser.add_argument('-beta1',"--beta1", type=float, default=0.5, help="Beta1 used by adam and nadam optimizers.")
    parser.add_argument('-beta2',"--beta2", type=float, default=0.5, help="Beta2 used by adam and nadam optimizers.")
    parser.add_argument('-eps',"--epsilon", type=float, default=1e-6, help="Epsilon used by optimizers.")
    parser.add_argument('-w_d',"--weight_decay", type=float, default=0.0, help="Weight decay (L2 regularization). Not implemented in all optimizers above.")
    parser.add_argument('-w_i',"--weight_init", type=str, default="random",choices=["random", "xavier"], help="Weight initialization method.")
    parser.add_argument("-nhl", "--num_layers", type=int, default=1, help="Number of hidden layers.")
    parser.add_argument("-sz", "--hidden_size", type=int, default=4, help="Number of neurons in feedforward layer.")
    parser.add_argument("-a", "--activation", type=str, default="sigmoid", choices=["identity", "sigmoid", "tanh", "relu"], help="Activation function for hidden layers.")
    return parser.parse_args()

def load_data(dataset_name):
    if dataset_name == "mnist":
        from keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    else:
        from keras.datasets import fashion_mnist
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], -1).T / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1).T / 255.0
    num_classes = 10

    y_train_oh = np.zeros((num_classes, y_train.shape[0]))
    for i, label in enumerate(y_train):
        y_train_oh[label, i] = 1
    y_test_oh = np.zeros((num_classes, y_test.shape[0]))
    for i, label in enumerate(y_test):
        y_test_oh[label, i] = 1

    return x_train, y_train_oh, x_test, y_test_oh

def create_model(args):
    model = Sequential(weight_init=args.weight_init)
    model.add(Layer(784, activation="identity"))
    for _ in range(args.num_layers):
        model.add(Layer(args.hidden_size, activation=args.activation))

    model.add(Layer(10, activation="softmax"))
    model.create_parameters()
    return model

def train_one_epoch(model, optimizer, x_train, y_train, loss_name, batch_size):
    loss_func, _ = LOSSES[loss_name]
    num_samples = x_train.shape[1]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    x_train = x_train[:, indices]
    y_train = y_train[:, indices]

    total_loss = 0.0
    num_batches = num_samples // batch_size

    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        x_batch = x_train[:, start:end]
        y_batch = y_train[:, start:end]

        pre_acts, acts = model.forward(x_batch)
        loss_value = loss_func(acts[-1], y_batch)
        total_loss += loss_value

        grad_W, grad_b = model.backward(y_batch, pre_acts, acts, loss_name)
        optimizer.update(model, grad_W, grad_b)

    if num_batches == 0:
        return total_loss
    else:
        return total_loss / num_batches
    
