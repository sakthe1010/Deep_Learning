import argparse
import numpy as np

from model import Layer, Sequential
from activations import ACTIVATIONS
from losses import LOSSES
from optimizers import OPTIMIZERS

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Neural Network with W&B logging")

    parser.add_argument("--wandb_project", type=str, default="myprojectname",
                        help="Project name used to track experiments in Weights & Biases dashboard.")
    parser.add_argument("--wandb_entity", type=str, default="myname",
                        help="W&B Entity used to track experiments in the Weights & Biases dashboard.")

    parser.add_argument("--dataset", type=str, default="fashion_mnist",
                        choices=["fashion_mnist", "mnist"],
                        help="Which dataset to train on. If not available, fallback to dummy data.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size used to train neural network.")
    parser.add_argument("--loss", type=str, default="cross_entropy",
                        choices=["mean_squared_error", "cross_entropy"],
                        help="Loss function to use.")
    parser.add_argument("--optimizer", type=str, default="sgd",
                        choices=["sgd", "momentum", "nesterov", "rmsprop", "adam", "nadam"],
                        help="Optimizer to use.")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate.")
    parser.add_argument("--momentum", type=float, default=0.5, help="Momentum used by momentum and nesterov.")
    parser.add_argument("--beta1", type=float, default=0.5 help="Beta1 used by adam and nadam optimizers.")
    parser.add_argument("--beta2", type=float, default=0.5, help="Beta2 used by adam and nadam optimizers.")
    parser.add_argument("--eps", type=float, default=1e-8, help="Epsilon used by optimizers.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay (L2 regularization). Not implemented in all optimizers above.")
    parser.add_argument("--weight_init", type=str, default="random",
                        choices=["random", "xavier"],
                        help="Weight initialization method.")
    parser.add_argument("--n_hidden_layers", type=int, default=1, help="Number of hidden layers.")
    parser.add_argument("--n_h1", type=int, default=64, help="Number of neurons in first hidden layer.")
    parser.add_argument("--n_h2", type=int, default=32, help="Number of neurons in second hidden layer.")
    parser.add_argument("--activation", type=str, default="sigmoid",
                        choices=["identity", "sigmoid", "tanh", "relu"],
                        help="Activation function for hidden layers.")
    return parser.parse_args()
