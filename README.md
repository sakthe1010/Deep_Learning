# Assignment 1: Weights & Biases Experimentation

This repository implements a feedforward neural network from scratch using NumPy, along with custom backpropagation and various gradient descent optimization algorithms. Experiment tracking and hyperparameter tuning are integrated using Weights & Biases (W&B).

## Overview
The goals of this assignment are to:
- **Implement a feedforward neural network** that takes a 28×28 input image and outputs a probability distribution over 10 classes.
- **Write your own backpropagation code** to train the network.
- **Support multiple optimizers**: SGD, Momentum, Nesterov, RMSprop, Adam, and Nadam.
- **Perform hyperparameter tuning** using W&B sweeps.
- **Visualize training results**, including confusion matrices and misclassified samples.
- **Compare loss functions**: cross-entropy versus mean squared error.

All details regarding the required experiments, code specifications, and final observations are provided in the project report PDF.

## File Structure
- **`losses.py`**: Implements cross-entropy and mean squared error loss functions and their derivatives.
- **`model.py`**: Defines the neural network model using a flexible sequential API. Supports customizable hidden layers and weight initialization (random or Xavier).
- **`optimizers.py`**: Contains implementations of various optimization algorithms (SGD, Momentum, Nesterov, RMSprop, Adam, Nadam).
- **`activations.py`**: Provides activation functions (ReLU, Sigmoid, Tanh, Identity, Softmax) and their derivatives.
- **`train.py`**: The main training script that loads the dataset, builds the model, trains it, evaluates performance, and logs results to W&B.
- **`visualize.py`**: Visualizes one sample image per class from the Fashion-MNIST dataset.
- **`sweep.yaml` & `sweep_2.yaml`**: W&B sweep configuration files for hyperparameter tuning. sweep.yaml is created for the initial sweep (required in question 4) and the sweep_2.yaml is done with tuned set of hyperparameters for finding difference between cross-entropy loss and mean squared error.

## Requirements
- Python 3.x
- NumPy
- Matplotlib
- scikit-learn
- WandB (`pip install wandb`)
- Keras (for dataset loading; install via `pip install keras`)

## Setup and Installation
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/sakthe1010/da6401_assignment1.git
   cd da6401_assignment1
2. ** Install the required packages:**
   ```bash
   pip install numpy matplotlib scikit-learn wandb keras

## Running the code
1. Training the model:
  To train the model and log experiments to wandb run:
  ```bash
  python train.py --wandb_entity your_wandb_username --wandb_project your_project_name
  ```
You can adjust hyperparameters using command-line arguments. 

2. Hyperparameter Sweeps with wandb:
Two sweep configurations are provided to explore different hyperparameter combinations:

Initial Sweep: Use sweep.yaml to search over epochs, number of layers, hidden size, weight decay, learning rate, optimizer, batch size, weight initialization, and activation.
Focused Sweep: Use sweep_2.yaml to include loss function choices along with a tuned set of hyperparameters.
To run a sweep:

- Initialize the Sweep:
```bash
wandb sweep sweep.yaml
wandb agent <SWEEP_ID>
```
Repeat similarly for sweep_2.yaml if needed.

3. Visualizing Data Samples (Question 1)
Run the visualization script to display one sample image per class:
bash
```
python visualize.py
```
## Experimentation and Report
For detailed instructions, experiment setups, and insights, refer to the project report link:
https://wandb.ai/sakthebalan1010-iit-madras/ASSIGNMENT_1/reports/ME21B174-DA6401-Assignment-1--VmlldzoxMTgzMzk4MA

The report includes:
1. Results and analysis from hyperparameter sweeps.
2. Comparison of loss functions (cross-entropy vs. mean squared error).
3. Observations and recommendations based on experiments.
4. Final configurations and test accuracy, along with confusion matrix and misclassified sample analysis.

Github repo link: https://github.com/sakthe1010/da6401_assignment1

