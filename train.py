import argparse
import numpy as np
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


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

    x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1) / 255.0
    num_classes = 10

    y_train_oh = np.zeros((y_train.shape[0], num_classes))
    for i, label in enumerate(y_train):
        y_train_oh[i, label] = 1
    y_test_oh = np.zeros((y_test.shape[0], num_classes))
    for i, label in enumerate(y_test):
        y_test_oh[i, label] = 1

    return x_train, y_train_oh, x_test, y_test_oh

def train_val_split(x, y, val_fraction=0.1):
    num_samples = x.shape[0]
    val_size = int(num_samples * val_fraction)
    indices = np.random.permutation(num_samples)
    x = x[indices]
    y = y[indices]
    x_val = x[:val_size]
    y_val = y[:val_size]
    x_train = x[val_size:]
    y_train = y[val_size:]
    return (x_train, y_train), (x_val, y_val)

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
    
def accuracy(y_pred, y_true):
    return np.mean(np.argmax(y_pred, axis=0) == np.argmax(y_true, axis=0))
    
if __name__ == "__main__":
    args = parse_args()

    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
    config = wandb.config

    args.epochs = config.epochs
    args.num_layers = config.num_layers
    args.hidden_size = config.hidden_size
    args.weight_decay = config.weight_decay
    args.learning_rate = config.learning_rate
    args.optimizer = config.optimizer
    args.batch_size = config.batch_size
    args.weight_init = config.weight_init
    args.activation = config.activation

    run_name = (f"ep_{args.epochs}_nl_{args.num_layers}_hs_{args.hidden_size}_wd_{args.weight_decay}_"
                f"lr_{args.learning_rate}_bs_{args.batch_size}_{args.optimizer}_{args.weight_init}_{args.activation}")
    wandb.run.name = run_name  
    wandb.run.save()

    x_train_full, y_train_full, x_test, y_test = load_data(args.dataset)
    (x_train, y_train), (x_val, y_val) = train_val_split(x_train_full, y_train_full, 0.1)

    model = create_model(args)

    opt_class = OPTIMIZERS[args.optimizer]
    opt_params = {
        "lr": args.learning_rate,
        "momentum": args.momentum,
        "beta": args.beta,
        "beta1": args.beta1,
        "beta2": args.beta2,
        "epsilon": args.epsilon,
        "weight_decay": args.weight_decay
    }
    optimizer = opt_class(**opt_params)

    loss_func, _ = LOSSES[args.loss]

    num_samples = x_train.shape[0]
    batch_size = args.batch_size
    for epoch in range(args.epochs):
        indices = np.random.permutation(num_samples)
        x_train = x_train[indices]
        y_train = y_train[indices]

        total_loss = 0.0
        total_acc = 0.0
        n_batches = num_samples // batch_size

        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            x_batch = x_train[start:end]
            y_batch = y_train[start:end]

            pre_acts, acts = model.forward(x_batch.T)
            loss_value = loss_func(acts[-1], y_batch.T)
            total_loss += loss_value

            acc = accuracy(acts[-1], y_batch.T)
            total_acc += acc

            grads_W, grads_b = model.backward(y_batch.T, pre_acts, acts, args.loss)
            optimizer.update(model, grads_W, grads_b)

        avg_train_loss = total_loss / n_batches if n_batches else total_loss
        avg_train_acc = total_acc / n_batches if n_batches else total_acc

        pre_acts_val, acts_val = model.forward(x_val.T)
        val_loss = loss_func(acts_val[-1], y_val.T)
        val_acc = accuracy(acts_val[-1], y_val.T)

        pre_acts_test, acts_test = model.forward(x_test.T)
        test_loss = loss_func(acts_test[-1], y_test.T)
        test_acc = accuracy(acts_test[-1], y_test.T)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_accuracy": avg_train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "test_loss": test_loss,
            "test_accuracy": test_acc
        })
        print(f"Epoch [{epoch+1}/{args.epochs}] train_loss: {avg_train_loss:.4f}, train_acc: {avg_train_acc:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}, test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}")

    print("Training complete.")

    pre_acts_test, acts_test = model.forward(x_test.T)
    y_pred_test = np.argmax(acts_test[-1], axis=0)
    y_true_test = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_true_test, y_pred_test)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=[str(i) for i in range(10)])
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_title("Confusion Matrix (Test Set)")

    wandb.log({"confusion_matrix": wandb.Image(fig)})

    plt.close(fig)

