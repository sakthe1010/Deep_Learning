import numpy as np
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

samples = []
labels = []

for class_label in range(10):
    index = np.where(y_train == class_label)[0][0]
    samples.append(x_train[index])
    labels.append(y_train[index])
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
axes = axes.ravel()

for i in range(10):
    axes[i].imshow(samples[i], cmap='gray')
    axes[i].set_title(f"Class: {labels[i]}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()