import nnfs
from nnfs.datasets import spiral_data
import numpy as np
import torch
from ModelClasses import *
nnfs.init()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create dataset
X, y = spiral_data(samples=1000, classes=3)
X_test, y_test = spiral_data(samples=100, classes=3)

# Instantiate the model
model = Model()

# Add layers
model.add(Layer_Dense(2, 256, weight_regularizer_l2=5e-4,
                              bias_regularizer_l2=5e-4))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.1))
model.add(Layer_Dense(256, 3))
model.add(Activation_Softmax())

# Set loss, optimizer and accuracy objects
model.set(
    loss=Loss_CategoricalCrossentropy(),
    optimizer=Optimizer_Adam(learning_rate=0.05, decay=5e-5),
    accuracy=Accuracy_Categorical()
)

# Finalize the model
model.finalize()

# Train the model
model.train(X, y, validation_data=(X_test, y_test),
            epochs=10000, print_every=100)


import matplotlib.pyplot as plt


# Generate a grid of points with distance h between them
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Flatten the grid to pass through the model
grid = np.c_[xx.ravel(), yy.ravel()]

# Perform a forward pass with the grid as input
Z = model.forward(grid, training=False)

# For models ending with a softmax layer, take the argmax to get predicted class labels
if Z.ndim > 1:  
    Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)

# Plot the contour and training examples
plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Model Fit Visualization')
plt.show()

