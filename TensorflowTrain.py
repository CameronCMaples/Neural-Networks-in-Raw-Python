import nnfs
from nnfs.datasets import spiral_data
import numpy as np
import torch
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

nnfs.init()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create dataset
X, y = spiral_data(samples=1000, classes=3)

X = np.array(X)  

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = tf.keras.models.Sequential([

    tf.keras.layers.Flatten(input_shape = X.shape[1:]),
    tf.keras.layers.Dense(256, activation = 'relu', name = 'Layer1'),
    tf.keras.layers.Dropout(0.1, name = 'DropoutLayer'),
    tf.keras.layers.Dense(256, activation = 'relu', name= 'Layer2'),
    tf.keras.layers.Dense(3, activation = tf.nn.softmax, name = 'Output')

])

class PrintLossCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 100 == 0:  # Every 100 epochs
            loss = logs.get('loss')
            print(f"Epoch {epoch}, Loss: {loss}")

model.compile(optimizer='Adam', loss= 'sparse_categorical_crossentropy', metrics= ['accuracy'])

model.fit(X_train, y_train, epochs=1000, callbacks=[PrintLossCallback()])

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

#Visualize Data
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)

# Plot the original spiral data points
scatter = plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')

# Add a legend
legend1 = plt.legend(*scatter.legend_elements(), title="Classes")
ax = plt.gca()
ax.add_artist(legend1)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Spiral Data and Model Decision Boundaries')
plt.show()




