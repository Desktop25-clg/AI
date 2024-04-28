import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np

# Load and preprocess the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define and compile a single-layer neural network
model_single_layer = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(4,)),
    layers.Dense(3, activation='softmax')
])

model_single_layer.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

# Train the single-layer neural network
model_single_layer.fit(X_train, y_train, epochs=22, validation_data=(X_test, y_test))

# Evaluate the single-layer model
y_pred_single_layer = np.argmax(model_single_layer.predict(X_test), axis=1)
single_layer_accuracy = accuracy_score(y_test, y_pred_single_layer)
print(f"\nSingle-layer Neural Network - Accuracy: {single_layer_accuracy}")

# Define and compile a multi-layer neural network
model_multi_layer = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(4,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(3, activation='softmax')
])

model_multi_layer.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

# Train the multi-layer neural network
model_multi_layer.fit(X_train, y_train, epochs=22, validation_data=(X_test, y_test))

# Evaluate the multi-layer model
y_pred_multi_layer = np.argmax(model_multi_layer.predict(X_test), axis=1)
multi_layer_accuracy = accuracy_score(y_test, y_pred_multi_layer)
print(f"\nMulti-layer Neural Network - Accuracy: {multi_layer_accuracy}")