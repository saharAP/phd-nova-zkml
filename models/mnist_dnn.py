import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist


# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype('float32') / 255.0  # Flatten and normalize
x_test = x_test.reshape(-1, 28*28).astype('float32') / 255.0

# Neural Network model with two hidden layers
# model = models.Sequential([
#     layers.InputLayer(shape=(28*28,)),  # Input layer
#     layers.Dense(256, activation='relu'),     # First hidden layer
#     layers.Dense(128, activation='relu'),     # Second hidden layer
#     layers.Dense(10, activation='softmax')    # Output layer (10 classes for digits 0-9)
# ])

# Neural Network model with one hidden layers
model = models.Sequential([
    layers.InputLayer(shape=(28*28,)),  # Input layer
    layers.Dense(10, activation='relu'), # Input layer
    layers.Dense(10, activation='relu'),     # First hidden layer
    layers.Dense(10, activation='relu'),     # Second hidden layer
    layers.Dense(10, activation=None),     # Output layer (10 classes for digits 0-9)
    layers.Softmax(),    # Output layer (10 classes for digits 0-9)
    # layers.Dense(10, activation='softmax')    # Output layer (10 classes for digits 0-9)
])
# print shape of the input layer
print("input shape:",model.input_shape)

model.summary()
# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc:.2f}')

# Saving the model after training
model.save('./data/mnist_DNN_model.keras')
# accuracy is 94% for this configuration