# from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Softmax, Dense, Lambda, BatchNormalization, ReLU
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers.legacy import SGD
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import json

p = 21888242871839275222246405745257275088548364400416034343698204186575808495617

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

index=4
X = x_test[index]
Y= y_test[index]
print("Label for x_test[1]:", Y)

print("label:", Y)

def DenseInt(nInputs, nOutputs, n, input, weights, bias):
    Input = [str(input[i] % p) for i in range(nInputs)]
    Weights = [[str(weights[i][j] % p) for j in range(nOutputs)] for i in range(nInputs)]
    Bias = [str(bias[i] % p) for i in range(nOutputs)]
    out = [0 for _ in range(nOutputs)]
    remainder = [None for _ in range(nOutputs)]
    for j in range(nOutputs):
        for i in range(nInputs):
            out[j] += input[i] * weights[i][j]
        out[j] += bias[j]
        remainder[j] = str(out[j] % n)
        out[j] = str(out[j] // n % p)
    return Input, Weights, Bias, out, remainder

print("len", len(X))
X_in =[int(X[i]*1e18) for i in range(784)]
dense_weights_in = [[int(model.layers[0].weights[0][i][j]*1e18) for j in range(10)] for i in range(784)]
dense_bias_in = [int(model.layers[0].weights[1][i]*1e36) for i in range(10)]
X_in, dense_weights_in, dense_bias_in, dense_in_out, dense_remainder_in = DenseInt(784, 10, 10**18, X_in, dense_weights_in, dense_bias_in)

relu_in_in = [int(dense_in_out[i]) for i in range(10)]
relu_out_in = [str(relu_in_in[i]) if relu_in_in[i] < p//2 else 0 for i in range(10)]
dense_input_h1=[int(x) for x in relu_out_in]

dense_weights_h1 = [[int(model.layers[1].weights[0][i][j]*1e18) for j in range(10)] for i in range(10)]
dense_bias_h1 = [int(model.layers[1].weights[1][i]*1e36) for i in range(10)]
_, dense_weights_h1, dense_bias_h1, dense_h1_out, dense_remainder_h1 = DenseInt(10, 10, 10**18, dense_input_h1, dense_weights_h1, dense_bias_h1)

relu_in_h1 = [int(dense_h1_out[i]) for i in range(10)]
relu_out_h1 = [str(relu_in_h1[i]) if relu_in_h1[i] < p//2 else 0 for i in range(10)]
dense_input_h2=[int(x) for x in relu_out_h1]

dense_weights_h2 = [[int(model.layers[2].weights[0][i][j]*1e18) for j in range(10)] for i in range(10)]
dense_bias_h2 = [int(model.layers[2].weights[1][i]*1e36) for i in range(10)]
_, dense_weights_h2, dense_bias_h2, dense_h2_out, dense_remainder_h2 = DenseInt(10, 10, 10**18, dense_input_h2, dense_weights_h2, dense_bias_h2)

relu_in_h2 = [int(dense_h2_out[i]) for i in range(10)]
relu_out_h2 = [str(relu_in_h2[i]) if relu_in_h2[i] < p//2 else 0 for i in range(10)]
dense_input_out=[int(x) for x in relu_out_h2]

dense_weights_out = [[int(model.layers[3].weights[0][i][j]*1e18) for j in range(10)] for i in range(10)]
dense_bias_out = [int(model.layers[3].weights[1][i]*1e36) for i in range(10)]
_, dense_weights_out, dense_bias_out, dense_out_out, dense_remainder_out = DenseInt(10, 10, 10**18, dense_input_out, dense_weights_out, dense_bias_out)


print("dense out:", dense_out_out)
# dense_model = Model(inputs, model.layers[-2].output)
# prediction=dense_model.predict(X.reshape(1,28,28,1))[0]
# print ("prediction orig:", prediction.argmax())


# Convert to integers and compute modulo p
# mod_values = [(p-int(x)) for x in dense_out]
# print("circiut output final:", np.array(mod_values).argmax())
dense_out = [(int(x)+1e18)%p for x in dense_out_out]
print("dens_out:", dense_out)
circiut_prediction=np.array(dense_out).argmax()
print("circit prediction:",circiut_prediction)

backbone=[]

backbone.append({
    "weight": dense_weights_h1,
    "bias": dense_bias_h1,
    "dense_out": dense_h1_out,
    "remainder": dense_remainder_h1,
    "activation": relu_out_h1,
})

backbone.append({
    "weight": dense_weights_h2,
    "bias": dense_bias_h2,
    "dense_out": dense_h2_out,
    "remainder": dense_remainder_h2,
    "activation": relu_out_h2,
})

in_json={
        "in": X_in,
        "head":{
            "weight": dense_weights_in,
            "bias": dense_bias_in,
            "dense_out": dense_in_out,
            "remainder": dense_remainder_in,
            "activation": relu_out_in
        },
        "backbone":backbone,
        "tail":{
            "weight": dense_weights_out,
            "bias": dense_bias_out,
            "dense_out": dense_out_out,
            "remainder": dense_remainder_out,
            "activation": str(circiut_prediction)
        }
    }
outPutFile="./data/circiut_inputs/mnist_input_"+str(Y)+".json"
with open(outPutFile, "w") as f:
    json.dump(in_json, f)