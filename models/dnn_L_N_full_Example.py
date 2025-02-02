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

# define number of neurons in each hidden layer
num_neurons = 56

# Neural Network model with one hidden layers
model = models.Sequential([
    layers.InputLayer(shape=(28*28,)),  # Input layer
    layers.Dense(num_neurons, activation='relu'), # Input layer
    layers.Dense(num_neurons, activation='relu'),     # First hidden layer
    layers.Dense(num_neurons, activation='relu'),     # Second hidden layer
    layers.Dense(num_neurons, activation='relu'),     # 3th hidden layer
    layers.Dense(num_neurons, activation='relu'),     # 4th hidden layer
    layers.Dense(num_neurons, activation='relu'),     # 5th hidden layer
    # next 5 layers
    layers.Dense(num_neurons, activation='relu'),     # First hidden layer
    layers.Dense(num_neurons, activation='relu'),     # Second hidden layer
    layers.Dense(num_neurons, activation='relu'),     # 3th hidden layer
    layers.Dense(num_neurons, activation='relu'),     # 4th hidden layer
    layers.Dense(num_neurons, activation='relu'),     # 5th hidden layer


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
model.fit(x_train, y_train, epochs=11, batch_size=64, validation_data=(x_test, y_test))
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc:.2f}')

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
dense_weights_in = [[int(model.layers[0].weights[0][i][j]*1e18) for j in range(num_neurons)] for i in range(784)]
dense_bias_in = [int(model.layers[0].weights[1][i]*1e36) for i in range(num_neurons)]
X_in, dense_weights_in, dense_bias_in, dense_in_out, dense_remainder_in = DenseInt(784, num_neurons, 10**18, X_in, dense_weights_in, dense_bias_in)

relu_in_in = [int(dense_in_out[i]) for i in range(num_neurons)]
relu_out_in = [str(relu_in_in[i]) if relu_in_in[i] < p//2 else '0' for i in range(num_neurons)]
dense_input_h1=[int(x) for x in relu_out_in]
print("len model.layers:", len(model.layers))

num_hidden_layers = len(model.layers)-3
print("Num_hidden_layers:", num_hidden_layers)
print("Num neurons:", num_neurons)
dense_input_next = dense_input_h1
backbone=[]
for i in range(1,num_hidden_layers+1):
 
    dense_weights_hi = [[int(model.layers[i].weights[0][k][j]*1e18) for j in range(num_neurons)] for k in range(num_neurons)]
    dense_bias_hi = [int(model.layers[i].weights[1][j]*1e36) for j in range(num_neurons)]
    _, dense_weights_hi, dense_bias_hi, dense_hi_out, dense_remainder_hi = DenseInt(num_neurons, num_neurons, 10**18, dense_input_next, dense_weights_hi, dense_bias_hi)
    relu_in_hi = [int(dense_hi_out[j]) for j in range(num_neurons)]
    relu_out_hi = [str(relu_in_hi[j]) if relu_in_hi[j] < p//2 else '0' for j in range(num_neurons)]
    dense_input_next=[int(x) for x in relu_out_hi]
    backbone.append({
    "weight": dense_weights_hi,
    "bias": dense_bias_hi,
    "dense_out": dense_hi_out,
    "remainder": dense_remainder_hi,
    "activation": relu_out_hi,
    })


out_layer_index = num_hidden_layers+1
dense_weights_out = [[int(model.layers[out_layer_index].weights[0][i][j]*1e18) for j in range(10)] for i in range(num_neurons)]
dense_bias_out = [int(model.layers[out_layer_index].weights[1][i]*1e36) for i in range(10)]
_, dense_weights_out, dense_bias_out, dense_out_out, dense_remainder_out = DenseInt(num_neurons, 10, 10**18, dense_input_next, dense_weights_out, dense_bias_out)


print("dense out:", dense_out_out)
# dense_model = Model(inputs, model.layers[-2].output)
# prediction=dense_model.predict(X.reshape(1,28,28,1))[0]
# print ("prediction orig:", prediction.argmax())


# Convert to integers and compute modulo p
# mod_values = [(p-int(x)) for x in dense_out]
# print("circiut output final:", np.array(mod_values).argmax())
dense_out = [(int(x)+1e18)%p for x in dense_out_out]
print("relue_out:", dense_out)
circiut_prediction=np.array(dense_out).argmax()
print("circit prediction:",circiut_prediction)
print ("backbone length:", len(backbone))
in_json={
        "x": X_in,
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
outPutFile="./data/circiut_inputs/mnist_input_"+str(num_hidden_layers)+"_"+str(num_neurons)+"_dig"+str(Y)+".json"
with open(outPutFile, "w") as f:
    json.dump(in_json, f)