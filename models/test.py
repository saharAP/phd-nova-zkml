import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import json


# Loading the saved model for inference
model = tf.keras.models.load_model('./data/mnist_DNN_model.keras')

layer_0_weights=[[int(model.layers[0].weights[0][i][j]*1e18) for j in range(10)] for i in range(784)]
layer_0_biases=[int(model.layers[0].weights[1][i]*1e36) for i in range(10)]

layer_1_weights=[[int(model.layers[2].weights[0][i][j]*1e18) for j in range(10)] for i in range(10)]
layer_1_biases=[int(model.layers[2].weights[1][i]*1e36) for i in range(10)]


# Prepare the weights in a serializable format (convert to lists)
weights_to_save = {
        "layer_input": {
        "weights": layer_0_weights,  # Weight matrix of layer input
        "biases": layer_0_biases    # Bias vector of layer input
    },
    "layer_1": {
        "weights": layer_1_weights,  # Weight matrix of layer 1
        "biases": layer_1_biases    # Bias vector of layer 1
    }
    # ,
    # "layer_out": {
    #     "weights": layer_2_weights[0].tolist(),  # Weight matrix of layer output
    #     "biases": layer_2_weights[1].tolist()    # Bias vector of layer output
    # }
}

# Convert the weights to JSON format
weights_json = json.dumps(weights_to_save, indent=4)

# Save the weights into a Json file
with open("./data/mnist_dnn_model_weights.json", "w") as json_file:
    json_file.write(weights_json)


print("Weights have been saved to model_weights.json")


