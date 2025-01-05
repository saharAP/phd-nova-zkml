import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import json

# Loading the saved model for inference
model = tf.keras.models.load_model('./data/mnist_DNN_model.keras')
# Scale factor for weights and biases
scale_factor_w = 1e18
scale_factor_b = 1e36

# Accessing the weights from layer 2 and layer 3
layer_in_weights = model.layers[0].get_weights() 
layer_h1_weights = model.layers[1].get_weights()  # Assuming layer 2 is at index 0 (first hidden layer)
layer_h2_weights = model.layers[2].get_weights()  # Assuming layer 3 is at index 2 (classification layer)
layer_out_weights = model.layers[3].get_weights()  # Assuming layer 3 is at index 3 (classification layer)

# Prepare the weights in a serializable format (convert to lists)
weights_to_save = {
        "layer_in": {
        "weights": [[int(layer_in_weights[0][i][j]*scale_factor_w) for i in range(784)] for j in range(10)],  # Weight matrix of layer input
        "biases": [int(layer_in_weights[1][i]*scale_factor_b) for i in range(10)]    # Bias vector of layer input
    },
    "layer_h1": {
        "weights": [[int(layer_h1_weights[0][i][j]*scale_factor_w) for i in range(10)] for j in range(10)],  # Weight matrix of layer 1
        "biases": [int(layer_h1_weights[1][i]*scale_factor_b) for i in range(10)]    # Bias vector of layer 1
    },
    "layer_h2": {
        "weights": [[int(layer_h2_weights[0][i][j]*scale_factor_w) for i in range(10)] for j in range(10)],  # Weight matrix of layer 1
        "biases": [int(layer_h2_weights[1][i]*scale_factor_b) for i in range(10)]   # Bias vector of layer 1
    },
    "layer_out": {
        "weights": [[int(layer_out_weights[0][i][j]*scale_factor_w) for i in range(10)] for j in range(10)],  # Weight matrix of layer output
        "biases": [int(layer_out_weights[1][i]*scale_factor_b) for i in range(10)]   # Bias vector of layer output
    }
}
print("out bias:",layer_out_weights[1])
print("out weights:",[int(layer_out_weights[1][i]*1e36) for i in range(10)])
print("Len",len(layer_in_weights[0][0]))
# [int(model.layers[1].weights[1][i]*1e36) for i in range(4)]
# Convert the weights to JSON format
weights_json = json.dumps(weights_to_save, indent=4)
save_to_file = "./data/mnist_dnn_model_weights.json"
# Save the weights into a Json file
with open(save_to_file, "w") as json_file:
    json_file.write(weights_json)

print("Weights have been saved to "+save_to_file)