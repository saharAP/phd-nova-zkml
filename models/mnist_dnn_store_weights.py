import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import json

# Loading the saved model for inference
model = tf.keras.models.load_model('./data/mnist_DNN_model.keras')

# Accessing the weights from layer 2 and layer 3
layer_in_weights = model.layers[0].get_weights() 
layer_h1_weights = model.layers[1].get_weights()  # Assuming layer 2 is at index 0 (first hidden layer)
layer_h2_weights = model.layers[2].get_weights()  # Assuming layer 3 is at index 2 (classification layer)
layer_out_weights = model.layers[3].get_weights()  # Assuming layer 3 is at index 3 (classification layer)

# Prepare the weights in a serializable format (convert to lists)
weights_to_save = {
        "layer_in": {
        "weights": layer_in_weights[0].tolist(),  # Weight matrix of layer input
        "biases": layer_in_weights[1].tolist()    # Bias vector of layer input
    },
    "layer_h1": {
        "weights": layer_h1_weights[0].tolist(),  # Weight matrix of layer 1
        "biases": layer_h1_weights[1].tolist()    # Bias vector of layer 1
    },
    "layer_h2": {
        "weights": layer_h2_weights[0].tolist(),  # Weight matrix of layer 1
        "biases": layer_h2_weights[1].tolist()    # Bias vector of layer 1
    },
    "layer_out": {
        "weights": layer_out_weights[0].tolist(),  # Weight matrix of layer output
        "biases": layer_out_weights[1].tolist()    # Bias vector of layer output
    }
}

# Convert the weights to JSON format
weights_json = json.dumps(weights_to_save, indent=4)

# Save the weights into a JS file
with open("./data/mnist_dnn_model_weights.js", "w") as js_file:
    js_file.write(f"const modelWeights = {weights_json};\n")
    js_file.write("export default modelWeights;\n")

print("Weights have been saved to model_weights.js")