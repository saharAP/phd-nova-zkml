import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import json

# Loading the saved model for inference
model = tf.keras.models.load_model('./data/mnist_DNN_model.keras')

# Accessing the weights from layer 2 and layer 3
layer_0_weights = model.layers[0].get_weights() 
layer_2_weights = model.layers[1].get_weights()  # Assuming layer 2 is at index 0 (first hidden layer)
layer_3_weights = model.layers[2].get_weights()  # Assuming layer 3 is at index 1 (classification layer)

# Prepare the weights in a serializable format (convert to lists)
weights_to_save = {
        "layer_input": {
        "weights": layer_0_weights[0].tolist(),  # Weight matrix of layer input
        "biases": layer_0_weights[1].tolist()    # Bias vector of layer input
    },
    "layer_1": {
        "weights": layer_2_weights[0].tolist(),  # Weight matrix of layer 1
        "biases": layer_2_weights[1].tolist()    # Bias vector of layer 1
    },
    "layer_out": {
        "weights": layer_3_weights[0].tolist(),  # Weight matrix of layer output
        "biases": layer_3_weights[1].tolist()    # Bias vector of layer output
    }
}

# Convert the weights to JSON format
weights_json = json.dumps(weights_to_save, indent=4)

# Save the weights into a JS file
with open("./data/mnist_dnn_model_weights.js", "w") as js_file:
    js_file.write(f"const modelWeights = {weights_json};\n")
    js_file.write("export default modelWeights;\n")

print("Weights have been saved to model_weights.js")