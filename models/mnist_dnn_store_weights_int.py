import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import json


# Converts all decimal numbers in weights and biases to integers
def remove_decimal_points(n, array):
    # multiply the array by 10^n and convert to integer
    return (array * 10**n).astype(int)

# Loading the saved model for inference
model = tf.keras.models.load_model('./data/mnist_DNN_model.keras')

# Accessing the weights from layer 2 and layer 3
layer_0_weights = model.layers[0].get_weights() 
layer_1_weights = model.layers[2].get_weights()  # Assuming layer 2 is at index 0 (first hidden layer)
# layer_2_weights = model.layers[2].get_weights()  # Assuming layer 3 is at index 1 (classification layer)
print("layer_1_weights[0]:", layer_1_weights[0])
scale_factor_w = 18
scale_factor_b = 18
# apply the remove_decimal_points function to the weights and biases
layer_0_weights[0] = (layer_0_weights[0]  * 10**scale_factor_w).astype(int)
layer_0_weights[1]=(layer_0_weights[1]  * 10**scale_factor_b).astype(int)
layer_1_weights[0] = (layer_1_weights[0]  * 10**scale_factor_w).astype(int)
layer_1_weights[1]=(layer_1_weights[1]  * 10**scale_factor_b).astype(int)
# layer_2_weights[0] = (layer_2_weights[0]  * 10**scale_factor_w).astype(int)
# layer_2_weights[1]=(layer_2_weights[1]  * 10**scale_factor_b).astype(int)


# Prepare the weights in a serializable format (convert to lists)
weights_to_save = {
        "layer_input": {
        "weights": layer_0_weights[0].tolist(),  # Weight matrix of layer input
        "biases": layer_0_weights[1].tolist()    # Bias vector of layer input
    },
    "layer_1": {
        "weights": layer_1_weights[0].tolist(),  # Weight matrix of layer 1
        "biases": layer_1_weights[1].tolist()    # Bias vector of layer 1
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

# with open("./data/mnist_dnn_model_weights.js", "w") as js_file:
#     js_file.write(f"const modelWeights = {weights_json};\n")
#     js_file.write("export default modelWeights;\n")

print("Weights have been saved to model_weights.json")


