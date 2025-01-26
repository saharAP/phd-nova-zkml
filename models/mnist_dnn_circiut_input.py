# the script generates circuits
from mnist_circiut_API import *
import json
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

def simulateModelOnImage(modelWeights,image, scale_factor_w):
    # # run inference for the first layer
    
    X_in, dense_weights_in, dense_bias_in, out, dense_remainder_in = DenseInt(784, 10, 10**scale_factor_w, image, modelWeights['layer_in']['weights'], modelWeights['layer_in']['biases'])
    # Apply  relu activation function to the output of the first layer
    relu_in_in = [int(out[i]) for i in range(10)]
    relu_out_in = [str(relu_in_in[i]) if relu_in_in[i] < p//2 else 0 for i in range(10)]
    dense_input_1=[int(x) for x in relu_out_in]
    print("dense_input_1:", dense_input_1)
    # # run inference for the second layer
    _, dense_weights_1, dense_bias_1, dense_out_1, dense_remainder_1 = DenseInt(10, 10, 10**scale_factor_w, dense_input_1, modelWeights['layer_h1']['weights'], modelWeights['layer_h1']['biases'])
    print("dense_out_1:", dense_out_1)
    int_numbers = [int(num) for num in dense_out_1]
    print("int_numbers:", int_numbers)
    predicted_digit=np.argmax(int_numbers)
    return predicted_digit
def saveCircuitInputParameters(modelWeights,image, scale_factor_w, argmax_out):
    # # run inference for the first layer
    X_in, dense_weights_in, dense_bias_in, out, dense_remainder_in = DenseInt(784, 10, 10**scale_factor_w, image, modelWeights['layer_in']['weights'], modelWeights['layer_in']['biases'])
    # Apply  relu activation function to the output of the first layer
    relu_in_in = [int(out[i]) for i in range(10)]
    relu_out_in = [str(relu_in_in[i]) if relu_in_in[i] < p//2 else 0 for i in range(10)]
    dense_input_1=[int(x) for x in relu_out_in]
    print("dense_input_1:", dense_input_1)
    # # run inference for the second layer
    _, dense_weights_1, dense_bias_1, dense_out_1, dense_remainder_1 = DenseInt(10, 10, 10**scale_factor_w, dense_input_1, modelWeights['layer_h1']['weights'], modelWeights['layer_h1']['biases'])
  
    # Apply  relu activation function to the output of the first layer
    relu_in_1 = [int(dense_out_1[i]) for i in range(10)]
    relu_out_1 = [str(relu_in_1[i]) if relu_in_1[i] < p//2 else 0 for i in range(10)]
    dense_input_2=[int(x) for x in relu_out_1]

    _, dense_weights_2, dense_bias_2, dense_out_2, dense_remainder_2 = DenseInt(10, 10, 10**scale_factor_w, dense_input_2, modelWeights['layer_h2']['weights'], modelWeights['layer_h2']['biases'])
  
    # Apply  relu activation function to the output of the first layer
    relu_in_2 = [int(dense_out_2[i]) for i in range(10)]
    relu_out_2 = [str(relu_in_2[i]) if relu_in_2[i] < p//2 else 0 for i in range(10)]
    dense_input_out=[int(x) for x in relu_out_2]
    print("dense_input_out:", dense_input_out)
    _, dense_weights_out, dense_bias_out, dense_out_out, dense_remainder_out = DenseInt(10, 10, 10**scale_factor_w, dense_input_out, modelWeights['layer_out']['weights'], modelWeights['layer_out']['biases'])

    # Apply argmax activation function to the output of the last layer
    dense_out_out = [(int(x)+1e18)%p for x in dense_out_out]
    # print("dens_out:", dense_out_out)
    # print("circit prediction:",np.array(dense_out_out).argmax())
    # print("argmax_out:", argmax_out)
    return np.array(dense_out_out).argmax()
    # in_json={
    #     "in": X_in,
    #     "head":{
    #         "weight": dense_weights_1,
    #         "bias": dense_bias_1,
    #         "dense_out": out,
    #         "remainder": dense_remainder_1,
    #         "activation": relu_out_1,
    #     },
    #     "tail":{
    #         "weight": dense_weights_2,
    #         "bias": dense_bias_2,
    #         "dense_out": dense_out_2,
    #         "remainder": dense_remainder_2,
    #         "activation": str(argmax_out)
    #     }
    # }
    # outPutFile="./data/circiut_inputs/mnist_input"+str(argmax_out)+".json"
    # with open(outPutFile, "w") as f:
    #     json.dump(in_json, f)



def main():
 
    scale_factor = 18
    model_weights_Int_file = "./data/mnist_dnn_model_weights.json"
    print("Running inference on the MNIST DNN model using the generated circuit")
    # Load the weights from the JSON file
    with open(model_weights_Int_file, "r") as json_file:
        modelWeights = json.load(json_file)
  
    # Load and example image as test from data/example_images.json
    # with open("./data/example_images.json", "r") as json_file:
    #     example_images = json.load(json_file)
    
    # image=np.array(example_images['image_j_9'][0])
    # image= (image * 10**scale_factor_img).astype(int)
    # print("len image:", len(image))
    (_, _), (x_test, y_test) = mnist.load_data()
    print ("y_test:", len(y_test))
    x_test = x_test.reshape(-1, 28*28).astype('float32') / 255.0
 
    accuraccy=0
    x_test=x_test[:3]
    for i in range(3):
        # image = (image * 10**scale_factor_img).astype(int)
        # print("image:", len(image))
        X_in=[int(x*10*scale_factor) for x in x_test[i]]
        # predicted_digit = simulateModelOnImage(modelWeights,X_in, scale_factor)
        predicted_digit=saveCircuitInputParameters(modelWeights,X_in, scale_factor, y_test[i])
        if predicted_digit==y_test[i]:
            accuraccy+=1
        
        # check if the predicted digit is correct

    print("accuraccy:", accuraccy)
  

if __name__ == "__main__":
    main()
