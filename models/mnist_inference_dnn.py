import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import json
# Loading the saved model for inference
model = tf.keras.models.load_model('./data/mnist_DNN_model.keras')

# Load the MNIST dataset (for demonstration purposes, we'll use the test set)
(_, _), (x_test, y_test) = mnist.load_data()

# For this example, let's assume the model is still in memory (from previous training)
# You can load any digit image from x_test to make a prediction, let's take the first one:
index = 10  # You can change this index to use a different digit from the test set
image = x_test[index]

# Display the image
plt.imshow(image, cmap='gray')
plt.title(f"True label: {y_test[index]}")
plt.show()

# Preprocess the image for the model

image = image.reshape(1, 28 * 28).astype('float32') / 255.0  # Flatten and normalize
print("image:", len(image))
# Make a prediction
prediction = model.predict(image)
print("prediction:", prediction)
# Get the predicted digit
predicted_digit = np.argmax(prediction)

# Display the prediction
print(f"The model predicts this digit is: {predicted_digit}")

 # Load and example image as test from data/example_images.json
with open("./data/example_images.json", "r") as json_file:
        example_images = json.load(json_file)

image_a_7=np.array(example_images['image_d_0'])
print("image::", image)
print("len image::", len(image))
print("len image[0]:", len(image[0]))
print("len image_a_7:", len(image_a_7))
print("len(image_a_7[0]):", len(image_a_7[0]))
print ("image_a_7:", image_a_7)
prediction=model.predict(image_a_7)
print("prediction:", prediction)
predicted_digit1=np.argmax(prediction)
    
print("predicted_digit:", predicted_digit1)