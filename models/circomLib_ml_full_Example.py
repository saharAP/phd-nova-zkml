from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Softmax, Dense, Lambda, BatchNormalization, ReLU
from tensorflow.keras import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers.legacy import SGD
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

p = 21888242871839275222246405745257275088548364400416034343698204186575808495617

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Convert y_train into one-hot format
temp = []
for i in range(len(y_train)):
    temp.append(to_categorical(y_train[i], num_classes=10))
y_train = np.array(temp)
# Convert y_test into one-hot format
temp = []
for i in range(len(y_test)):    
    temp.append(to_categorical(y_test[i], num_classes=10))
y_test = np.array(temp)

#reshaping
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

#normalizing
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.0
X_test /= 255.0

inputs = Input(shape=(28,28,1))
out = Conv2D(4, 3, use_bias=True)(inputs)
out = BatchNormalization()(out)
out = ReLU()(out)
out = AveragePooling2D(pool_size=(2,2))(out)
out = Conv2D(8, 3, use_bias=True)(out)
out = BatchNormalization()(out)
out = ReLU()(out)
out = AveragePooling2D(pool_size=(2,2))(out)
out = Flatten()(out)
out = Dense(10, activation=None)(out)
out = Softmax()(out)
model = Model(inputs, out)

model.summary()

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['acc']
    )

model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

X = X_test[0]
X.shape, X.min(), X.max()
model.predict(X.reshape(1,28,28,1)).argmax()
print("label for x_test[0]:", y_test[11])

def Conv2DInt(nRows, nCols, nChannels, nFilters, kernelSize, strides, n, input, weights, bias):
    Input = [[[str(input[i][j][k] % p) for k in range(nChannels)] for j in range(nCols)] for i in range(nRows)]
    Weights = [[[[str(weights[i][j][k][l] % p) for l in range(nFilters)] for k in range(nChannels)] for j in range(kernelSize)] for i in range(kernelSize)]
    Bias = [str(bias[i] % p) for i in range(nFilters)]
    out = [[[0 for _ in range(nFilters)] for _ in range((nCols - kernelSize)//strides + 1)] for _ in range((nRows - kernelSize)//strides + 1)]
    remainder = [[[None for _ in range(nFilters)] for _ in range((nCols - kernelSize)//strides + 1)] for _ in range((nRows - kernelSize)//strides + 1)]
    for i in range((nRows - kernelSize)//strides + 1):
        for j in range((nCols - kernelSize)//strides + 1):
            for m in range(nFilters):
                for k in range(nChannels):
                    for x in range(kernelSize):
                        for y in range(kernelSize):
                            out[i][j][m] += input[i*strides+x][j*strides+y][k] * weights[x][y][k][m]
                out[i][j][m] += bias[m]
                remainder[i][j][m] = str(out[i][j][m] % n)
                out[i][j][m] = str(out[i][j][m] // n % p)
    return Input, Weights, Bias, out, remainder

X_in = [[[int(X[i][j][0]*1e18)] for j in range(28)] for i in range(28)]
conv2d_1_weights = [[[[int(model.layers[1].weights[0][i][j][k][l]*1e18) for l in range(4)] for k in range(1)] for j in range(3)] for i in range(3)]
conv2d_1_bias = [int(model.layers[1].weights[1][i]*1e36) for i in range(4)]

X_in, conv2d_1_weights, conv2d_1_bias, conv2d_1_out, conv2d_1_remainder = Conv2DInt(28, 28, 1, 4, 3, 1, 10**18, X_in, conv2d_1_weights, conv2d_1_bias)
conv2d_1_out[0][0]

conv2d_model = Model(inputs, model.layers[1].output)
conv2d_model.predict(X.reshape(1,28,28,1))[0][0][0]

gamma = model.layers[2].weights[0].numpy()
beta = model.layers[2].weights[1].numpy()
moving_mean = model.layers[2].weights[2].numpy()
moving_var = model.layers[2].weights[3].numpy()
epsilon = model.layers[2].epsilon

a1 = gamma/(moving_var+epsilon)**.5
b1 = beta-gamma*moving_mean/(moving_var+epsilon)**.5
a1, b1

def BatchNormalizationInt(nRows, nCols, nChannels, n, X_in, a_in, b_in):
    X = [[[str(X_in[i][j][k] % p) for k in range(nChannels)] for j in range(nCols)] for i in range(nRows)]
    A = [str(a_in[k] % p) for k in range(nChannels)]
    B = [str(b_in[k] % p) for k in range(nChannels)]
    out = [[[None for _ in range(nChannels)] for _ in range(nCols)] for _ in range(nRows)]
    remainder = [[[None for _ in range(nChannels)] for _ in range(nCols)] for _ in range(nRows)]
    for i in range(nRows):
        for j in range(nCols):
            for k in range(nChannels):
                out[i][j][k] = (X_in[i][j][k]*a_in[k] + b_in[k])
                remainder[i][j][k] = str(out[i][j][k] % n)
                out[i][j][k] = str(out[i][j][k] // n % p)
    return X, A, B, out, remainder

bn_1_in = [[[int(conv2d_1_out[i][j][k]) if int(conv2d_1_out[i][j][k]) < p//2 else int(conv2d_1_out[i][j][k]) - p for k in range(4)] for j in range(26)] for i in range(26)]
bn_1_a = [int(a1[i]*1e18) for i in range(4)]
bn_1_b = [int(b1[i]*1e36) for i in range(4)]

_, bn_1_a, bn_1_b, bn_1_out, bn_1_remainder = BatchNormalizationInt(26, 26, 4, 10**18, bn_1_in, bn_1_a, bn_1_b)
bn_1_out[0][0]

bn_1_model = Model(inputs, model.layers[2].output)
bn_1_model.predict(X.reshape(1,28,28,1))[0][0][0]


relu_1_in = [[[int(bn_1_out[i][j][k]) for k in range(4)] for j in range(26)] for i in range(26)]
relu_1_out = [[[str(relu_1_in[i][j][k]) if relu_1_in[i][j][k] < p//2 else 0 for k in range(4)] for j in range(26)] for i in range(26)]

avg2d_1_in = [[[int(relu_1_out[i][j][k]) for k in range(4)] for j in range(26)] for i in range(26)]

def AveragePooling2DInt (nRows, nCols, nChannels, poolSize, strides, input):
    Input = [[[str(input[i][j][k] % p) for k in range(nChannels)] for j in range(nCols)] for i in range(nRows)]
    out = [[[0 for _ in range(nChannels)] for _ in range((nCols-poolSize)//strides + 1)] for _ in range((nRows-poolSize)//strides + 1)]
    remainder = [[[None for _ in range(nChannels)] for _ in range((nCols-poolSize)//strides + 1)] for _ in range((nRows-poolSize)//strides + 1)]
    for i in range((nRows-poolSize)//strides + 1):
        for j in range((nCols-poolSize)//strides + 1):
            for k in range(nChannels):
                for x in range(poolSize):
                    for y in range(poolSize):
                        out[i][j][k] += input[i*strides+x][j*strides+y][k]
                remainder[i][j][k] = str(out[i][j][k] % poolSize**2 % p)
                out[i][j][k] = str(out[i][j][k] // poolSize**2 % p)
    return Input, out, remainder

_, avg2d_1_out, avg2d_1_remainder = AveragePooling2DInt(26, 26, 4, 2, 2, avg2d_1_in)
avg2d_1_out[5][6]

avg2d_1_model = Model(inputs, model.layers[4].output)
avg2d_1_model.predict(X.reshape(1,28,28,1))[0][5][6]

conv2d_2_in = [[[int(avg2d_1_out[i][j][k]) for k in range(4)] for j in range(13)] for i in range(13)]
conv2d_2_weights = [[[[int(model.layers[5].weights[0][i][j][k][l]*1e18) for l in range(8)] for k in range(4)] for j in range(3)] for i in range(3)]
conv2d_2_bias = [int(model.layers[5].weights[1][i]*1e36) for i in range(8)]

_, conv2d_2_weights, conv2d_2_bias, conv2d_2_out, conv2d_2_remainder = Conv2DInt(13, 13, 4, 8, 3, 1, 10**18, conv2d_2_in, conv2d_2_weights, conv2d_2_bias)
conv2d_2_out[0][0]

conv2d_2_model = Model(inputs, model.layers[5].output)
conv2d_2_model.predict(X.reshape(1,28,28,1))[0][0][0]

gamma = model.layers[6].weights[0].numpy()
beta = model.layers[6].weights[1].numpy()
moving_mean = model.layers[6].weights[2].numpy()
moving_var = model.layers[6].weights[3].numpy()
epsilon = model.layers[6].epsilon

a2 = gamma/(moving_var+epsilon)**.5
b2 = beta-gamma*moving_mean/(moving_var+epsilon)**.5
a2, b2

bn_2_in = [[[int(conv2d_2_out[i][j][k]) if int(conv2d_2_out[i][j][k]) < p//2 else int(conv2d_2_out[i][j][k]) - p for k in range(8)] for j in range(11)] for i in range(11)]
bn_2_a = [int(a2[i]*1e18) for i in range(8)]
bn_2_b = [int(b2[i]*1e36) for i in range(8)]

_, bn_2_a, bn_2_b, bn_2_out, bn_2_remainder = BatchNormalizationInt(11, 11, 8, 10**18, bn_2_in, bn_2_a, bn_2_b)
bn_2_out[0][0]

bn_2_model = Model(inputs, model.layers[6].output)
bn_2_model.predict(X.reshape(1,28,28,1))[0][0][0]

relu_2_in = [[[int(bn_2_out[i][j][k]) for k in range(8)] for j in range(11)] for i in range(11)]
relu_2_out = [[[str(relu_2_in[i][j][k]) if relu_2_in[i][j][k] < p//2 else 0 for k in range(8)] for j in range(11)] for i in range(11)]
relu_2_out[0][0]

relu_2_model = Model(inputs, model.layers[7].output)
relu_2_model.predict(X.reshape(1,28,28,1))[0][0][0]

avg2d_2_in = [[[int(relu_2_out[i][j][k]) if int(relu_2_out[i][j][k]) < p//2 else int(relu_2_out[i][j][k]) - p for k in range(8)] for j in range(11)] for i in range(11)]

_, avg2d_2_out, avg2d_2_remainder = AveragePooling2DInt(11, 11, 8, 2, 2, avg2d_2_in)
avg2d_2_out[3][3]

avg2d_2_model = Model(inputs, model.layers[8].output)
avg2d_2_model.predict(X.reshape(1,28,28,1))[0][3][3]

flatten_out = [avg2d_2_out[i][j][k] for i in range(5) for j in range(5) for k in range(8)]
flatten_out[100:120]

flatten_model = Model(inputs, model.layers[9].output)
flatten_model.predict(X.reshape(1,28,28,1))[0][100:120]

dense_in = [int(flatten_out[i]) if int(flatten_out[i]) < p//2 else int(flatten_out[i]) - p for i in range(200)]
dense_weights = [[int(model.layers[10].weights[0][i][j]*1e18) for j in range(10)] for i in range(200)]
dense_bias = [int(model.layers[10].weights[1][i]*1e36) for i in range(10)]

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

_, dense_weights, dense_bias, dense_out, dense_remainder = DenseInt(200, 10, 10**18, dense_in, dense_weights, dense_bias)
dense_out
print("dense out:", dense_out)
dense_model = Model(inputs, model.layers[-2].output)
prediction=dense_model.predict(X.reshape(1,28,28,1))[0]
print ("prediction orig:", prediction.argmax())


# Convert to integers and compute modulo p
# mod_values = [(p-int(x)) for x in dense_out]
# print("circiut output final:", np.array(mod_values).argmax())
dense_out = [(int(x)+1e18)%p for x in dense_out]
print("dens_out:", dense_out)
print("circit prediction:",np.array(dense_out).argmax())