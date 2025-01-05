# utilities for runing inference and generating input for the circiut
import numpy as np

p = 21888242871839275222246405745257275088548364400416034343698204186575808495617

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

