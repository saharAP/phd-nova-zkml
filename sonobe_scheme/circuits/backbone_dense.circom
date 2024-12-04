pragma circom 2.0.0;
include "./node_modules/circomlib-ml/circuits/ReLU.circom";
include "./node_modules/circomlib-ml/circuits/circomlib/mimc.circom";
include "./node_modules/circomlib-ml/circuits/Dense.circom";
include "./node_modules/circomlib-ml/circuits/ArgMax.circom";
include "./utils/mimcsponge.circom";
include "./utils/utils.circom";

template BackboneDense(nInputs, nOutputs,n) {
    // activation outputed by the previous layer
    signal input a_prev[nInputs]; // IVC input

    signal input relu_out[nOutputs];

    signal input dense_1_weights[nInputs][nOutputs];
    signal input dense_1_bias[nOutputs];
    signal input dense_1_out[nOutputs];
    signal input dense_1_remainder[nOutputs];


    signal output out[nOutputs]; // IVC output

    component dense_1 = Dense(nInputs,nOutputs,10**n);
    component relu[nOutputs];
  

    for (var i = 0; i < nInputs; i++) {
        dense_1.in[i] <== in[i];
        for (var j=0; j<nOutputs; j++) {
            dense_1.weights[i][j] <== dense_1_weights[i][j];
        }
    }
       for (var i=0; i<nOutputs; i++) {
        dense_1.bias[i] <== dense_1_bias[i];
    }

    for (var i=0; i<nOutputs; i++) {
        dense_1.out[i] <== dense_1_out[i];
        dense_1.remainder[i] <== dense_1_remainder[i];
    }
    // apply ReLU to the output of the first dense layer
    for (var i=0; i<nOutputs; i++) {
        relu[i] = ReLU();
        relu[i].in <== dense_1.out[i];
        relu[i].out <== relu_out[i];
    }
    for (var i=0; i<nOutputs; i++) {
        relu[i] = ReLU();
        relu[i].in <== dense_1.out[i];
        relu[i].out <== relu_out[i];
    }
    for (var i=0; i<nOutputs; i++) {
        relu[i].out <== relu_out[i];
        out[i] <== relu[i].out;
    }
}
