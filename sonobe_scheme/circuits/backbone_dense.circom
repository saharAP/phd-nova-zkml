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

    signal input external_inputs[nInputs * nOutputs+ nOutputs+ nOutputs+ nOutputs+ nOutputs];

    signal output out[nOutputs]; // IVC output
// external inputs
    
    // signal input dense_1_weights[nInputs][nOutputs];
    // signal input dense_1_bias[nOutputs];
    // signal input dense_1_out[nOutputs];
    // signal input dense_1_remainder[nOutputs];
    // signal input relu_out[nOutputs];
    // convert external inputs to the circit inputs

    // decoding external inputs to the circuit inputs
    signal dense_1_weights[nInputs][nOutputs];
    signal dense_1_bias[nOutputs];
    signal dense_1_out[nOutputs];
    signal dense_1_remainder[nOutputs];
    signal relu_out[nOutputs];
    for (var i = 0; i < nInputs; i++) {
        for (var j=0; j<nOutputs; j++) {
            dense_1_weights[i][j] <-- external_inputs[i*nOutputs+j];
        }
    }
    
    for (var i=0; i<nOutputs; i++) {
        dense_1_bias[i] <-- external_inputs[nInputs*nOutputs+i];
    }
    
    for (var i=0; i<nOutputs; i++) {
        dense_1_out[i] <-- external_inputs[nInputs*nOutputs+nOutputs+i];
    }
    
    for (var i=0; i<nOutputs; i++) {
        dense_1_remainder[i] <-- external_inputs[nInputs*nOutputs+nOutputs+nOutputs+i];
    }
    
    for (var i=0; i<nOutputs; i++) {
        relu_out[i] <-- external_inputs[nInputs*nOutputs+nOutputs+nOutputs+nOutputs+i];
    }
    
// Circuit logic starts here
    component dense_1 = Dense(nInputs,nOutputs,10**n);
    component relu[nOutputs];
  

    for (var i = 0; i < nInputs; i++) {
        dense_1.in[i] <== a_prev[i];
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
    }
    for (var i=0; i<nOutputs; i++) {
        relu[i].out <== relu_out[i];
        out[i] <== relu[i].out;
    }
}
