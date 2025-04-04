pragma circom 2.0.0;
include "./node_modules/circomlib-ml/circuits/ReLU.circom";
include "./node_modules/circomlib-ml/circuits/Dense.circom";
include "./node_modules/circomlib/circuits/poseidon.circom";

template BackboneDense(nInputs, nOutputs,n) {
    // Poseidon hash of the activation outputed by the previous layer
    signal input ivc_input[1]; // IVC input

    signal input external_inputs[nInputs+nInputs * nOutputs+ nOutputs+ nOutputs+ nOutputs+ nOutputs];

    // Poseidon hash of the activation outputed by the current layer
    signal output ivc_output[1]; // IVC output

    // decoding external inputs to the circuit inputs
    signal a_prev[nInputs]; // activation outputed by the previous layer
    signal dense_weights[nInputs][nOutputs];
    signal dense_bias[nOutputs];
    signal dense_out[nOutputs];
    signal dense_remainder[nOutputs];
    signal relu_out[nOutputs];

    for (var i = 0; i < nInputs; i++) {
        a_prev[i] <-- external_inputs[i];
    }
    for (var i = 0; i < nInputs; i++) {
        for (var j=0; j<nOutputs; j++) {
            dense_weights[i][j] <-- external_inputs[nInputs+i*nOutputs+j];
        }
    }
    
    for (var i=0; i<nOutputs; i++) {
        dense_bias[i] <-- external_inputs[nInputs+nInputs*nOutputs+i];
    }
    
    for (var i=0; i<nOutputs; i++) {
        dense_out[i] <-- external_inputs[nInputs+nInputs*nOutputs+nOutputs+i];
    }
    
    for (var i=0; i<nOutputs; i++) {
        dense_remainder[i] <-- external_inputs[nInputs+nInputs*nOutputs+nOutputs+nOutputs+i];
    }
    
    for (var i=0; i<nOutputs; i++) {
        relu_out[i] <-- external_inputs[nInputs+nInputs*nOutputs+nOutputs+nOutputs+nOutputs+i];
    }
    
// Circuit logic starts here
    component dense = Dense(nInputs,nOutputs,10**n);
    component relu[nOutputs];
    component inputHasher = Poseidon(nInputs);
    // check if the ivc_input is the same as the hash of the activation outputed by the previous layer
    for (var i=0; i<nInputs; i++) {
        inputHasher.inputs[i] <== a_prev[i];
    }
    inputHasher.out === ivc_input[0];

  
    dense.in <== a_prev;
    dense.weights <== dense_weights;
    dense.bias <== dense_bias;
    dense.out <== dense_out;
    dense.remainder <== dense_remainder;

    // apply ReLU to the output of the dense layer
    for (var i=0; i<nOutputs; i++) {
        relu[i] = ReLU();
        relu[i].in <== dense.out[i];
        relu[i].out <== relu_out[i];
    }
    // compute hash of the current activation as ivc_output
    component outputHasher = Poseidon(nInputs);
    for (var i=0; i<nOutputs; i++) {
        outputHasher.inputs[i] <== relu_out[i];
    }
    ivc_output[0]<== outputHasher.out;
}
component main { public [ivc_input] } = BackboneDense(10,10,18);
