pragma circom 2.1.1;

include "mimcsponge.circom";

// MiMC hash a 3D matrix, with 91 iterations for MiMC
// Elements are unrolled into a single flattened vector
template MimcHashMatrix3D(rows, cols, depth) {
    signal input matrix[rows][cols][depth];
    signal output hash;

    component mimc = MiMCSponge(rows * cols * depth, 91, 1);
    mimc.k <== 0;

    for (var row = 0; row < rows; row++) {
        for (var col = 0; col < cols; col++) {
            for (var dep = 0; dep < depth; dep++) {
                var indexFlattenedVector = (row * cols * depth) + (col * depth) + dep;
                mimc.ins[indexFlattenedVector] <== matrix[row][col][dep];
            }
        }
    }

    hash <== mimc.outs[0];
}

// MiMC hash a 4D matrix, with 91 iterations for MiMC
// Elements are unrolled into a single flattened vector
template MimcHashMatrix4D(rows, cols, depth, dim4length) {
    signal input matrix[rows][cols][depth][dim4length];
    signal output hash;

    component mimc = MiMCSponge(rows * cols * depth * dim4length, 91, 1);
    mimc.k <== 0;

    for (var row = 0; row < rows; row++) {
        for (var col = 0; col < cols; col++) {
            for (var dep = 0; dep < depth; dep++) {
                for (var d4 = 0; d4 < dim4length; d4++) {
                    var indexFlattenedVector = (row * cols * depth * dim4length) + (col * depth * dim4length) + (dep * dim4length) + d4;
                    mimc.ins[indexFlattenedVector] <== matrix[row][col][dep][d4];
                }
            }
        }
    }

    hash <== mimc.outs[0];
}