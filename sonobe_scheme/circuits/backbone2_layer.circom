
pragma circom 2.1.1;
include "./Backbone_dense.circom";

component main { public [a_prev] } = BackboneDense(10,10,18);