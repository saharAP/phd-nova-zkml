
pragma circom 2.1.1;
include "./Backbone_dense.circom";

component main { public [step_in] } = BackboneDense(10,10,18);