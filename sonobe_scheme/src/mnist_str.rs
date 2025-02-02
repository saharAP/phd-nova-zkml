#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(clippy::upper_case_acronyms)]
///
/// This example performs the full flow:
/// - define the circuit to be folded
/// - fold the circuit with Nova+CycleFold's IVC
/// - generate a DeciderEthCircuit final proof
/// - generate the Solidity contract that verifies the proof
/// - verify the proof in the EVM
///
use std::any::type_name;

mod network; // Import the `network` module
use network::Network;


#[macro_use]
mod memory_utils;

// use jemalloc_ctl::{epoch, stats};

use serde_json::{json, Value};
use serde::{Deserialize, Serialize, Deserializer};
use std::{
    collections::HashMap, env::current_dir, fs, fs::File, io::BufReader, path::PathBuf,
    time::Instant,
};
use std::str::FromStr;

use ark_bn254::{Bn254, Fr, G1Projective as G1};
use ark_groth16::Groth16;
use ark_grumpkin::Projective as G2;


use experimental_frontends::{circom::CircomFCircuit, utils::VecF};
use folding_schemes::{
    commitment::{kzg::KZG, pedersen::Pedersen},
    folding::{
        nova::{
            decider_eth::{prepare_calldata, Decider as DeciderEth},
            Nova, PreprocessorParam,
        },
        traits::CommittedInstanceOps,
    },
    frontend::FCircuit,
    transcript::poseidon::poseidon_canonical_config,
    Decider, Error, FoldingScheme,
};
use solidity_verifiers::{
    evm::{compile_solidity, Evm},
    utils::get_function_selector_for_nova_cyclefold_verifier,
    verifiers::nova_cyclefold::get_decider_template_for_cyclefold_decider,
    NovaCycleFoldVerifierKey,
};

/*
 * Load in the forward pass (i.e. parameters and inputs/outputs for each layer).
 */
 fn read_circiut_input2(f: &str) -> Network {
    let f = File::open(f).unwrap();
    let rdr = BufReader::new(f);
    println!("- Working");
    serde_json::from_reader(rdr).unwrap()
}
fn read_circiut_input(f: &str) -> Network {
     // Read the JSON file
         // Read the file content as a string
    let file_content = fs::read_to_string(f).unwrap();

       // Deserialize the JSON content into the Network struct
       serde_json::from_str(&file_content).unwrap()
}

// #[global_allocator]
// static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;
fn main() -> Result<(), Error> {

    // import data from a sample digit image 7 as input
    const MNIST_INPUT: &str = "../data/circiut_inputs/mnist_input_20_10_dig4.json";

    // read the input data for backbone circiut from the file
    let network = read_circiut_input(MNIST_INPUT);
     // set the initial state with the activation function of the head
    let z_0= network.head.activation.clone();

    // define number of steps to be done in the IVC
    let n_steps =  network.backbone.len();
    // all the external inputs for all backbone layers
    let mut external_inputs: Vec<Vec<Fr>> = Vec::new();
    for i in 0..n_steps {
        let mut flat_vec: Vec<Fr> = network.backbone[i].weight.clone().into_iter().flat_map(|v| v.into_iter()).collect();
        flat_vec.extend( network.backbone[i].bias.clone()); 
        flat_vec.extend( network.backbone[i].dense_out.clone());
        flat_vec.extend( network.backbone[i].remainder.clone()); 
        flat_vec.extend( network.backbone[i].activation.clone());
        external_inputs.push(flat_vec);
    }

    println!("Number of steps: {:?}", external_inputs.len());
    // initialize the Circom circuit
    let r1cs_path = PathBuf::from("./circuits/out/backbone_layer_dnn.r1cs");
    let wasm_path = PathBuf::from(
        "./circuits/out/backbone_layer_dnn_js/backbone_layer_dnn.wasm",
    );
   // 10= size of the input, external input size :140= 10 * 10 + 4 * 10
    let f_circuit_params = (r1cs_path.into(), wasm_path.into(), 10); // state len = 10
    const EXT_INP_LEN: usize = 140; // external inputs len = 140
    let f_circuit = CircomFCircuit::<Fr, EXT_INP_LEN>::new(f_circuit_params)?;

    pub type N =
        Nova<G1, G2, CircomFCircuit<Fr, EXT_INP_LEN>, KZG<'static, Bn254>, Pedersen<G2>, false>;
    pub type D = DeciderEth<
        G1,
        G2,
        CircomFCircuit<Fr, EXT_INP_LEN>,
        KZG<'static, Bn254>,
        Pedersen<G2>,
        Groth16<Bn254>,
        N,
    >;

    let poseidon_config = poseidon_canonical_config::<Fr>();
    let mut rng = ark_std::rand::rngs::OsRng;
    // let mut rng = rand::rngs::OsRng;
    // prepare the Nova prover & verifier params
    let nova_preprocess_params = PreprocessorParam::new(poseidon_config, f_circuit.clone());
    let nova_params = N::preprocess(&mut rng, &nova_preprocess_params)?;

    // prepare the Decider prover & verifier params
    let (decider_pp, decider_vp) =
    D::preprocess(&mut rng, (nova_params.clone(), f_circuit.state_len()))?;

    // initialize the folding scheme engine, in our case we use Nova
    let mut nova = N::init(&nova_params, f_circuit.clone(), z_0)?;

    // run n steps of the folding iteration
    for (i, external_inputs_at_step) in external_inputs.iter().enumerate() {
            let start = Instant::now();
            nova.prove_step(rng, VecF(external_inputs_at_step.clone()), None)?;
            println!("Nova::prove_step {}: {:?}", i, start.elapsed());
    }

    // verify the last IVC proof
    let ivc_proof = nova.ivc_proof();
    N::verify(
        nova_params.1, // Nova's verifier params
        ivc_proof,
    )?;
 
//     let peak_beforefree = stats::resident::read().unwrap();
//     println!("🚀 Peak Memory Usage before free: {:.3} MB",
//     (peak_beforefree) as f64 / (1024.0 * 1024.0));
//    // ============================
//     // 🛑 Free Memory Before Proof Generation
//     // ============================
    // epoch::mib().unwrap().advance().unwrap();

    // let peak_before = stats::resident::read().unwrap();
    // println!("🚀 Peak Memory Usage before: {:.3} MB",
    // (peak_before) as f64 / (1024.0 * 1024.0));
//    let mut proof = new D::Proof();
    let mut start = Instant::now();
    let result = measure_memory!(|| {
    let proof = D::prove(rng, decider_pp, nova.clone()).unwrap();
    proof
    });
    let proof = result();
    println!("✅ generated Decider proof: {:?}", start.elapsed());
  
     // ============================
    // 🛑 Refresh Stats & Get Peak Memory Usage
    // ============================
    // epoch::mib().unwrap().advance().unwrap();

    // let peak_after = stats::resident::read().unwrap();
    // println!("🚀 Peak Memory Usage after: {:.3} MB",
    // (peak_after) as f64 / (1024.0 * 1024.0));

    // println!("🚀 Peak Memory Usage for Decider Proof Generation: {:.3} MB",
    //     (peak_after - peak_before) as f64 / (1024.0 * 1024.0));
    //verify the Decider proof
    start = Instant::now();
    let verified = D::verify(
        decider_vp.clone(),
        nova.i,
        nova.z_0.clone(),
        nova.z_i.clone(),
        &nova.U_i.get_commitments(),
        &nova.u_i.get_commitments(),
        &proof,
    )?;
    assert!(verified);
    println!("✅ Decider proof verification: {}", verified);
    println!("✅ Decider proof verification time: {:?}", start.elapsed());

    // Now, let's generate the Solidity code that verifies this Decider final proof
    let function_selector =
        get_function_selector_for_nova_cyclefold_verifier(nova.z_0.len() * 2 + 1);

    let calldata: Vec<u8> = prepare_calldata(
        function_selector,
        nova.i,
        nova.z_0,
        nova.z_i,
        &nova.U_i,
        &nova.u_i,
        proof,
    )?;

    // prepare the setup params for the solidity verifier
    let nova_cyclefold_vk = NovaCycleFoldVerifierKey::from((decider_vp, f_circuit.state_len()));

    // generate the solidity code
    let decider_solidity_code = get_decider_template_for_cyclefold_decider(nova_cyclefold_vk);

    // verify the proof against the solidity code in the EVM
    let nova_cyclefold_verifier_bytecode = compile_solidity(&decider_solidity_code, "NovaDecider");
    let mut evm = Evm::default();
    let verifier_address = evm.create(nova_cyclefold_verifier_bytecode);
    let (_, output) = evm.call(verifier_address, calldata.clone());
    assert_eq!(*output.last().unwrap(), 1);

    // save smart contract and the calldata
    println!("storing nova-verifier.sol and the calldata into files");
    use std::fs;
    fs::write(
        "./src/solidity/nova-verifier.sol",
        decider_solidity_code.clone(),
    )?;
    fs::write("./src/solidity/solidity-calldata.calldata", calldata.clone())?;
    let s = solidity_verifiers::utils::get_formatted_calldata(calldata.clone());
    fs::write("./src/solidity/solidity-calldata.inputs", s.join(",\n")).expect("");

    Ok(())    
}
