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
mod network; // Import the `network` module
use network::Network;

use serde_json::{json, Value};
use serde::{Deserialize, Serialize, Deserializer};
use std::{
    collections::HashMap, env::current_dir, fs, fs::File, io::BufReader, path::PathBuf,
    time::Instant,
};
use std::str::FromStr;


use ark_bn254::{constraints::GVar, Bn254, Fr, G1Projective as G1};

use ark_groth16::Groth16;
use ark_grumpkin::{constraints::GVar as GVar2, Projective as G2};

// use std::path::PathBuf;
// use std::time::Instant;

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
    Decider, FoldingScheme,
};
use frontends::circom::CircomFCircuit;
use solidity_verifiers::{
    evm::{compile_solidity, Evm},
    utils::get_function_selector_for_nova_cyclefold_verifier,
    verifiers::nova_cyclefold::get_decider_template_for_cyclefold_decider,
    NovaCycleFoldVerifierKey,
};
// // Custom deserialization function for Vec<Fr>
// fn deserialize_vec_fr<'de, D>(deserializer: D) -> Result<Vec<Fr>, D::Error>
// where
//     D: Deserializer<'de>,
// {
//     let str_vec: Vec<String> = Deserialize::deserialize(deserializer)?;
//     str_vec
//         .into_iter()
//         .map(|s| {
//             Fr::from_str(&s).map_err(|_| {
//                 serde::de::Error::custom(format!("Failed to parse Fr from string: {}", s))
//             })
//         })
//         .collect()
// }

// // network model and struct
// #[derive(Debug)]
// struct DenseLayer {
//     // dims: [nInputs x nOutput]
//     weight: Vec<Vec<Fr>>,
//     // dims: [nOutputs]
//     bias: Vec<Fr>,
//     // dims: [nOutputs]
//     dense_out: Vec<Fr>,
//     // dims: [nOutputs]
//     remainder: Vec<Fr>,
//      // dims: [nOutputs]
//     activation: Vec<Fr>,
// }
// #[derive(Debug)]
// struct TailLayer {
//     // dims: [nInputs x nOutput]
   
//     weight: Vec<Vec<Fr>>,
//     // dims: [nOutputs]
//     bias: Vec<Fr>,
//     // dims: [nOutputs]
//     dense_out: Vec<Fr>,
//     // dims: [nOutputs]
//     remainder: Vec<Fr>,
//      // dims: [nOutputs]
//     activation: Fr,
// }
// #[derive(Debug)]
// struct Network{
//     // dims: [nRows x nCols]
    
//     x: Vec<Fr>,
//     head: DenseLayer,
//     backbone: Vec<DenseLayer>,
//     tail: TailLayer,
// }
// struct for the inputs of the recursion for all the steps
// #[derive(Debug)]
// struct RecursionInputs {
//     all_private: Vec<Vec<i64>>,
//     start_pub_primary: Vec<F1>,
//     start_pub_secondary: Vec<F2>,
// }
// fn read_circiut_input(path: &str) -> Vec<Vec<Fr>> {
//     let input = std::fs
//         .read_to_string(path)
//         .expect("Unable to read file");
//     let input: Vec<Vec<Fr>> = serde_json::from_str(&input).unwrap();
// }
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
    let file_content = fs::read_to_string(f).expect("Unable to read file");

       // Deserialize the JSON content into the Network struct
       serde_json::from_str(&file_content).expect("JSON was not well-formatted")

    // let f = File::open(f).unwrap();
    // let rdr = BufReader::new(f);
    // println!("- Working");
    // serde_json::from_reader(rdr).unwrap()
}
/*
* Constructs the inputs necessary for recursion. This includes 1) private
* inputs for every step, and 2) initial public inputs for the first step of the
* primary & secondary circuits.
*/
// fn construct_inputs(
//     network: &Network,
//     num_steps: usize,
//     mimc3d_r1cs: &R1CS<F1>,
//     mimc3d_wasm: PathBuf,
// ) -> RecursionInputs {
//     let mut private_inputs = Vec::new();
//     for i in 0..num_steps {
//         let a = if i > 0 {
//             &network.backbone[i - 1].out
//         } else {
//             &network.head.out
//         };
//         let mut priv_in= Vec::new();
//         priv_in.extend(json!(a));
//         priv_in.extend(json!(network.backbone[i].weights));
//         priv_in.extend(json!(network.backbone[i].bias));
//         // let priv_in = HashMap::from([
//         //     (String::from("a_prev"), json!(a)),
//         //     (String::from("W"), json!(network.backbone[i].weights)),
//         //     (String::from("b"), json!(network.backbone[i].bias)),
//         // ]);
//         private_inputs.push(priv_in);
//     }

//     let v_1 = mimc3d(
//         mimc3d_r1cs,
//         mimc3d_wasm,
//         rm_padding(&fwd_pass.head.a, fwd_pass.padding),
//     )
//     .to_str_radix(10);
//     let z0_primary = vec![
//         F1::from(0),
//         F1::from_raw(U256::from_dec_str(&v_1).unwrap().0),
//     ];

//     // Secondary circuit is TrivialTestCircuit, filler val
//     let z0_secondary = vec![F2::zero()];

//     println!("- Done");
//     RecursionInputs {
//         all_private: private_inputs,
//         start_pub_primary: z0_primary,
//         start_pub_secondary: z0_secondary,
//     }
// }
fn main() {
    // define number of steps to be done in the IVC
    let n_steps = 2;
    // import data from a sample digit image 7 as input
    const MNIST_INPUT: &str = "../data/circiut_inputs/mnist_input_4.json";

    // read the input data for backbone circiut from the file
    let network = read_circiut_input(MNIST_INPUT);
     // Print the activation of the head
     println!("Head activation: {:?}", network.head.activation);




//     // set the initial state with size of 10
//     let z_0= vec![
//         Fr::from(1_u32),
//         Fr::from(2_u32),
//         Fr::from(3_u32),
//         Fr::from(4_u32),
//         Fr::from(5_u32),
//         Fr::from(6_u32),
//         Fr::from(7_u32),
//         Fr::from(8_u32),
//         Fr::from(9_u32),
//         Fr::from(10_u32),
//     ];

//     // set the external inputs to be used at each step of the IVC, it has length of 10 since this
//     // is the number of steps that we will do
//     let external_inputs = vec![
//         vec![Fr::from(6u32), Fr::from(7u32)],
//         vec![Fr::from(8u32), Fr::from(9u32)],
//         vec![Fr::from(10u32), Fr::from(11u32)],
//         vec![Fr::from(12u32), Fr::from(13u32)],
//         vec![Fr::from(14u32), Fr::from(15u32)],
//         vec![Fr::from(6u32), Fr::from(7u32)],
//         vec![Fr::from(8u32), Fr::from(9u32)],
//         vec![Fr::from(10u32), Fr::from(11u32)],
//         vec![Fr::from(12u32), Fr::from(13u32)],
//         vec![Fr::from(14u32), Fr::from(15u32)],
//     ];
//     // initialize the Circom circuit
//     let r1cs_path = PathBuf::from("./circuits/out/backbone_layer_dnn.r1cs");
//     let wasm_path = PathBuf::from(
//         "./circuits/out/backbone_layer_dnn_js/backbone_layer_dnn.wasm",
//     );
//    // 10= size of the input, external input size :140= 10 * 10 + 4 * 10
//     let f_circuit_params = (r1cs_path.into(), wasm_path.into(), 10, 140);
//     let f_circuit = CircomFCircuit::<Fr>::new(f_circuit_params).unwrap();

//     pub type N =
//         Nova<G1, GVar, G2, GVar2, CircomFCircuit<Fr>, KZG<'static, Bn254>, Pedersen<G2>, false>;
//     pub type D = DeciderEth<
//         G1,
//         GVar,
//         G2,
//         GVar2,
//         CircomFCircuit<Fr>,
//         KZG<'static, Bn254>,
//         Pedersen<G2>,
//         Groth16<Bn254>,
//         N,
//     >;

//     let poseidon_config = poseidon_canonical_config::<Fr>();
//     let mut rng = rand::rngs::OsRng;

//     // prepare the Nova prover & verifier params
//     let nova_preprocess_params = PreprocessorParam::new(poseidon_config, f_circuit.clone());
//     let nova_params = N::preprocess(&mut rng, &nova_preprocess_params).unwrap();

//     // initialize the folding scheme engine, in our case we use Nova
//     let mut nova = N::init(&nova_params, f_circuit.clone(), z_0).unwrap();

//     // prepare the Decider prover & verifier params
//     let (decider_pp, decider_vp) =
//         D::preprocess(&mut rng, nova_params.clone(), nova.clone()).unwrap();

//     // run n steps of the folding iteration
//     for (i, external_inputs_at_step) in external_inputs.iter().enumerate() {
//         let start = Instant::now();
//         nova.prove_step(rng, external_inputs_at_step.clone(), None)
//             .unwrap();
//         println!("Nova::prove_step {}: {:?}", i, start.elapsed());
//     }

//     // verify the last IVC proof
//     let ivc_proof = nova.ivc_proof();
//     N::verify(
//         nova_params.1, // Nova's verifier params
//         ivc_proof,
//     )
//     .unwrap();

//     let start = Instant::now();
//     let proof = D::prove(rng, decider_pp, nova.clone()).unwrap();
//     println!("generated Decider proof: {:?}", start.elapsed());

//     let verified = D::verify(
//         decider_vp.clone(),
//         nova.i,
//         nova.z_0.clone(),
//         nova.z_i.clone(),
//         &nova.U_i.get_commitments(),
//         &nova.u_i.get_commitments(),
//         &proof,
//     )
//     .unwrap();
//     assert!(verified);
//     println!("Decider proof verification: {}", verified);

//     // Now, let's generate the Solidity code that verifies this Decider final proof
//     let function_selector =
//         get_function_selector_for_nova_cyclefold_verifier(nova.z_0.len() * 2 + 1);

//     let calldata: Vec<u8> = prepare_calldata(
//         function_selector,
//         nova.i,
//         nova.z_0,
//         nova.z_i,
//         &nova.U_i,
//         &nova.u_i,
//         proof,
//     )
//     .unwrap();

//     // prepare the setup params for the solidity verifier
//     let nova_cyclefold_vk = NovaCycleFoldVerifierKey::from((decider_vp, f_circuit.state_len()));

//     // generate the solidity code
//     let decider_solidity_code = get_decider_template_for_cyclefold_decider(nova_cyclefold_vk);

//     // verify the proof against the solidity code in the EVM
//     let nova_cyclefold_verifier_bytecode = compile_solidity(&decider_solidity_code, "NovaDecider");
//     let mut evm = Evm::default();
//     let verifier_address = evm.create(nova_cyclefold_verifier_bytecode);
//     let (_, output) = evm.call(verifier_address, calldata.clone());
//     assert_eq!(*output.last().unwrap(), 1);

//     // save smart contract and the calldata
//     println!("storing nova-verifier.sol and the calldata into files");
//     use std::fs;
//     fs::write(
//         "./src/solidity/nova-verifier.sol",
//         decider_solidity_code.clone(),
//     )
//     .unwrap();
//     fs::write("./src/solidity/solidity-calldata.calldata", calldata.clone()).unwrap();
//     let s = solidity_verifiers::utils::get_formatted_calldata(calldata.clone());
//     fs::write("./src/solidity/solidity-calldata.inputs", s.join(",\n")).expect("");

    
}
