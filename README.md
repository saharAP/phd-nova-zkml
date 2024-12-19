Running inside the `circiuts` folder:

```sh
circom input_layer.circom --r1cs --wasm --sym --prime bn128 -o out
```

Run the project inside the `sonobe_scheme` folder:

```sh
cargo run --release
```

circomlib-ml
https://github.com/socathie/circomlib-ml

add rs files to Cargo.toml like the following:
```conf
[[example]]
name = "mnist_test"
path = "./src/mnist_test.rs"
```
run the specific rs file from `sonobe_scheme` folder as follows:
```sh
cargo run --release --example mnist_test
```
