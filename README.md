Running inside the `circiuts` folder:

```sh
circom backbone_layer_dnn.circom --r1cs --wasm --sym --prime bn128 -o out
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
name = "mnist_str"
path = "./src/mnist_str.rs"
```
run the specific rs file from `sonobe_scheme` folder as follows:
```sh
cargo run --release --example mnist_str
```

Get memory usage:
```sh
> cargo build --release --example mnist_str 
> /usr/bin/time -l ./target/release/examples/mnist_str
```
## References
- https://github.com/privacy-scaling-explorations/sonobe


