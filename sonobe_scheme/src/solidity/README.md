## Foundry

**Foundry is a blazing fast, portable and modular toolkit for Ethereum application development written in Rust.**

Foundry consists of:

-   **Forge**: Ethereum testing framework (like Truffle, Hardhat and DappTools).
-   **Cast**: Swiss army knife for interacting with EVM smart contracts, sending transactions and getting chain data.
-   **Anvil**: Local Ethereum node, akin to Ganache, Hardhat Network.
-   **Chisel**: Fast, utilitarian, and verbose solidity REPL.

## Documentation

https://book.getfoundry.sh/

## Usage

### Build

```shell
$ forge build
```

### Test

```shell
$ forge test
```

### Format

```shell
$ forge fmt
```

### Gas Snapshots

```shell
$ forge snapshot
```

### Anvil

```shell
$ anvil
```

### Deploy

```shell
$ forge script script/DeployOnchainVerifier.s.sol:DeployOnchainVerifier --rpc-url $ETH_RPC_URL --broadcast --private-key $PRIVATE_KEY  --verify --resume --etherscan-api-key $ETHERSCAN_API_KEY

$ forge script script/CallVerify.s.sol:CallVerify --rpc-url $ETH_RPC_URL --broadcast --private-key $PRIVATE_KEY
```
### parse input 
```shell
$ python3 sonobe_scheme/src/py_script_parsing/parse_input.py sonobe_scheme/src/solidity/inputData/solidity-calldata-dnn-2-10.inputs 113
```

