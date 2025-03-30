// SPDX-License-Identifier: MIT
pragma solidity 0.8.25;

import "forge-std/Script.sol";
import {OnchainVerifier} from "../src/OnchainVerifier.sol";
import {NovaDecider} from "../src/nova-verifier-dnn-2-10.sol";

contract DeployOnchainVerifier is Script {
    function run() external returns(address verifierAddress) {
        // Start the transaction broadcast
        vm.startBroadcast();

        // Deploy the NovaVerifier contract
        NovaDecider novaDecider = new NovaDecider();
        OnchainVerifier verifier = new OnchainVerifier(address(novaDecider));

        // Stop the broadcast
        vm.stopBroadcast();
        verifierAddress = address(verifier);
    }
}
