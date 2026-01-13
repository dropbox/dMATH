# ⚠️ Project is in Maintenance Mode ⚠️

This project is no longer internally developed and maintained. However, we are happy to review and
accept small, well-written pull requests by the community. We will only consider bug fixes and minor
enhancements.

Any new or currently open issues and discussions shall be answered and supported by the community.

# Manticore



[[Build Status]][1] [[Coverage Status]][2] [[PyPI Version]][3] [[Slack Status]][4] [[Documentation
Status]][5] [[Example Status]][6] [[LGTM Total Alerts]][7]

Manticore is a symbolic execution tool for the analysis of smart contracts and binaries.

## Features

* **Program Exploration**: Manticore can execute a program with symbolic inputs and explore all the
  possible states it can reach
* **Input Generation**: Manticore can automatically produce concrete inputs that result in a given
  program state
* **Error Discovery**: Manticore can detect crashes and other failure cases in binaries and smart
  contracts
* **Instrumentation**: Manticore provides fine-grained control of state exploration via event
  callbacks and instruction hooks
* **Programmatic Interface**: Manticore exposes programmatic access to its analysis engine via a
  Python API

Manticore can analyze the following types of programs:

* Ethereum smart contracts (EVM bytecode)
* Linux ELF binaries (x86, x86_64, aarch64, and ARMv7)
* WASM Modules

## Installation

> Note: We recommend installing Manticore in a [virtual environment][8] to prevent conflicts with
> other projects or packages

Option 1: Installing from PyPI:

pip install manticore

Option 2: Installing from PyPI, with extra dependencies needed to execute native binaries:

pip install "manticore[native]"

Option 3: Installing a nightly development build:

pip install --pre "manticore[native]"

Option 4: Installing from the `master` branch:

git clone https://github.com/trailofbits/manticore.git
cd manticore
pip install -e ".[native]"

Option 5: Install via Docker:

docker pull trailofbits/manticore

Once installed, the `manticore` CLI tool and Python API will be available.

For a development installation, see our [wiki][9].

## Usage

### CLI

Manticore has a command line interface which can perform a basic symbolic analysis of a binary or
smart contract. Analysis results will be placed into a workspace directory beginning with `mcore_`.
For information about the workspace, see the [wiki][10].

#### EVM

Manticore CLI automatically detects you are trying to test a contract if (for ex.) the contract has
a `.sol` or a `.vy` extension. See a [demo][11].

Click to expand:
$ manticore examples/evm/umd_example.sol 
 [9921] m.main:INFO: Registered plugins: DetectUninitializedMemory, DetectReentrancySimple, DetectEx
ternalCallAndLeak, ...
 [9921] m.e.manticore:INFO: Starting symbolic create contract
 [9921] m.e.manticore:INFO: Starting symbolic transaction: 0
 [9921] m.e.manticore:INFO: 4 alive states, 6 terminated states
 [9921] m.e.manticore:INFO: Starting symbolic transaction: 1
 [9921] m.e.manticore:INFO: 16 alive states, 22 terminated states
[13761] m.c.manticore:INFO: Generated testcase No. 0 - STOP(3 txs)
[13754] m.c.manticore:INFO: Generated testcase No. 1 - STOP(3 txs)
...
[13743] m.c.manticore:INFO: Generated testcase No. 36 - THROW(3 txs)
[13740] m.c.manticore:INFO: Generated testcase No. 37 - THROW(3 txs)
[9921] m.c.manticore:INFO: Results in ~/manticore/mcore_gsncmlgx
Manticore-verifier

An alternative CLI tool is provided that simplifies contract testing and allows writing properties
methods in the same high-level language the contract uses. Checkout manticore-verifier
[documentation][12]. See a [demo][13]

#### Native

Click to expand:
$ manticore examples/linux/basic
[9507] m.n.manticore:INFO: Loading program examples/linux/basic
[9507] m.c.manticore:INFO: Generated testcase No. 0 - Program finished with exit status: 0
[9507] m.c.manticore:INFO: Generated testcase No. 1 - Program finished with exit status: 0
[9507] m.c.manticore:INFO: Results in ~/manticore/mcore_7u7hgfay
[9507] m.n.manticore:INFO: Total time: 2.8029580116271973

### API

Manticore provides a Python programming interface which can be used to implement powerful custom
analyses.

#### EVM

For Ethereum smart contracts, the API can be used for detailed verification of arbitrary contract
properties. Users can set the starting conditions, execute symbolic transactions, and then review
discovered states to ensure invariants for a contract hold.

Click to expand:
from manticore.ethereum import ManticoreEVM
contract_src="""
contract Adder {
    function incremented(uint value) public returns (uint){
        if (value == 1)
            revert();
        return value + 1;
    }
}
"""
m = ManticoreEVM()

user_account = m.create_account(balance=10000000)
contract_account = m.solidity_create_contract(contract_src,
                                              owner=user_account,
                                              balance=0)
value = m.make_symbolic_value()

contract_account.incremented(value)

for state in m.ready_states:
    print("can value be 1? {}".format(state.can_be_true(value == 1)))
    print("can value be 200? {}".format(state.can_be_true(value == 200)))

#### Native

It is also possible to use the API to create custom analysis tools for Linux binaries. Tailoring the
initial state helps avoid state explosion problems that commonly occur when using the CLI.

Click to expand:
# example Manticore script
from manticore.native import Manticore

m = Manticore.linux('./example')

@m.hook(0x400ca0)
def hook(state):
  cpu = state.cpu
  print('eax', cpu.EAX)
  print(cpu.read_int(cpu.ESP))

  m.kill()  # tell Manticore to stop

m.run()

#### WASM

Manticore can also evaluate WebAssembly functions over symbolic inputs for property validation or
general analysis.

Click to expand:
from manticore.wasm import ManticoreWASM

m = ManticoreWASM("collatz.wasm")

def arg_gen(state):
    # Generate a symbolic argument to pass to the collatz function.
    # Possible values: 4, 6, 8
    arg = state.new_symbolic_value(32, "collatz_arg")
    state.constrain(arg > 3)
    state.constrain(arg < 9)
    state.constrain(arg % 2 == 0)
    return [arg]


# Run the collatz function with the given argument generator.
m.collatz(arg_gen)

# Manually collect return values
# Prints 2, 3, 8
for idx, val_list in enumerate(m.collect_returns()):
    print("State", idx, "::", val_list[0])

## Requirements

* Manticore requires Python 3.7 or greater
* Manticore officially supports the latest LTS version of Ubuntu provided by Github Actions
  
  * Manticore has experimental support for EVM and WASM (but not native Linux binaries) on MacOS
* We recommend running with increased stack size. This can be done by running `ulimit -s 100000` or
  by passing `--ulimit stack=100000000:100000000` to `docker run`

### Compiling Smart Contracts

* Ethereum smart contract analysis requires the [`solc`][14] program in your `$PATH`.
* Manticore uses [crytic-compile][15] to build smart contracts. If you're having compilation issues,
  consider running `crytic-compile` on your code directly to make it easier to identify any issues.
* We're still in the process of implementing full support for the EVM Istanbul instruction
  semantics, so certain opcodes may not be supported. In a pinch, you can try compiling with
  Solidity 0.4.x to avoid generating those instructions.

## Using a different solver (Yices, Z3, CVC4)

Manticore relies on an external solver supporting smtlib2. Currently Z3, Yices and CVC4 are
supported and can be selected via command-line or configuration settings. If Yices is available,
Manticore will use it by default. If not, it will fall back to Z3 or CVC4. If you want to manually
choose which solver to use, you can do so like this: `manticore --smt.solver Z3`

### Installing CVC4

For more details go to [https://cvc4.github.io/][16]. Otherwise, just get the binary and use it.

`    sudo wget -O /usr/bin/cvc4 https://github.com/CVC4/CVC4/releases/download/1.7/cvc4-1.7-x86_64-l
inux-opt
    sudo chmod +x /usr/bin/cvc4
`

### Installing Yices

Yices is incredibly fast. More details here [https://yices.csl.sri.com/][17]

`    sudo add-apt-repository ppa:sri-csl/formal-methods
    sudo apt-get update
    sudo apt-get install yices2
`

## Getting Help

Feel free to stop by our #manticore slack channel in [Empire Hacking][18] for help using or
extending Manticore.

Documentation is available in several places:

* The [wiki][19] contains information about getting started with Manticore and contributing
* The [API reference][20] has more thorough and in-depth documentation on our API
* The [examples][21] directory has some small examples that showcase API features
* The [manticore-examples][22] repository has some more involved examples, including some real CTF
  problems

If you'd like to file a bug report or feature request, please use our [issues][23] page.

For questions and clarifications, please visit the [discussion][24] page.

## License

Manticore is licensed and distributed under the AGPLv3 license. [Contact us][25] if you're looking
for an exception to the terms.

## Publications

* [Manticore: A User-Friendly Symbolic Execution Framework for Binaries and Smart Contracts][26],
  Mark Mossberg, Felipe Manzano, Eric Hennenfent, Alex Groce, Gustavo Grieco, Josselin Feist, Trent
  Brunson, Artem Dinaburg - ASE 19

If you are using Manticore in academic work, consider applying to the [Crytic $10k Research
Prize][27].

## Demo Video from ASE 2019

[[Brief Manticore demo video]][28]

## Tool Integrations

* [MATE: Merged Analysis To prevent Exploits][29]
  
  * [Mantiserve:][30] REST API interaction with Manticore to start, kill, and check Manticore
    instance
  * [Dwarfcore:][31] Plugins and detectors for use within Mantiserve engine during exploration
  * [Under-constrained symbolic execution][32] Interface for symbolically exploring single functions
    with Manticore

[1]: https://github.com/trailofbits/manticore/actions?query=workflow%3ACI
[2]: https://coveralls.io/github/trailofbits/manticore
[3]: https://badge.fury.io/py/manticore
[4]: https://slack.empirehacking.nyc
[5]: http://manticore.readthedocs.io/en/latest/?badge=latest
[6]: https://github.com/trailofbits/manticore-examples/actions?query=workflow%3ACI
[7]: https://lgtm.com/projects/g/trailofbits/manticore/alerts/
[8]: https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#installing-v
irtualenv
[9]: https://github.com/trailofbits/manticore/wiki/Hacking-on-Manticore
[10]: https://github.com/trailofbits/manticore/wiki/What's-in-the-workspace%3F
[11]: https://asciinema.org/a/154012
[12]: http://manticore.readthedocs.io/en/latest/verifier.html
[13]: https://asciinema.org/a/xd0XYe6EqHCibae0RP6c7sJVE
[14]: https://github.com/ethereum/solidity
[15]: https://github.com/crytic/crytic-compile
[16]: https://cvc4.github.io/
[17]: https://yices.csl.sri.com/
[18]: https://slack.empirehacking.nyc/
[19]: https://github.com/trailofbits/manticore/wiki
[20]: http://manticore.readthedocs.io/en/latest/
[21]: /trailofbits/manticore/blob/master/examples
[22]: https://github.com/trailofbits/manticore-examples
[23]: https://github.com/trailofbits/manticore/issues/choose
[24]: https://github.com/trailofbits/manticore/discussions
[25]: mailto:opensource@trailofbits.com
[26]: https://arxiv.org/abs/1907.03890
[27]: https://blog.trailofbits.com/2019/11/13/announcing-the-crytic-10k-research-prize/
[28]: https://youtu.be/o6pmBJZpKAc
[29]: https://github.com/GaloisInc/MATE
[30]: https://galoisinc.github.io/MATE/mantiserve.html
[31]: https://galoisinc.github.io/MATE/dwarfcore.html
[32]: https://github.com/GaloisInc/MATE/blob/main/doc/under-constrained-manticore.rst
