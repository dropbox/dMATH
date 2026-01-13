# MN-BaB [[portfolio_view]][1]

Multi-Neuron Guided Branch-and-Bound ([MN-BaB][2]) is a state-of-the-art complete neural network
verifier that builds on the tight multi-neuron constraints proposed in [PRIMA][3] and leverages
these constraints within a BaB framework to yield an efficient, GPU based dual solver. MN-BaB is
developed at the [SRI Lab, Department of Computer Science, ETH Zurich][4] as part of the [Safe AI
project][5].

This version is an adaptation of the [VNN-COMP'22][6] entry allowing for the certification of models
trained with the novel certified training method [SABR][7], without modifications.

### Cloning

This repository contains a submodule. Please make sure that you have access rights to the submodule
repository for cloning. After that either clone recursively via

`git clone --branch SABR_ready --recurse-submodules https://github.com/eth-sri/mn-bab
`

or clone normally and initialize the submodule later on

`git clone --branch SABR_ready https://github.com/eth-sri/mn-bab
git submodule init
git submodule update
`

There's no need for a further installation of the submodules.

### Installation

Create and activate a conda environment:

`  conda create --name MNBAB python=3.7 -y
  conda activate MNBAB
`

This script installs a few necessary prerequisites including the ELINA library and GUROBI solver and
sets some PATHS. It was tested on an AWS Deep Learning AMI (Ubuntu 18.04) instance.

`source setup.sh
`

Install remaining dependencies:

`python3 -m pip install -r requirements.txt
PYTHONPATH=$PYTHONPATH:$PWD
`

Download the full MNIST, CIFAR10, and TinyImageNet test datasets in the right format and copy them
into the `test_data` directory:
[MNIST][8]
[CIFAR10][9]
[TinyImageNet][10]

### Example usage

`python src/verify.py -c configs/cifar10_conv_small.json
`

## Contributors

* [Claudio Ferrari ][11] - [c101@gmx.ch][12]
* [Mark Niklas Müller][13] - [mark.mueller@inf.ethz.ch][14]
* [Nikola Jovanovic][15] - [nikola.jovanovic@inf.ethz.ch][16]
* [Robin Staab][17]
* [Dr. Timon Gehr][18]

## Citing This Work

If you find this work useful for your research, please cite it as:

`@inproceedings{
    ferrari2022complete,
    title={Complete Verification via Multi-Neuron Relaxation Guided Branch-and-Bound},
    author={Claudio Ferrari and Mark Niklas Mueller and Nikola Jovanovi{\'c} and Martin Vechev},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=l_amHf1oaK}
}
`

## License and Copyright

* Copyright (c) 2022 [Secure, Reliable, and Intelligent Systems Lab (SRI), Department of Computer
  Science ETH Zurich][19]
* Licensed under the [Apache License][20]

[1]: https://camo.githubusercontent.com/04ec00690643981ec02cef1d550b7477dac53440fd7439eb84f40c1f1cff
7fa2/687474703a2f2f7361666561692e6574687a2e63682f696d672f7372692d6c6f676f2e737667
[2]: https://www.sri.inf.ethz.ch/publications/ferrari2022complete
[3]: https://www.sri.inf.ethz.ch/publications/mueller2021precise
[4]: https://www.sri.inf.ethz.ch/
[5]: http://safeai.ethz.ch/
[6]: https://arxiv.org/abs/2212.10376
[7]: https://openreview.net/forum?id=7oFuxtJtUMH
[8]: https://files.sri.inf.ethz.ch/sabr/mnist_test_full.csv
[9]: https://files.sri.inf.ethz.ch/sabr/cifar10_test_full.csv
[10]: https://files.sri.inf.ethz.ch/sabr/tin_val.csv
[11]: https://github.com/ferraric
[12]: mailto:c101@gmx.ch
[13]: https://www.sri.inf.ethz.ch/people/mark
[14]: mailto:mark.mueller@inf.ethz.ch
[15]: https://www.sri.inf.ethz.ch/people/nikola
[16]: mailto:nikola.jovanovic@inf.ethz.ch
[17]: /eth-sri/mn-bab/blob/SABR_ready
[18]: https://www.sri.inf.ethz.ch/people/timon
[19]: https://www.sri.inf.ethz.ch/
[20]: https://www.apache.org/licenses/LICENSE-2.0
