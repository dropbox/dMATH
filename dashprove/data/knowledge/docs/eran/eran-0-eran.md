# ERAN [[portfolio_view]][1]

[[High Level]][2]

ETH Robustness Analyzer for Neural Networks (ERAN) is a state-of-the-art sound,
precise, scalable, and extensible analyzer based on [abstract interpretation][3]
for the complete and incomplete verification of MNIST, CIFAR10, and ACAS Xu
based networks. ERAN produces state-of-the-art precision and performance for
both complete and incomplete verification and can be tuned to provide best
precision and scalability (see recommended configuration settings at the
bottom). ERAN is developed at the [SRI Lab, Department of Computer Science, ETH
Zurich][4] as part of the [Safe AI project][5]. The goal of ERAN is to
automatically verify safety properties of neural networks with feedforward,
convolutional, and residual layers against input perturbations (e.g., L∞-norm
attacks, geometric transformations, vector field deformations, etc).

ERAN combines abstract domains with custom multi-neuron relaxations from PRIMA
to support fully-connected, convolutional, and residual networks with ReLU,
Sigmoid, Tanh, and Maxpool activations. ERAN is sound under floating point
arithmetic with the exception of the (MI)LP solver used in RefineZono and
RefinePoly. The employed abstract domains are specifically designed for the
setting of neural networks and aim to balance scalability and precision.
Specifically, ERAN supports the following analysis:

* DeepZ [NIPS'18]: contains specialized abstract Zonotope transformers for
  handling ReLU, Sigmoid and Tanh activation functions.
* DeepPoly [POPL'19]: based on a domain that combines floating point Polyhedra
  with Intervals.
* GPUPoly [MLSys'2021]: leverages an efficient GPU implementation to scale
  DeepPoly to much larger networks.
* RefineZono [ICLR'19]: combines DeepZ analysis with MILP and LP solvers for
  more precision.
* RefinePoly/RefineGPUPoly [NeurIPS'19]: combines DeepPoly/GPUPoly analysis with
  (MI)LP refinement and PRIMA framework [arXiv'2021] to compute group-wise joint
  neuron abstractions for state-of-the-art precision and scalability.

All analysis are implemented using the [ELINA][6] library for numerical
abstractions. More details can be found in the publications below.

## ERAN vs AI2

Note that ERAN subsumes the first abstract interpretation based analyzer
[AI2][7], so if you aim to compare, please use ERAN as a baseline.

## USER MANUAL

For a detailed desciption of the options provided and the implentation of ERAN,
you can download the [user manual][8].

## Requirements

GNU C compiler, ELINA, Gurobi's Python interface,

python3.6 or higher, tensorflow 1.11 or higher, numpy.

## Installation

Clone the ERAN repository via git as follows:

`git clone https://github.com/eth-sri/ERAN.git
cd ERAN
`

The dependencies for ERAN can be installed step by step as follows (sudo rights
might be required):
Note that it might be required to use `sudo -E` to for the right environment
variables to be set.

Ensure that the following tools are available before using the install script:

* cmake (>=3.17.1),
* m4 (>=1.4.18)
* autoconf,
* libtool,
* pdftex.

On Ubuntu systems they can be installed using:

`sudo apt-get install m4
sudo apt-get install build-essential
sudo apt-get install autoconf
sudo apt-get install libtool
sudo apt-get install texlive-latex-base
`

Consult [https://cmake.org/cmake/help/latest/command/install.html][9] for the
install of cmake or use:

`wget https://github.com/Kitware/CMake/releases/download/v3.19.7/cmake-3.19.7-Li
nux-x86_64.sh
sudo bash ./cmake-3.19.7-Linux-x86_64.sh
sudo rm /usr/bin/cmake
sudo ln -s ./cmake-3.19.7-Linux-x86_64/bin/cmake /usr/bin/cmake
`

The steps following from here can be done automatically using `sudo bash
./install.sh`

Install gmp:

`wget https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz
tar -xvf gmp-6.1.2.tar.xz
cd gmp-6.1.2
./configure --enable-cxx
make
make install
cd ..
rm gmp-6.1.2.tar.xz
`

Install mpfr:

`wget https://files.sri.inf.ethz.ch/eran/mpfr/mpfr-4.1.0.tar.xz
tar -xvf mpfr-4.1.0.tar.xz
cd mpfr-4.1.0
./configure
make
make install
cd ..
rm mpfr-4.1.0.tar.xz
`

Install cddlib:

`wget https://github.com/cddlib/cddlib/releases/download/0.94m/cddlib-0.94m.tar.
gz
tar zxf cddlib-0.94m.tar.gz
rm cddlib-0.94m.tar.gz
cd cddlib-0.94m
./configure
make
make install
cd ..
`

Install Gurobi:

`wget https://packages.gurobi.com/9.1/gurobi9.1.2_linux64.tar.gz
tar -xvf gurobi9.1.2_linux64.tar.gz
cd gurobi912/linux64/src/build
sed -ie 's/^C++FLAGS =.*$/& -fPIC/' Makefile
make
cp libgurobi_c++.a ../../lib/
cd ../../
cp lib/libgurobi91.so /usr/local/lib
python3 setup.py install
cd ../../
`

Update environment variables:

`export GUROBI_HOME="$PWD/gurobi912/linux64"
export PATH="$PATH:${GUROBI_HOME}/bin"
export CPATH="$CPATH:${GUROBI_HOME}/include"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib:${GUROBI_HOME}/lib
`

Install ELINA:

`git clone https://github.com/eth-sri/ELINA.git
cd ELINA
./configure -use-deeppoly -use-gurobi -use-fconv -use-cuda
cd ./gpupoly/
cmake .
cd ..
make
make install
cd ..
`

Install DeepG (note that with an already existing version of ERAN you have to
start at step Install Gurobi):

`git clone https://github.com/eth-sri/deepg.git
cd deepg/code
mkdir build
make shared_object
cp ./build/libgeometric.so /usr/lib
cd ../..
`

We also provide scripts that will install ELINA and all the necessary
dependencies. One can run it as follows (remove the `-use-cuda` argument on
machines without GPU):

`sudo ./install.sh -use-cuda
source gurobi_setup_path.sh
`

Note that to run ERAN with Gurobi one needs to obtain an academic license for
gurobi from [https://user.gurobi.com/download/licenses/free-academic][10]. If
you plan on running ERAN on Windows WSL2, you might prefer requesting a
cloud-based academic license at [https://license.gurobi.com][11], in order to
avoid [this issue][12] with early-expiring licenses.

To install the remaining python dependencies (numpy and tensorflow), type:

`pip3 install -r requirements.txt
`

ERAN may not be compatible with older versions of tensorflow (we have tested
ERAN with versions >= 1.11.0), so if you have an older version and want to keep
it, then we recommend using the python virtual environment for installing
tensorflow.

If gurobipy is not found despite executing `python setup.py install` in the
corresponding gurobi directory, gurobipy can alternatively be installed using
conda with:

`conda config --add channels http://conda.anaconda.org/gurobi
conda install gurobi
`

## Usage

`cd tf_verify

python3 . --netname <path to the network file> --epsilon <float between 0 and 1>
 --domain <deepzono/deeppoly/refinezono/refinepoly> --dataset <mnist/cifar10/aca
sxu> --zonotope <path to the zonotope specfile>  [optional] --complete <True/Fal
se> --timeout_complete <float> --timeout_lp <float> --timeout_milp <float> --use
_area_heuristic <True/False> --mean <float(s)> --std <float(s)> --use_milp <True
/False> --use_2relu --use_3relu --dyn_krelu --numproc <int>
`

* `<epsilon>`: specifies bound for the L∞-norm based perturbation (default is
  0). This parameter is not required for testing ACAS Xu networks.
* `<zonotope>`: The Zonotope specification file can be comma or whitespace
  separated file where the first two integers can specify the number of input
  dimensions D and the number of error terms per dimension N. The following D*N
  doubles specify the coefficient of error terms. For every dimension i, the
  error terms are numbered from 0 to N-1 where the 0-th error term is the
  central error term. See an example here
  [[https://github.com/eth-sri/eran/files/3653882/zonotope_example.txt][13]].
  This option only works with the "deepzono" or "refinezono" domain.
* `<use_area_heuristic>`: specifies whether to use area heuristic for the ReLU
  approximation in DeepPoly (default is true).
* `<mean>`: specifies mean used to normalize the data. If the data has multiple
  channels the mean for every channel has to be provided (e.g. for cifar10
  --mean 0.485, 0.456, 0.406) (default is 0 for non-geometric mnist and 0.5 0.5
  0.5 otherwise)
* `<std>`: specifies standard deviation used to normalize the data. If the data
  has multiple channels the standard deviaton for every channel has to be
  provided (e.g. for cifar10 --std 0.2 0.3 0.2) (default is 1 1 1)
* `<use_milp>`: specifies whether to use MILP (default is true).
* `<sparse_n>`: specifies the size of "k" for the kReLU framework (default is
  70).
* `<numproc>`: specifies how many processes to use for MILP, LP and k-ReLU
  (default is the number of processors in your machine).
* `<geometric>`: specifies whether to do geometric analysis (default is false).
* `<geometric_config>`: specifies the geometric configuration file location.
* `<data_dir>`: specifies the geometric data location.
* `<num_params>`: specifies the number of transformation parameters (default is
  0)
* `<attack>`: specifies whether to verify attack images (default is false).
* `<specnumber>`: the property number for the ACASXu networks
* Refinezono and RefinePoly refines the analysis results from the DeepZ and
  DeepPoly domain respectively using the approach in our ICLR'19 paper. The
  optional parameters timeout_lp and timeout_milp (default is 1 sec for both)
  specify the timeouts for the LP and MILP forumlations of the network
  respectively.
* Since Refinezono and RefinePoly uses timeout for the gurobi solver, the
  results will vary depending on the processor speeds.
* Setting the parameter "complete" (default is False) to True will enable MILP
  based complete verification using the bounds provided by the specified domain.
* When ERAN fails to prove the robustness of a given network in a specified
  region, it searches for an adversarial example and prints an adversarial image
  within the specified adversarial region along with the misclassified label and
  the correct label. ERAN does so for both complete and incomplete verification.

## Example

L_oo Specification

`python3 . --netname ../nets/onnx/mnist/convBig__DiffAI.onnx --epsilon 0.1 --dom
ain deepzono --dataset mnist
`

will evaluate the local robustness of the MNIST convolutional network (upto 35K
neurons) with ReLU activation trained using DiffAI on the 100 MNIST test images.
In the above setting, epsilon=0.1 and the domain used by our analyzer is the
deepzono domain. Our analyzer will print the following:

* 'Verified safe' for an image when it can prove the robustness of the network
* 'Verified unsafe' for an image for which it can provide a concrete adversarial
  example
* 'Failed' when it cannot.
* It will also print an error message when the network misclassifies an image.
* the timing in seconds.
* The ratio of images on which the network is robust versus the number of images
  on which it classifies correctly.

Zonotope Specification

`python3 . --netname ../nets/onnx/mnist/convBig__DiffAI.onnx --zonotope some_pat
h/zonotope_example.txt --domain deepzono 
`

will check if the Zonotope specification specified in "zonotope_example" holds
for the network and will output "Verified safe", "Verified unsafe" or "Failed"
along with the timing.

Similarly, for the ACAS Xu networks, ERAN will output whether the property has
been verified along with the timing.

ACASXu Specification

`python3 . --netname ../data/acasxu/nets/ACASXU_run2a_3_3_batch_2000.onnx --data
set acasxu --domain deepzono  --specnumber 9
`

will run DeepZ for analyzing property 9 of ACASXu benchmarks. The ACASXU
networks are in data/acasxu/nets directory and the one chosen for a given
property is defined in the Reluplex paper.

Geometric analysis

`python3 . --netname ../nets/onnx/mnist/convBig__DiffAI.onnx --geometric --geome
tric_config ../deepg/code/examples/example1/config.txt --num_params 1 --dataset 
mnist
`

will on the fly generate geometric perturbed images and evaluate the network
against them. For more information on the geometric configuration file please
see [Format of the configuration file in DeepG][14].

`python3 . --netname ../nets/onnx/mnist/convBig__DiffAI.onnx --geometric --data_
dir ../deepg/code/examples/example1/ --num_params 1 --dataset mnist --attack
`

will evaluate the generated geometric perturbed images in the given data_dir and
also evaluate generated attack images.

## Recommended Configuration for Scalable Complete Verification

Use the "deeppoly" or "deepzono" domain with "--complete True" option

## Recommended Configuration for More Precise but relatively expensive
## Incomplete Verification

Use the "refinepoly" or if a gpu is available "refinegpupoly" domain with ,
"--sparse_n 100", and "--timeout_final_lp 100".
For MLPs use "--refine_neurons", "--use_milp True", "--timeout_milp 10",
"--timeout_lp 10" to do a neuronweise bound refinement.
For Conv networks use "--partial_milp {1,2}" (choose at most number of linear
layers), "--max_milp_neurons 100", and "--timeout_final_milp 250" to use a MILP
encoding for the last layers.

Examples:
To certify e.g. [CNN-B-ADV][15] introduced as a benchmark for SDP-FO in
[[1]][16] on the [100 random samples][17] from [[2]][18] against L-inf
perturbations of magnitude 2/255 use:

`python3 . --netname ../nets/CNN_B_CIFAR_ADV.onnx --dataset cifar10  --subset b_
adv --domain refinegpupoly --epsilon 0.00784313725 --sparse_n 100 --partial_milp
 1 --max_milp_neurons 250 --timeout_final_milp 500 --mean 0.49137255 0.48235294 
0.44666667 --std 0.24705882 0.24352941 0.26156863
`

to certify 43 of the 100 samples as correct with an average runtime of around
260s per sample (including timed out attempts).

## Recommended Configuration for Faster but relatively imprecise Incomplete
## Verification

Use the "deeppoly" or if a gpu is available "gpupoly" domain

## Certification of Vector Field Deformations

[[High Level]][19]

Vector field deformations, which displace pixels instead of directly
manipulating pixel values, can be intuitively parametrized by their displacement
magnitude delta, i.e., how far every pixel can move, and their smoothness gamma,
i.e., how much neighboring displacement vectors can differ from each other (more
details can be found in Section 3 of [our paper][20]). ERAN can certify both
non-smooth vector fields:

`python3 . --netname ../nets/onnx/mnist/convBig__DiffAI.onnx --dataset mnist --d
omain deeppoly --spatial --t-norm inf --delta 0.3
`

and smooth vector fields:

`python3 . --netname ../nets/pytorch/onnx/convBig__DiffAI.onnx --dataset mnist -
-domain deeppoly --spatial --t-norm inf --delta 0.3 --gamma 0.1
`

Certification of vector field deformations is compatible with the "deeppoly" and
"refinepoly" domains, and can be made more precise with the kReLU framework
(e.g., "--use_milp True", "--sparse_n 15", "--refine_neurons", "timeout_milp
10", and "timeout_lp 10") or complete certification ("--complete True").

## Publications

* [PRIMA: Precise and General Neural Network Certification via Multi-Neuron
  Convex Relaxations][21]
  
  Mark Niklas Müller, Gleb Makarchuk, Gagandeep Singh, Markus Püschel, Martin
  Vechev
  
  POPL 2022.
* [Scaling Polyhedral Neural Network Verification on GPUs][22]
  
  Christoph Müller, Francois Serre, Gagandeep Singh, Markus Puschel, Martin
  Vechev
  
  MLSys 2021.
* [Efficient Certification of Spatial Robustness][23]
  
  Anian Ruoss, Maximilian Baader, Mislav Balunovic, Martin Vechev
  
  AAAI 2021.
* [Certifying Geometric Robustness of Neural Networks][24]
  
  Mislav Balunovic, Maximilian Baader, Gagandeep Singh, Timon Gehr, Martin
  Vechev
  
  NeurIPS 2019.
* [Beyond the Single Neuron Convex Barrier for Neural Network
  Certification][25].
  
  Gagandeep Singh, Rupanshu Ganvir, Markus Püschel, and Martin Vechev.
  
  NeurIPS 2019.
* [Boosting Robustness Certification of Neural Networks][26].
  
  Gagandeep Singh, Timon Gehr, Markus Püschel, and Martin Vechev.
  
  ICLR 2019.
* [An Abstract Domain for Certifying Neural Networks][27].
  
  Gagandeep Singh, Timon Gehr, Markus Püschel, and Martin Vechev.
  
  POPL 2019.
* [Fast and Effective Robustness Certification][28].
  
  Gagandeep Singh, Timon Gehr, Matthew Mirman, Markus Püschel, and Martin
  Vechev.
  
  NeurIPS 2018.

## Neural Networks and Datasets

We provide a number of pretrained MNIST and CIAFR10 defended and undefended
feedforward and convolutional neural networks with ReLU, Sigmoid and Tanh
activations trained with the PyTorch and TensorFlow frameworks. The adversarial
training to obtain the defended networks is performed using PGD and
[DiffAI][29]. We report the (maximum) number of activation layers (including
MaxPool) of any path through a network.

──────┬─────────┬────────────┬─────┬─────────────┬────────┬─────────────┬───────
Datase│Model    │Type        │#unit│#activation  │Activati│Training     │Downloa
t     │         │            │s    │layers       │on      │Defense      │d      
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
MNIST │3x50     │fully       │110  │3            │ReLU    │None         │[⬇️][30]
      │         │connected   │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │3x100    │fully       │210  │3            │ReLU    │None         │[⬇️][31]
      │         │connected   │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │5x100    │fully       │510  │6            │ReLU    │DiffAI       │[⬇️][32]
      │         │connected   │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │6x100    │fully       │510  │6            │ReLU    │None         │[⬇️][33]
      │         │connected   │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │9x100    │fully       │810  │9            │ReLU    │None         │[⬇️][34]
      │         │connected   │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │6x200    │fully       │1,010│6            │ReLU    │None         │[⬇️][35]
      │         │connected   │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │9x200    │fully       │1,610│9            │ReLU    │None         │[⬇️][36]
      │         │connected   │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │6x500    │fully       │3,000│6            │ReLU    │None         │[⬇️][37]
      │         │connected   │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │6x500    │fully       │3,000│6            │ReLU    │PGD ε=0.1    │[⬇️][38]
      │         │connected   │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │6x500    │fully       │3,000│6            │ReLU    │PGD ε=0.3    │[⬇️][39]
      │         │connected   │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │6x500    │fully       │3,000│6            │Sigmoid │None         │[⬇️][40]
      │         │connected   │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │6x500    │fully       │3,000│6            │Sigmoid │PGD ε=0.1    │[⬇️][41]
      │         │connected   │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │6x500    │fully       │3,000│6            │Sigmoid │PGD ε=0.3    │[⬇️][42]
      │         │connected   │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │6x500    │fully       │3,000│6            │Tanh    │None         │[⬇️][43]
      │         │connected   │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │6x500    │fully       │3,000│6            │Tanh    │PGD ε=0.1    │[⬇️][44]
      │         │connected   │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │6x500    │fully       │3,000│6            │Tanh    │PGD ε=0.3    │[⬇️][45]
      │         │connected   │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │4x1024   │fully       │3,072│3            │ReLU    │None         │[⬇️][46]
      │         │connected   │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │ConvSmall│convolutiona│3,604│3            │ReLU    │None         │[⬇️][47]
      │         │l           │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │ConvSmall│convolutiona│3,604│3            │ReLU    │PGD          │[⬇️][48]
      │         │l           │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │ConvSmall│convolutiona│3,604│3            │ReLU    │DiffAI       │[⬇️][49]
      │         │l           │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │ConvMed  │convolutiona│5,704│3            │ReLU    │None         │[⬇️][50]
      │         │l           │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │ConvMed  │convolutiona│5,704│3            │ReLU    │PGD ε=0.1    │[⬇️][51]
      │         │l           │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │ConvMed  │convolutiona│5,704│3            │ReLU    │PGD ε=0.3    │[⬇️][52]
      │         │l           │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │ConvMed  │convolutiona│5,704│3            │Sigmoid │None         │[⬇️][53]
      │         │l           │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │ConvMed  │convolutiona│5,704│3            │Sigmoid │PGD ε=0.1    │[⬇️][54]
      │         │l           │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │ConvMed  │convolutiona│5,704│3            │Sigmoid │PGD ε=0.3    │[⬇️][55]
      │         │l           │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │ConvMed  │convolutiona│5,704│3            │Tanh    │None         │[⬇️][56]
      │         │l           │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │ConvMed  │convolutiona│5,704│3            │Tanh    │PGD ε=0.1    │[⬇️][57]
      │         │l           │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │ConvMed  │convolutiona│5,704│3            │Tanh    │PGD ε=0.3    │[⬇️][58]
      │         │l           │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │ConvMaxpo│convolutiona│13,79│9            │ReLU    │None         │[⬇️][59]
      │ol       │l           │8    │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │ConvBig  │convolutiona│48,06│6            │ReLU    │DiffAI       │[⬇️][60]
      │         │l           │4    │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │ConvSuper│convolutiona│88,54│6            │ReLU    │DiffAI       │[⬇️][61]
      │         │l           │4    │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │Skip     │Residual    │71,65│6            │ReLU    │DiffAI       │[⬇️][62]
      │         │            │0    │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
CIFAR1│4x100    │fully       │410  │5            │ReLU    │None         │[⬇️][63]
0     │         │connected   │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │6x100    │fully       │610  │7            │ReLU    │None         │[⬇️][64]
      │         │connected   │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │9x200    │fully       │1,810│10           │ReLU    │None         │[⬇️][65]
      │         │connected   │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │6x500    │fully       │3,000│6            │ReLU    │None         │[⬇️][66]
      │         │connected   │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │6x500    │fully       │3,000│6            │ReLU    │PGD ε=0.0078 │[⬇️][67]
      │         │connected   │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │6x500    │fully       │3,000│6            │ReLU    │PGD ε=0.0313 │[⬇️][68]
      │         │connected   │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │6x500    │fully       │3,000│6            │Sigmoid │None         │[⬇️][69]
      │         │connected   │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │6x500    │fully       │3,000│6            │Sigmoid │PGD ε=0.0078 │[⬇️][70]
      │         │connected   │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │6x500    │fully       │3,000│6            │Sigmoid │PGD ε=0.0313 │[⬇️][71]
      │         │connected   │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │6x500    │fully       │3,000│6            │Tanh    │None         │[⬇️][72]
      │         │connected   │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │6x500    │fully       │3,000│6            │Tanh    │PGD ε=0.0078 │[⬇️][73]
      │         │connected   │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │6x500    │fully       │3,000│6            │Tanh    │PGD ε=0.0313 │[⬇️][74]
      │         │connected   │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │7x1024   │fully       │6,144│6            │ReLU    │None         │[⬇️][75]
      │         │connected   │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │ConvSmall│convolutiona│4,852│3            │ReLU    │None         │[⬇️][76]
      │         │l           │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │ConvSmall│convolutiona│4,852│3            │ReLU    │PGD          │[⬇️][77]
      │         │l           │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │ConvSmall│convolutiona│4,852│3            │ReLU    │DiffAI       │[⬇️][78]
      │         │l           │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │ConvMed  │convolutiona│7,144│3            │ReLU    │None         │[⬇️][79]
      │         │l           │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │ConvMed  │convolutiona│7,144│3            │ReLU    │PGD ε=0.0078 │[⬇️][80]
      │         │l           │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │ConvMed  │convolutiona│7,144│3            │ReLU    │PGD ε=0.0313 │[⬇️][81]
      │         │l           │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │ConvMed  │convolutiona│7,144│3            │Sigmoid │None         │[⬇️][82]
      │         │l           │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │ConvMed  │convolutiona│7,144│3            │Sigmoid │PGD ε=0.0078 │[⬇️][83]
      │         │l           │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │ConvMed  │convolutiona│7,144│3            │Sigmoid │PGD ε=0.0313 │[⬇️][84]
      │         │l           │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │ConvMed  │convolutiona│7,144│3            │Tanh    │None         │[⬇️][85]
      │         │l           │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │ConvMed  │convolutiona│7,144│3            │Tanh    │PGD ε=0.0078 │[⬇️][86]
      │         │l           │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │ConvMed  │convolutiona│7,144│3            │Tanh    │PGD ε=0.0313 │[⬇️][87]
      │         │l           │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │ConvMaxpo│convolutiona│53,93│9            │ReLU    │None         │[⬇️][88]
      │ol       │l           │8    │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │ConvBig  │convolutiona│62,46│6            │ReLU    │DiffAI       │[⬇️][89]
      │         │l           │4    │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │ResNetTin│Residual    │311K │12           │ReLU    │PGD ε=0.0313 │[⬇️][90]
      │y        │            │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │ResNetTin│Residual    │311K │12           │ReLU    │DiffAI       │[⬇️][91]
      │y        │            │     │             │        │             │       
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │ResNet18 │Residual    │558K │19           │ReLU    │PGD          │[⬇️][92]
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │ResNet18 │Residual    │558K │19           │ReLU    │DiffAI       │[⬇️][93]
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │SkipNet18│Residual    │558K │19           │ReLU    │DiffAI       │[⬇️][94]
──────┼─────────┼────────────┼─────┼─────────────┼────────┼─────────────┼───────
      │ResNet34 │Residual    │967K │35           │ReLU    │DiffAI       │[⬇️][95]
──────┴─────────┴────────────┴─────┴─────────────┴────────┴─────────────┴───────

We provide the first 100 images from the testset of both MNIST and CIFAR10
datasets in the 'data' folder. Our analyzer first verifies whether the neural
network classifies an image correctly before performing robustness analysis. In
the same folder, we also provide ACAS Xu networks and property specifications.

## Experimental Results

We ran our experiments for the feedforward networks on a 3.3 GHz 10 core Intel
i9-7900X Skylake CPU with a main memory of 64 GB whereas our experiments for the
convolutional networks were run on a 2.6 GHz 14 core Intel Xeon CPU E5-2690 with
512 GB of main memory. We first compare the precision and performance of DeepZ
and DeepPoly vs [Fast-Lin][96] on the MNIST 6x100 network in single threaded
mode. It can be seen that DeepZ has the same precision as Fast-Lin whereas
DeepPoly is more precise while also being faster.

[[High Level]][97]

In the following, we compare the precision and performance of DeepZ and DeepPoly
on a subset of the neural networks listed above in multi-threaded mode. In can
be seen that DeepPoly is overall more precise than DeepZ but it is slower than
DeepZ on the convolutional networks.

[[High Level]][98]

[[High Level]][99]

[[High Level]][100]

[[High Level]][101]

The table below compares the performance and precision of DeepZ and DeepPoly on
our large networks trained with DiffAI.

────────┬──────────┬──────┬─────────────────────┬─────────────────────
Dataset │Model     │ε     │% Verified Robustness│% Average Runtime (s)
────────┼──────────┼──────┼──────────┬──────────┼──────────┬──────────
        │          │      │DeepZ     │DeepPoly  │DeepZ     │DeepPoly  
────────┼──────────┼──────┼──────────┼──────────┼──────────┼──────────
MNIST   │ConvBig   │0.1   │97        │97        │5         │50        
────────┼──────────┼──────┼──────────┼──────────┼──────────┼──────────
        │ConvBig   │0.2   │79        │78        │7         │61        
────────┼──────────┼──────┼──────────┼──────────┼──────────┼──────────
        │ConvBig   │0.3   │37        │43        │17        │88        
────────┼──────────┼──────┼──────────┼──────────┼──────────┼──────────
        │ConvSuper │0.1   │97        │97        │133       │400       
────────┼──────────┼──────┼──────────┼──────────┼──────────┼──────────
        │Skip      │0.1   │95        │N/A       │29        │N/A       
────────┼──────────┼──────┼──────────┼──────────┼──────────┼──────────
CIFAR10 │ConvBig   │0.006 │50        │52        │39        │322       
────────┼──────────┼──────┼──────────┼──────────┼──────────┼──────────
        │ConvBig   │0.008 │33        │40        │46        │331       
────────┴──────────┴──────┴──────────┴──────────┴──────────┴──────────

The table below compares the timings of complete verification with ERAN for all
ACASXu benchmarks.

────────┬────────┬─────────────────────
Property│Networks│% Average Runtime (s)
────────┼────────┼─────────────────────
1       │all 45  │15.5                 
────────┼────────┼─────────────────────
2       │all 45  │11.4                 
────────┼────────┼─────────────────────
3       │all 45  │1.9                  
────────┼────────┼─────────────────────
4       │all 45  │1.1                  
────────┼────────┼─────────────────────
5       │1_1     │26                   
────────┼────────┼─────────────────────
6       │1_1     │10                   
────────┼────────┼─────────────────────
7       │1_9     │83                   
────────┼────────┼─────────────────────
8       │2_9     │111                  
────────┼────────┼─────────────────────
9       │3_3     │9                    
────────┼────────┼─────────────────────
10      │4_5     │2.1                  
────────┴────────┴─────────────────────

The table below shows the certification performance of PRIMA (refinepoly with
Precise Multi-Neuron Relacations). For MLPs we use CPU only certificaiton, while
we use GPUPoly for the certification of the convolutional networks.

─────────┬────────┬────┬────┬──────┬────────┬──────────┬───┬─┬─────┬────────────
Network  │Data    │Accu│Epsi│Upper │PRIMA   │PRIMA     │N  │K│Refin│Partial MILP
         │subset  │racy│lon │Bound │certifie│runtime   │   │ │ement│(layers/max_
         │        │    │    │      │d       │[s]       │   │ │     │neurons)    
─────────┼────────┼────┼────┼──────┼────────┼──────────┼───┼─┼─────┼────────────
MNIST    │        │    │    │      │        │          │   │ │     │            
─────────┼────────┼────┼────┼──────┼────────┼──────────┼───┼─┼─────┼────────────
6x100    │first   │960 │0.02│842   │510     │159.2     │100│3│y    │            
[NOR]    │1000    │    │6   │      │        │          │   │ │     │            
─────────┼────────┼────┼────┼──────┼────────┼──────────┼───┼─┼─────┼────────────
9x100    │first   │947 │0.02│820   │428     │300.63    │100│3│y    │            
[NOR]    │1000    │    │6   │      │        │          │   │ │     │            
─────────┼────────┼────┼────┼──────┼────────┼──────────┼───┼─┼─────┼────────────
6x200    │first   │972 │0.01│901   │690     │223.6     │50 │3│y    │            
[NOR]    │1000    │    │5   │      │        │          │   │ │     │            
─────────┼────────┼────┼────┼──────┼────────┼──────────┼───┼─┼─────┼────────────
9x200    │first   │950 │0.01│911   │624     │394.6     │50 │3│y    │            
[NOR]    │1000    │    │5   │      │        │          │   │ │     │            
─────────┼────────┼────┼────┼──────┼────────┼──────────┼───┼─┼─────┼────────────
ConvSmall│first   │980 │0.12│746   │598     │41.7      │100│3│n    │1/30        
[NOR]    │1000    │    │    │      │        │          │   │ │     │            
─────────┼────────┼────┼────┼──────┼────────┼──────────┼───┼─┼─────┼────────────
ConvBig  │first   │929 │0.3 │804   │775     │15.3      │100│3│n    │2/30        
[DiffAI] │1000    │    │    │      │        │          │   │ │     │            
─────────┼────────┼────┼────┼──────┼────────┼──────────┼───┼─┼─────┼────────────
CIFAR-10 │        │    │    │      │        │          │   │ │     │            
─────────┼────────┼────┼────┼──────┼────────┼──────────┼───┼─┼─────┼────────────
ConvSmall│first   │630 │2/25│482   │446     │13.25     │100│3│n    │1/100       
[PGD]    │1000    │    │5   │      │        │          │   │ │     │            
─────────┼────────┼────┼────┼──────┼────────┼──────────┼───┼─┼─────┼────────────
ConvBig  │first   │631 │2/25│613   │483     │175.9     │100│3│n    │2/512       
[PGD]    │1000    │    │5   │      │        │          │   │ │     │            
─────────┼────────┼────┼────┼──────┼────────┼──────────┼───┼─┼─────┼────────────
ResNet   │first   │289 │8/25│253   │249     │63.5      │50 │3│n    │            
[Wong]   │1000    │    │5   │      │        │          │   │ │     │            
─────────┼────────┼────┼────┼──────┼────────┼──────────┼───┼─┼─────┼────────────
CNN-A    │Beta-CRO│100 │2/25│69    │50      │20.96     │100│3│n    │1/100       
[MIX]    │WN 100  │    │5   │      │        │          │   │ │     │            
─────────┼────────┼────┼────┼──────┼────────┼──────────┼───┼─┼─────┼────────────
CNN-B    │Beta-CRO│100 │2/25│83    │43      │259.7     │100│3│n    │1/250       
[ADV]    │WN 100  │    │5   │      │        │          │   │ │     │            
─────────┴────────┴────┴────┴──────┴────────┴──────────┴───┴─┴─────┴────────────

More experimental results can be found in our papers.

## Contributors

* [Mark Niklas Müller][102] (lead contact) - [mark.mueller@inf.ethz.ch][103]
* [Gagandeep Singh][104] (contact for ELINA) - [ggnds@illinois.edu][105]
  [gagandeepsi@vmware.com][106]
* [Mislav Balunovic][107] (contact for geometric certification) -
  [mislav.balunovic@inf.ethz.ch][108]
* Gleb Makarchuk (contact for FConv library) - [hlebm@ethz.ch][109]
  [gleb.makarchuk@gmail.com][110]
* Anian Ruoss (contact for spatial certification) - [anruoss@ethz.ch][111]
* [François Serre][112] (contact for GPUPoly) - [serref@inf.ethz.ch][113]
* [Maximilian Baader][114] - [mbaader@inf.ethz.ch][115]
* [Dana Drachsler Cohen][116] - [dana.drachsler@inf.ethz.ch][117]
* [Timon Gehr][118] - [timon.gehr@inf.ethz.ch][119]
* Adrian Hoffmann - [adriahof@student.ethz.ch][120]
* Jonathan Maurer - [maurerjo@student.ethz.ch][121]
* [Matthew Mirman][122] - [matt@mirman.com][123]
* Christoph Müller - [christoph.mueller@inf.ethz.ch][124]
* [Markus Püschel][125] - [pueschel@inf.ethz.ch][126]
* [Petar Tsankov][127] - [petar.tsankov@inf.ethz.ch][128]
* [Martin Vechev][129] - [martin.vechev@inf.ethz.ch][130]

## License and Copyright

* Copyright (c) 2020 [Secure, Reliable, and Intelligent Systems Lab (SRI),
  Department of Computer Science ETH Zurich][131]
* Licensed under the [Apache License][132]

[1]: https://camo.githubusercontent.com/04ec00690643981ec02cef1d550b7477dac53440
fd7439eb84f40c1f1cff7fa2/687474703a2f2f7361666561692e6574687a2e63682f696d672f737
2692d6c6f676f2e737667
[2]: https://raw.githubusercontent.com/eth-sri/eran/master/overview.png
[3]: https://en.wikipedia.org/wiki/Abstract_interpretation
[4]: https://www.sri.inf.ethz.ch/
[5]: http://safeai.ethz.ch/
[6]: http://elina.ethz.ch/
[7]: https://www.sri.inf.ethz.ch/publications/gehr2018ai
[8]: https://files.sri.inf.ethz.ch/eran/docs/eran_manual.pdf
[9]: https://cmake.org/cmake/help/latest/command/install.html
[10]: https://user.gurobi.com/download/licenses/free-academic
[11]: https://license.gurobi.com
[12]: https://github.com/microsoft/WSL/issues/5352
[13]: https://github.com/eth-sri/eran/files/3653882/zonotope_example.txt
[14]: https://github.com/eth-sri/deepg#format-of-configuration-file
[15]: https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/CNN_B_CIFAR_ADV.onnx
[16]: https://arxiv.org/abs/2010.11645
[17]: https://files.sri.inf.ethz.ch/eran/data/cifar10_test_b_adv.csv
[18]: https://arxiv.org/abs/2103.06624
[19]: https://raw.githubusercontent.com/eth-sri/eran/master/spatial.png
[20]: https://arxiv.org/abs/2009.09318
[21]: https://www.sri.inf.ethz.ch/publications/mueller2021precise
[22]: https://www.sri.inf.ethz.ch/publications/mller2021neural
[23]: https://arxiv.org/abs/2009.09318
[24]: https://www.sri.inf.ethz.ch/publications/balunovic2019geometric
[25]: https://www.sri.inf.ethz.ch/publications/singh2019krelu
[26]: https://www.sri.inf.ethz.ch/publications/singh2019refinement
[27]: https://www.sri.inf.ethz.ch/publications/singh2019domain
[28]: https://www.sri.inf.ethz.ch/publications/singh2018effective
[29]: https://github.com/eth-sri/diffai
[30]: https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/mnist_relu_3_50.onnx
[31]: https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/mnist_relu_3_100.onnx
[32]: https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/mnist_relu_5_100.onnx
[33]: https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/mnist_relu_6_100.onnx
[34]: https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/mnist_relu_9_100.onnx
[35]: https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/mnist_relu_6_200.onnx
[36]: https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/mnist_relu_9_200.onnx
[37]: https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/ffnnRELU__Point_6_500.o
nnx
[38]: https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/ffnnRELU__PGDK_w_0.1_6_
500.onnx
[39]: https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/ffnnRELU__PGDK_w_0.3_6_
500.onnx
[40]: https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/ffnnSIGMOID__Point_6_50
0.onnx
[41]: https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/ffnnSIGMOID__PGDK_w_0.1
_6_500.onnx
[42]: https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/ffnnSIGMOID__PGDK_w_0.3
_6_500.onnx
[43]: https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/ffnnTANH__Point_6_500.o
nnx
[44]: https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/ffnnTANH__PGDK_w_0.1_6_
500.onnx
[45]: https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/ffnnTANH__PGDK_w_0.3_6_
500.onnx
[46]: https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/mnist_relu_4_1024.onnx
[47]: https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/convSmallRELU__Point.on
nx
[48]: https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/convSmallRELU__PGDK.onn
x
[49]: https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/convSmallRELU__DiffAI.o
nnx
[50]: https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/convMedGRELU__Point.onn
x
[51]: https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/convMedGRELU__PGDK_w_0.
1_6_500.onnx
[52]: https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/convMedGRELU__PGDK_w_0.
3_6_500.onnx
[53]: https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/convMedGSIGMOID__Point.
onnx
[54]: https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/convMedGSIGMOID__PGDK_w
_0.1_6_500.onnx
[55]: https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/convMedGSIGMOID__PGDK_w
_0.3_6_500.onnx
[56]: https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/convMedGTANH__Point.onn
x
[57]: https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/convMedGTANH__PGDK_w_0.
1_6_500.onnx
[58]: https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/convMedGTANH__PGDK_w_0.
3_6_500.onnx
[59]: https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/mnist_conv_maxpool.onnx
[60]: https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/convBigRELU__DiffAI.onn
x
[61]: https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/convSuperRELU__DiffAI.o
nnx
[62]: https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/skip__DiffAI.onnx
[63]: https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/cifar_relu_4_100.onnx
[64]: https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/cifar_relu_6_100.onnx
[65]: https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/cifar_relu_9_200.onnx
[66]: https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/ffnnRELU__Point_6_500.o
nnx
[67]: https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/ffnnRELU__PGDK_w_0.0078
_6_500.onnx
[68]: https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/ffnnRELU__PGDK_w_0.0313
_6_500.onnx
[69]: https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/ffnnSIGMOID__Point_6_50
0.onnx
[70]: https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/ffnnSIGMOID__PGDK_w_0.0
078_6_500.onnx
[71]: https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/ffnnSIGMOID__PGDK_w_0.0
313_6_500.onnx
[72]: https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/ffnnTANH__Point_6_500.o
nnx
[73]: https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/ffnnTANH__PGDK_w_0.0078
_6_500.onnx
[74]: https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/ffnnTANH__PGDK_w_0.0313
_6_500.onnx
[75]: https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/cifar_relu_7_1024.onnx
[76]: https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/convSmallRELU__Point.on
nx
[77]: https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/convSmallRELU__PGDK.onn
x
[78]: https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/convSmallRELU__DiffAI.o
nnx
[79]: https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/convMedGRELU__Point.onn
x
[80]: https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/convMedGRELU__PGDK_w_0.
0078.onnx
[81]: https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/convMedGRELU__PGDK_w_0.
0313.onnx
[82]: https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/convMedGSIGMOID__Point.
onnx
[83]: https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/convMedGSIGMOID__PGDK_w
_0.0078.onnx
[84]: https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/convMedGSIGMOID__PGDK_w
_0.0313.onnx
[85]: https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/convMedGTANH__Point.onn
x
[86]: https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/convMedGTANH__PGDK_w_0.
0078.onnx
[87]: https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/convMedGTANH__PGDK_w_0.
0313.onnx
[88]: https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/cifar_conv_maxpool.onnx
[89]: https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/convBigRELU__DiffAI.onn
x
[90]: https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/ResNetTiny_PGD.onnx
[91]: https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/ResNetTiny_DiffAI.onnx
[92]: https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/ResNet18_PGD.onnx
[93]: https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/ResNet18_DiffAI.onnx
[94]: https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/SkipNet18_DiffAI.onnx
[95]: https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/ResNet34_DiffAI.onnx
[96]: https://github.com/huanzhang12/CertifiedReLURobustness
[97]: https://camo.githubusercontent.com/e9d2bd121e53dea33645868cf4a9960574330a4
8b302804c469603a6a6c7547c/68747470733a2f2f66696c65732e7372692e696e662e6574687a2e
63682f6572616e2f706c6f74732f6d6e6973745f365f3130302e706e67
[98]: https://camo.githubusercontent.com/99e24ff6d451c92afdca79d0efc6c83cc7cd6d7
1047271b7d8062a5e9d964112/68747470733a2f2f66696c65732e7372692e696e662e6574687a2e
63682f6572616e2f706c6f74732f6d6e6973745f365f3530302e706e67
[99]: https://camo.githubusercontent.com/328ce33bb06031274a2d9ea5f5977c37ed65936
90ca03376adb24d1a7e569a44/68747470733a2f2f66696c65732e7372692e696e662e6574687a2e
63682f6572616e2f706c6f74732f6d6e6973745f636f6e76736d616c6c2e706e67
[100]: https://camo.githubusercontent.com/62e770573e2413c0b4ec14bb0369b4e8a0ae8e
1b9ba34ec256af7547c539cc63/68747470733a2f2f66696c65732e7372692e696e662e6574687a2
e63682f6572616e2f706c6f74732f6d6e6973745f7369676d6f69645f74616e682e706e67
[101]: https://camo.githubusercontent.com/26c3c942ee5a9b40a0633cd98069e4f065cf94
29b0bac0f510057fad2975f692/68747470733a2f2f66696c65732e7372692e696e662e6574687a2
e63682f6572616e2f706c6f74732f636966617231305f636f6e76736d616c6c2e706e67
[102]: https://www.sri.inf.ethz.ch/people/mark
[103]: mailto:mark.mueller@inf.ethz.ch
[104]: https://ggndpsngh.github.io/
[105]: mailto:ggnds@illinois.edu
[106]: mailto:gagandeepsi@vmware.com
[107]: https://www.sri.inf.ethz.ch/people/mislav
[108]: mailto:mislav.balunovic@inf.ethz.ch
[109]: mailto:hlebm@ethz.ch
[110]: mailto:gleb.makarchuk@gmail.com
[111]: mailto:anruoss@ethz.ch
[112]: https://fserre.github.io/
[113]: mailto:serref@inf.ethz.ch
[114]: https://www.sri.inf.ethz.ch/people/max
[115]: mailto:mbaader@inf.ethz.ch
[116]: https://www.sri.inf.ethz.ch/people/dana
[117]: mailto:dana.drachsler@inf.ethz.ch
[118]: https://www.sri.inf.ethz.ch/tg.php
[119]: mailto:timon.gehr@inf.ethz.ch
[120]: mailto:adriahof@student.ethz.ch
[121]: mailto:maurerjo@student.ethz.ch
[122]: https://www.mirman.com
[123]: mailto:matt@mirman.com
[124]: mailto:christoph.mueller@inf.ethz.ch
[125]: https://acl.inf.ethz.ch/people/markusp/
[126]: mailto:pueschel@inf.ethz.ch
[127]: https://www.sri.inf.ethz.ch/people/petar
[128]: mailto:petar.tsankov@inf.ethz.ch
[129]: https://www.sri.inf.ethz.ch/vechev.php
[130]: mailto:martin.vechev@inf.ethz.ch
[131]: https://www.sri.inf.ethz.ch/
[132]: https://www.apache.org/licenses/LICENSE-2.0
