# auto_LiRPA: Automatic Linear Relaxation based Perturbation Analysis for Neural Networks

[[Documentation Status]][1] [[Open In Colab]][2] [[Video Introduction]][3] [[BSD license]][4]


## What's New?

* [α,β-CROWN][5] (using `auto_LiRPA` as its core library) is the winner of [VNN-COMP 2025][6] and is
  **ranked top-1** in all [scored benchmarks][7]. (08/2025)
* Bounding of computation graphs containing Jacobian operators now supports more nonlinear operators
  (e.g., `tanh`, `sigmoid`), enabling verification of [continuous-time Lyapunov stability][8].
  (12/2025)
* [α,β-CROWN][9] (using `auto_LiRPA` as its core library) is the winner of [VNN-COMP 2024][10]. Our
  tool is **ranked top-1** in all benchmarks (including 12 [regular track][11] and 9 [extended
  track][12] benchmarks). (08/2024)
* The [INVPROP algorithm][13] allows to compute overapproximationsw of preimages (the set of inputs
  of an NN generating a given output set) and tighten bounds using output constraints. (03/2024)
* Branch-and-bound support for non-ReLU and general nonlinearities ([GenBaB][14]) with optimizable
  bounds (α-CROWN) for new nonlinear functions (sin, cos, GeLU). We achieve significant improvements
  on verifying neural networks with non-ReLU nonlinearities such as Transformers, LSTM, and
  [ML4ACOPF][15]. (09/2023)
* [α,β-CROWN][16] ([alpha-beta-CROWN][17]) (using `auto_LiRPA` as its core library) **won**
  [VNN-COMP 2023][18]. (08/2023)
* Bound computation for higher-order computational graphs to support bounding Jacobian,
  Jacobian-vector products, and [local Lipschitz constants][19]. (11/2022)
* Our neural network verification tool [α,β-CROWN][20] ([alpha-beta-CROWN][21]) (using `auto_LiRPA`
  as its core library) **won** [VNN-COMP 2022][22]. Our library supports the large CIFAR100,
  TinyImageNet and ImageNet models in VNN-COMP 2022. (09/2022)
* Implementation of **general cutting planes** ([GCP-CROWN][23]), support of more activation
  functions and improved performance and scalability. (09/2022)
* Our neural network verification tool [α,β-CROWN][24] ([alpha-beta-CROWN][25]) **won** [VNN-COMP
  2021][26] **with the highest total score**, outperforming 11 SOTA verifiers. α,β-CROWN uses the
  `auto_LiRPA` library as its core bound computation library. (09/2021)
* [Optimized CROWN/LiRPA][27] bound (α-CROWN) for ReLU, **sigmoid**, **tanh**, and **maxpool**
  activation functions, which can significantly outperform regular CROWN bounds. See
  [simple_verification.py][28] for an example. (07/31/2021)
* Handle split constraints for ReLU neurons ([β-CROWN][29]) for complete verifiers. (07/31/2021)
* A memory efficient GPU implementation of backward (CROWN) bounds for convolutional layers.
  (10/31/2020)
* Certified defense models for downscaled ImageNet, TinyImageNet, CIFAR-10, LSTM/Transformer.
  (08/20/2020)
* Adding support to **complex vision models** including DenseNet, ResNeXt and WideResNet.
  (06/30/2020)
* **Loss fusion**, a technique that reduces training cost of tight LiRPA bounds (e.g. CROWN-IBP) to
  the same asymptotic complexity of IBP, making LiRPA based certified defense scalable to large
  datasets (e.g., TinyImageNet, downscaled ImageNet). (06/30/2020)
* **Multi-GPU** support to scale LiRPA based training to large models and datasets. (06/30/2020)
* Initial release. (02/28/2020)

## Introduction

`auto_LiRPA` is a library for automatically deriving and computing bounds with linear relaxation
based perturbation analysis (LiRPA) (e.g. [CROWN][30] and [DeepPoly][31]) for neural networks, which
is a useful tool for formal robustness verification. We generalize existing LiRPA algorithms for
feed-forward neural networks to a graph algorithm on general computational graphs, defined by
PyTorch. Additionally, our implementation is also automatically **differentiable**, allowing
optimizing network parameters to shape the bounds into certain specifications (e.g., certified
defense). You can find [a video ▶️ introduction here][32].

Our library supports the following algorithms:

* Backward mode LiRPA bound propagation ([CROWN][33]/[DeepPoly][34])
* Backward mode LiRPA bound propagation with optimized bounds ([α-CROWN][35])
* Backward mode LiRPA bound propagation with split constraints ([β-CROWN][36] for ReLU, and
  [GenBaB][37] for general nonlinear functions)
* Generalized backward mode LiRPA bound propagation with general cutting plane constraints
  ([GCP-CROWN][38])
* Backward mode LiRPA bound propagation with bounds tightened using output constraints
  ([INVPROP][39])
* Generalized backward mode LiRPA bound propagation for higher-order computational graphs ([Shi et
  al., 2022][40])
* Forward mode LiRPA bound propagation ([Xu et al., 2020][41])
* Forward mode LiRPA bound propagation with optimized bounds (similar to [α-CROWN][42])
* Interval bound propagation ([IBP][43])
* Hybrid approaches, e.g., Forward+Backward, IBP+Backward ([CROWN-IBP][44]), [α,β-CROWN][45]
  ([alpha-beta-CROWN][46])
* MIP/LP formulation of neural networks

Our library allows automatic bound derivation and computation for general computational graphs, in a
similar manner that gradients are obtained in modern deep learning frameworks -- users only define
the computation in a forward pass, and `auto_LiRPA` traverses through the computational graph and
derives bounds for any nodes on the graph. With `auto_LiRPA` we free users from deriving and
implementing LiPRA for most common tasks, and they can simply apply LiPRA as a tool for their own
applications. This is especially useful for users who are not experts of LiRPA and cannot derive
these bounds manually (LiRPA is significantly more complicated than backpropagation).

## Technical Background in 1 Minute

Deep learning frameworks such as PyTorch represent neural networks (NN) as a computational graph,
where each mathematical operation is a node and edges define the flow of computation:


Normally, the inputs of a computation graph (which defines a NN) are data and model weights, and
PyTorch goes through the graph and produces model prediction (a bunch of numbers):


Our `auto_LiRPA` library conducts perturbation analysis on a computational graph, where the input
data and model weights are defined within some user-defined ranges. We get guaranteed output ranges
(bounds):


## Installation

Python 3.11+ and PyTorch 2.0+ are required. It is highly recommended to have a pre-installed PyTorch
that matches your system and our version requirement (see [PyTorch Get Started][47]). Then you can
install `auto_LiRPA` via:

git clone https://github.com/Verified-Intelligence/auto_LiRPA
cd auto_LiRPA
pip install .

If you intend to modify this library, use `pip install -e .` instead.

## Quick Start

First define your computation as a `nn.Module` and wrap it using `auto_LiRPA.BoundedModule()`. Then,
you can call the `compute_bounds` function to obtain certified lower and upper bounds under input
perturbations:

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

# Define computation as a nn.Module.
class MyModel(nn.Module):
    def forward(self, x):
        # Define your computation here.

model = MyModel()
my_input = load_a_batch_of_data()
# Wrap the model with auto_LiRPA.
model = BoundedModule(model, my_input)
# Define perturbation. Here we add Linf perturbation to input data.
ptb = PerturbationLpNorm(norm=np.inf, eps=0.1)
# Make the input a BoundedTensor with the pre-defined perturbation.
my_input = BoundedTensor(my_input, ptb)
# Regular forward propagation using BoundedTensor works as usual.
prediction = model(my_input)
# Compute LiRPA bounds using the backward mode bound propagation (CROWN).
lb, ub = model.compute_bounds(x=(my_input,), method="backward")

Checkout [examples/vision/simple_verification.py][48] for a complete but very basic example.

We also provide a [Google Colab Demo][49] including an example of computing verification bounds for
a 18-layer ResNet model on CIFAR-10 dataset. Once the ResNet model is defined as usual in Pytorch,
obtaining provable output bounds is as easy as obtaining gradients through autodiff. Bounds are
efficiently computed on GPUs.

## More Working Examples

We provide [a wide range of examples][50] of using `auto_LiRPA`:

* [Basic Bound Computation on a Toy Neural Network (simplest example)][51]
* [Basic Bound Computation with **Robustness Verification** of Neural Networks as an example][52]
* [MIP/LP Formulation of Neural Networks][53]
* [Basic **Certified Adversarial Defense** Training][54]
* [Large-scale Certified Defense Training on **ImageNet**][55]
* [Certified Adversarial Defense Training on Sequence Data with **LSTM**][56]
* [Certifiably Robust Language Classifier using **Transformers**][57]
* [Certified Robustness against **Model Weight Perturbations**][58]
* [Bounding **Jacobian** and **local Lipschitz constants**][59]
* [Compute an Overapproximate of Neural Network **Preimage**][60]

`auto_LiRPA` has also been used in the following works:

* [**α,β-CROWN for complete neural network verification**][61]
* [**Fast certified robust training**][62]
* [**Computing local Lipschitz constants**][63]

## Full Documentations

For more documentations, please refer to:

* [Documentation homepage][64]
* [API documentation][65]
* [Adding custom operators][66]
* [Guide][67] for reproducing [our NeurIPS 2020 paper][68]

## Publications

Please kindly cite our papers if you use the `auto_LiRPA` library. Full [BibTeX entries][69] can be
found [here][70].

The general LiRPA based bound propagation algorithm was originally proposed in our paper:

* [Automatic Perturbation Analysis for Scalable Certified Robustness and Beyond][71]. NeurIPS 2020.
  Kaidi Xu*, Zhouxing Shi*, Huan Zhang*, Yihan Wang, Kai-Wei Chang, Minlie Huang, Bhavya Kailkhura,
  Xue Lin, Cho-Jui Hsieh (* Equal contribution)

The `auto_LiRPA` library is further extended to support:

* Optimized bounds (α-CROWN):
  
  [Fast and Complete: Enabling Complete Neural Network Verification with Rapid and Massively
  Parallel Incomplete Verifiers][72]. ICLR 2021. Kaidi Xu*, Huan Zhang*, Shiqi Wang, Yihan Wang,
  Suman Jana, Xue Lin and Cho-Jui Hsieh (* Equal contribution).
* Split constraints (β-CROWN):
  
  [Beta-CROWN: Efficient Bound Propagation with Per-neuron Split Constraints for Complete and
  Incomplete Neural Network Verification][73]. NeurIPS 2021. Shiqi Wang*, Huan Zhang*, Kaidi Xu*,
  Suman Jana, Xue Lin, Cho-Jui Hsieh and Zico Kolter (* Equal contribution).
* General constraints (GCP-CROWN):
  
  [GCP-CROWN: General Cutting Planes for Bound-Propagation-Based Neural Network Verification][74].
  Huan Zhang*, Shiqi Wang*, Kaidi Xu*, Linyi Li, Bo Li, Suman Jana, Cho-Jui Hsieh and Zico Kolter (*
  Equal contribution).
* Higher-order computational graphs (Lipschitz constants and Jacobian):
  
  [Efficiently Computing Local Lipschitz Constants of Neural Networks via Bound Propagation][75].
  NeurIPS 2022. Zhouxing Shi, Yihan Wang, Huan Zhang, Zico Kolter, Cho-Jui Hsieh.
* Branch-and-bound for non-ReLU and general nonlinear functions (GenBaB):
  
  [Neural Network Verification with Branch-and-Bound for General Nonlinearities][76]. TACAS 2025.
  Zhouxing Shi*, Qirui Jin*, Zico Kolter, Suman Jana, Cho-Jui Hsieh, Huan Zhang (* Equal
  contribution).
* Tightening of bounds and preimage computation using the INVPROP algorithm:
  
  [Provably Bounding Neural Network Preimages][77]. NeurIPS 2023. Suhas Kotha*, Christopher Brix*,
  Zico Kolter, Krishnamurthy (Dj) Dvijotham**, Huan Zhang** (* Equal contribution; ** Equal
  advising).

Certified training (verification-aware training by optimizing bounds) using `auto_LiRPA` is improved
with:

* Much shorter warmup schedule and faster training:
  
  [Fast Certified Robust Training with Short Warmup][78]. NeurIPS 2021. Zhouxing Shi*, Yihan Wang*,
  Huan Zhang, Jinfeng Yi and Cho-Jui Hsieh (* Equal contribution).
* Training-time branch-and-bound:
  
  [Certified Training with Branch-and-Bound: A Case Study on Lyapunov-stable Neural Control][79].
  Zhouxing Shi, Cho-Jui Hsieh, and Huan Zhang.

## Developers and Copyright

Team leaders:

* Faculty: Huan Zhang ([huan@huan-zhang.com][80]), UIUC
* Student: Xiangru Zhong ([xiangru4@illinois.edu][81]), UIUC

Current developers (* indicates members of VNN-COMP 2025 team):

* *Duo Zhou ([duozhou2@illinois.edu][82]), UIUC
* *Keyi Shen ([keyis2@illinois.edu][83]), UIUC (graduated, now at Georgia Tech)
* *Hesun Chen ([hesunc2@illinois.edu][84]), UIUC
* *Haoyu Li ([haoyuli5@illinois.edu][85]), UIUC
* *Ruize Gao ([ruizeg2@illinois.edu][86]), UIUC
* *Hao Cheng ([haoc539@illinois.edu][87]), UIUC
* Zhouxing Shi ([zhouxingshichn@gmail.com][88]), UCLA/UC Riverside
* Lei Huang ([leih5@illinois.edu][89]), UIUC
* Taobo Liao ([taobol2@illinois.edu][90]), UIUC
* Jorge Chavez ([jorgejc2@illinois.edu][91]), UIUC

Past developers:

* Hongji Xu ([hx84@duke.edu][92]), Duke University (intern with Prof. Huan Zhang)
* Christopher Brix ([brix@cs.rwth-aachen.de][93]), RWTH Aachen University
* Hao Chen ([haoc8@illinois.edu][94]), UIUC
* Keyu Lu ([keyulu2@illinois.edu][95]), UIUC
* Kaidi Xu ([kx46@drexel.edu][96]), Drexel University
* Sanil Chawla ([schawla7@illinois.edu][97]), UIUC
* Linyi Li ([linyi2@illinois.edu][98]), UIUC
* Zhuolin Yang ([zhuolin5@illinois.edu][99]), UIUC
* Zhuowen Yuan ([realzhuowen@gmail.com][100]), UIUC
* Qirui Jin ([qiruijin@umich.edu][101]), University of Michigan
* Shiqi Wang ([sw3215@columbia.edu][102]), Columbia University
* Yihan Wang ([yihanwang@ucla.edu][103]), UCLA
* Jinqi (Kathryn) Chen ([jinqic@cs.cmu.edu][104]), CMU

`auto_LiRPA` is currently supported in part by the National Science Foundation (NSF; award 2331967,
2525287), the AI2050 program at Schmidt Science, the Virtual Institute for Scientific Software
(VISS) at Georgia Tech, the University Research Program at Toyota Research Institute (TRI), and a
Mathworks research award.

We thank the [commits][105] and [pull requests][106] from community contributors.

Our library is released under the BSD 3-Clause license.

[1]: https://auto-lirpa.readthedocs.io/en/latest/?badge=latest
[2]: http://PaperCode.cc/AutoLiRPA-Demo
[3]: http://PaperCode.cc/AutoLiRPA-Video
[4]: https://opensource.org/licenses/BSD-3-Clause
[5]: https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
[6]: https://sites.google.com/view/vnn2025
[7]: https://github.com/VNN-COMP/vnncomp2025_results/blob/main/SCORING-SMALL-TOL/latex/main.pdf
[8]: https://github.com/Verified-Intelligence/Two-Stage_Neural_Controller_Training
[9]: https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
[10]: https://sites.google.com/view/vnn2024
[11]: https://github.com/ChristopherBrix/vnncomp2024_results/blob/main/SCORING/latex/results_regular
_track.pdf
[12]: https://github.com/ChristopherBrix/vnncomp2024_results/blob/main/SCORING/latex/results_extende
d_track.pdf
[13]: https://arxiv.org/pdf/2302.01404.pdf
[14]: https://arxiv.org/pdf/2405.21063
[15]: https://github.com/AI4OPT/ml4acopf_benchmark
[16]: https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
[17]: https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
[18]: https://sites.google.com/view/vnn2023
[19]: https://arxiv.org/abs/2210.07394
[20]: https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
[21]: https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
[22]: https://sites.google.com/view/vnn2022
[23]: https://arxiv.org/pdf/2208.05740.pdf
[24]: https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
[25]: https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
[26]: https://sites.google.com/view/vnn2021
[27]: https://arxiv.org/pdf/2011.13824.pdf
[28]: /Verified-Intelligence/auto_LiRPA/blob/master/examples/vision/simple_verification.py#L59
[29]: https://arxiv.org/pdf/2103.06624.pdf
[30]: https://arxiv.org/pdf/1811.00866.pdf
[31]: https://files.sri.inf.ethz.ch/website/papers/DeepPoly.pdf
[32]: http://PaperCode.cc/AutoLiRPA-Video
[33]: https://arxiv.org/pdf/1811.00866.pdf
[34]: https://files.sri.inf.ethz.ch/website/papers/DeepPoly.pdf
[35]: https://arxiv.org/pdf/2011.13824.pdf
[36]: https://arxiv.org/pdf/2103.06624.pdf
[37]: https://arxiv.org/pdf/2405.21063
[38]: https://arxiv.org/pdf/2208.05740.pdf
[39]: https://arxiv.org/pdf/2302.01404.pdf
[40]: https://arxiv.org/abs/2210.07394
[41]: https://arxiv.org/pdf/2002.12920
[42]: https://arxiv.org/pdf/2011.13824.pdf
[43]: https://arxiv.org/pdf/1810.12715.pdf
[44]: https://arxiv.org/pdf/1906.06316.pdf
[45]: https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
[46]: https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
[47]: https://pytorch.org/get-started
[48]: /Verified-Intelligence/auto_LiRPA/blob/master/examples/vision/simple_verification.py
[49]: http://PaperCode.cc/AutoLiRPA-Demo
[50]: /Verified-Intelligence/auto_LiRPA/blob/master/doc/src/examples.md
[51]: /Verified-Intelligence/auto_LiRPA/blob/master/examples/simple/toy.py
[52]: /Verified-Intelligence/auto_LiRPA/blob/master/doc/src/examples.md#basic-bound-computation-and-
robustness-verification-of-neural-networks
[53]: /Verified-Intelligence/auto_LiRPA/blob/master/examples/simple/mip_lp_solver.py
[54]: /Verified-Intelligence/auto_LiRPA/blob/master/doc/src/examples.md#basic-certified-adversarial-
defense-training
[55]: /Verified-Intelligence/auto_LiRPA/blob/master/doc/src/examples.md#certified-adversarial-defens
e-on-downscaled-imagenet-and-tinyimagenet-with-loss-fusion
[56]: /Verified-Intelligence/auto_LiRPA/blob/master/doc/src/examples.md#certified-adversarial-defens
e-training-for-lstm-on-mnist
[57]: /Verified-Intelligence/auto_LiRPA/blob/master/doc/src/examples.md#certifiably-robust-language-
classifier-with-transformer-and-lstm
[58]: /Verified-Intelligence/auto_LiRPA/blob/master/doc/src/examples.md#certified-robustness-against
-model-weight-perturbations-and-certified-defense
[59]: /Verified-Intelligence/auto_LiRPA/blob/master/examples/vision/jacobian.py
[60]: /Verified-Intelligence/auto_LiRPA/blob/master/examples/simple/invprop.py
[61]: https://github.com/Verified-Intelligence/alpha-beta-CROWN
[62]: https://github.com/shizhouxing/Fast-Certified-Robust-Training
[63]: https://github.com/shizhouxing/Local-Lipschitz-Constants
[64]: https://auto-lirpa.readthedocs.io
[65]: https://auto-lirpa.readthedocs.io/en/latest/api.html
[66]: https://auto-lirpa.readthedocs.io/en/latest/custom_op.html
[67]: https://auto-lirpa.readthedocs.io/en/latest/paper.html
[68]: https://arxiv.org/abs/2002.12920
[69]: /Verified-Intelligence/auto_LiRPA/blob/master/doc/src/examples.md#bibtex-entries
[70]: /Verified-Intelligence/auto_LiRPA/blob/master/doc/src/examples.md#bibtex-entries
[71]: https://arxiv.org/pdf/2002.12920
[72]: https://arxiv.org/pdf/2011.13824.pdf
[73]: https://arxiv.org/pdf/2103.06624.pdf
[74]: https://arxiv.org/abs/2208.05740
[75]: https://arxiv.org/abs/2210.07394
[76]: https://arxiv.org/pdf/2405.21063
[77]: https://arxiv.org/pdf/2302.01404.pdf
[78]: https://arxiv.org/pdf/2103.17268.pdf
[79]: https://arxiv.org/abs/2411.18235
[80]: mailto:huan@huan-zhang.com
[81]: mailto:xiangru4@illinois.edu
[82]: mailto:duozhou2@illinois.edu
[83]: mailto:keyis2@illinois.edu
[84]: mailto:hesunc2@illinois.edu
[85]: mailto:haoyuli5@illinois.edu
[86]: mailto:ruizeg2@illinois.edu
[87]: mailto:haoc539@illinois.edu
[88]: mailto:zhouxingshichn@gmail.com
[89]: mailto:leih5@illinois.edu
[90]: mailto:taobol2@illinois.edu
[91]: mailto:jorgejc2@illinois.edu
[92]: mailto:hx84@duke.edu
[93]: mailto:brix@cs.rwth-aachen.de
[94]: mailto:haoc8@illinois.edu
[95]: mailto:keyulu2@illinois.edu
[96]: mailto:kx46@drexel.edu
[97]: mailto:schawla7@illinois.edu
[98]: mailto:linyi2@illinois.edu
[99]: mailto:zhuolin5@illinois.edu
[100]: mailto:realzhuowen@gmail.com
[101]: mailto:qiruijin@umich.edu
[102]: mailto:sw3215@columbia.edu
[103]: mailto:yihanwang@ucla.edu
[104]: mailto:jinqic@cs.cmu.edu
[105]: https://github.com/Verified-Intelligence/auto_LiRPA/commits
[106]: https://github.com/Verified-Intelligence/auto_LiRPA/pulls
