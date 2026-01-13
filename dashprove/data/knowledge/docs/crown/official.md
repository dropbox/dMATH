# α,β-CROWN (alpha-beta-CROWN): A Fast and Scalable Neural Network Verifier with Efficient Bound
# Propagation


α,β-CROWN (alpha-beta-CROWN) is a neural network verifier based on an efficient linear bound
propagation framework and branch and bound. It can be accelerated efficiently on **GPUs** and can
scale to relatively large convolutional networks (e.g., millions of parameters). It also supports a
wide range of neural network architectures (e.g., **CNN**, **ResNet**, and various activation
functions), thanks to the versatile **[auto_LiRPA][1] library developed by us**. α,β-CROWN can
provide provable robustness guarantees against adversarial attacks and can also verify other general
properties of neural networks, such as [Lyapunov stability][2] in control.

α,β-CROWN is the **winning verifier** in [VNN-COMP 2021][3], [VNN-COMP 2022][4], [VNN-COMP 2023][5],
[VNN-COMP 2024][6], and [VNN-COMP 2025][7] (International Verification of Neural Networks
Competition) with the highest total score, outperforming many other neural network verifiers on a
wide range of benchmarks over 5 years. Details of competition results can be found in [VNN-COMP 2021
slides][8], [report][9], [VNN-COMP 2022 report][10], [VNN-COMP 2023 slides][11] and [report][12],
and [VNN-COMP 2024 slides][13] and [report][14], [VNN-COMP 2025 slides][15].

The α,β-CROWN team is created and led by Prof. [Huan Zhang][16] at UIUC with contributions from
multiple institutions. See the **list of contributors** [below][17]. α,β-CROWN combines our efforts
in neural network verification in a series of papers building up the bound propagation framework
since 2018. See [Publications][18] below.

## News (2025 - )

* α,β-CROWN now provides a **new Python API** that allows users to run the verifier programmatically
  in Python **without** manually exporting ONNX models, writing VNNLIB specifications, or preparing
  config files. A good set of **default configuration automatically handles diverse verification
  scenarios**, while all options remain fully customizable through the API. (**See [API
  documentation][19]**).
* α,β-CROWN is the winner of [VNN-COMP 2025][20] and is **ranked top-1** in all [scored
  benchmarks][21]. (08/2025)
* Bounding of computation graphs containing Jacobian operators now supports more nonlinear operators
  (e.g., `tanh`, `sigmoid`), enabling verification of [continuous-time Lyapunov stability][22].
  (12/2025)
* Clip-and-Verify ([Zhou et al., NeurIPS 2025][23]) efficiently handles linear constraints and can
  significantly reduce the number of subproblems handled during BaB. It consistently tightens bounds
  across multiple benchmarks. (12/2025)

## Supported Features


Our verifier consists of the following core algorithms:

* **CROWN** ([Zhang et al., 2018][24]): the basic linear bound propagation framework for neural
  networks.
* **auto_LiRPA** ([Xu et al. 2020][25]): linear bound propagation for general computational graphs.
* **α-CROWN** ([Xu et al., 2021][26]): incomplete verification with gradient optimized bound
  propagation.
* **β-CROWN** ([Wang et al., 2021][27]): complete verification with bound propagation and branch and
  bound for ReLU networks.
* **GenBaB** ([Shi et al., 2024][28]): Branch and bound for general nonlinear functions.
* **GCP-CROWN** ([Zhang et al., 2022][29]): CROWN-like bound propagation with general cutting plane
  constraints.
* **BaB-Attack** ([Zhang et al., 2022][30]): Branch and bound based adversarial attack for tackling
  hard instances.
* **MIP** ([Tjeng et al., 2017][31]): mixed integer programming (slow but can be useful on small
  models).
* **INVPROP** ([Kotha et al., 2023][32]): tightens bounds with constraints on model outputs, and
  computes provable preimages for neural networks.
* **BICCOS** ([Zhou et al., 2024][33]): an effective cutting plane generation method outperforming
  the MIP-based cuts in GCP-CROWN.

The bound propagation engine in α,β-CROWN is implemented as a separate library, **[auto_LiRPA][34]
([Xu et al. 2020][35])**, for computing symbolic bounds for general computational graphs. We support
these neural network architectures:

* Layers: fully connected (FC), convolutional (CNN), pooling (average pool and max pool), transposed
  convolution
* Activation functions or nonlinear functions: ReLU, sigmoid, tanh, arctan, sin, cos, tan, gelu,
  pow, multiplication and self-attention
* Residual connections, Transformers, LSTMs, and other irregular graphs

We support the following verification specifications:

* Lp norm perturbation (p=1,2,infinity, as often used in robustness verification)
* VNNLIB format input (at most two layers of AND/OR clause, as used in VNN-COMP)
* Any linear specifications on neural network output (which can be added as a linear layer)

We provide many example configurations in [`complete_verifier/exp_configs`][36] directory to start
with:

* MNIST: MLP and CNN models (small models to help you get started)
* CIFAR-10, CIFAR-100, TinyImageNet: CNN and ResNet models with high dimensional inputs
* ACASXu, NN4sys, ML4ACOPF and other low input dimension models

And more examples in other repositories:

* Stability verification of NN controllers for discrete-time systems:
  [Verified-Intelligence/Lyapunov_Stable_NN_Controllers][37] and continuous-time systems:
  [Verified-Intelligence/Two-Stage_Neural_Controller_Training][38].
* Branch-and-bound for models with non-ReLU nonlinearities and high dimensional inputs: [GenBaB][39]

See the [Guide on Algorithm Selection][40] to find the most suitable example to get started.

## Installation and Setup

α,β-CROWN is tested on Python 3.11 and PyTorch 2.8.0 (recent versions may also work). It can be
installed easily into a conda environment. If you don't have conda, you can install [miniconda][41].

Clone our verifier including the [auto_LiRPA][42] submodule:

git clone --recursive https://github.com/Verified-Intelligence/alpha-beta-CROWN.git

Setup the conda environment from [`environment.yaml`][43] with pinned dependencies versions
(CUDA>=12.8 is required):

# Remove the old environment, if necessary.
conda deactivate; conda env remove --name alpha-beta-crown
# install all dependents into the alpha-beta-crown environment
conda env create -f complete_verifier/environment.yaml --name alpha-beta-crown
# activate the environment
conda activate alpha-beta-crown

Alternatively, you may use `pip` (if you want to add α,β-CROWN to your existing environment, or if
your system is not compatible with [`environment.yaml`][44]). It is highly recommended to have a
pre-installed PyTorch that matches your system and our version requirement (see [PyTorch Get
Started][45]). Then, you can run:

(cd auto_LiRPA; pip install -e .)
pip install -r complete_verifier/requirements.txt

Unless you use MIP-based verification algorithms, a Gurobi license is *not needed* (in most use
cases). If you want to use MIP-based verification algorithms (which are feasible only for small
models), you need to install a Gurobi license with the `grbgetkey` command. If you don't have access
to a license, by default, the above installation procedure includes a free and restricted license,
which is actually sufficient for many relatively small NNs. If you use the GCP-CROWN verifier, an
installation of IBM CPlex solver is required. Instructions to install the CPlex solver can be found
in the [VNN-COMP benchmark instructions][46].

If you want to run α,β-CROWN verifier on the VNN-COMP benchmarks (e.g., to make a comparison to a
new verifier), you can follow [this guide][47].

## Instructions

The verifier can be invoked through the **[new Python API][48]** or through the command-line
interface with configuration files. Checkout the [API documentation][49] for API usage. For the
command-line interface, we provide a unified front-end for the verifier, `abcrown.py`. All
parameters for the verifier are defined in a `yaml` config file. For example, to run robustness
verification on a CIFAR-10 ResNet network, you just run:

conda activate alpha-beta-crown  # activate the conda environment
cd complete_verifier
python abcrown.py --config exp_configs/tutorial_examples/cifar_resnet_2b.yaml

You can find explanations for the most useful parameters in [this example config file][50]. For
detailed usage and tutorial examples, please see the [Usage Documentation][51]. We also provide a
large range of examples in the [`complete_verifier/exp_configs`][52] folder.

## Publications

If you use our verifier in your work, **please kindly cite our papers**:

* **CROWN** ([Zhang et al., 2018][53]), **auto_LiRPA** ([Xu et al., 2020][54]), **α-CROWN** ([Xu et
  al., 2021][55]), **β-CROWN** ([Wang et al., 2021][56]), **GenBaB** ([Shi et al. 2024][57]),
  **GCP-CROWN** ([Zhang et al., 2022][58]), and **BICCOS** ([Zhou et al., NeurIPS 2024][59]).
* **[Kotha et al., 2023][60]** if you use constraints on the outputs of neural networks.
* **[Salman et al., 2019][61]**, if your work involves the convex relaxation of the NN verification.
* **[Zhang et al. 2022][62]**, if you use our branch-and-bound based adversarial attack (falsifier).
  We provide bibtex entries at the end of this section.

α,β-CROWN represents our continued efforts on neural network verification:

* **CROWN** ([Zhang et al. NeurIPS 2018][63]) is a very efficient bound propagation based
  verification algorithm. CROWN propagates a linear inequality backward through the network and
  utilizes linear bounds to relax activation functions.
* The **"convex relaxation barrier"** ([Salman et al., NeurIPS 2019][64]) paper concludes that
  optimizing the ReLU relaxation allows CROWN (referred to as a "greedy" primal space solver) to
  achieve the same solution as linear programming (LP) based verifiers.
* **auto_LiRPA** ([Xu et al., NeurIPS 2020][65]) is a generalization of CROWN on general
  computational graphs and we also provide an efficient GPU implementation, the [auto_LiRPA][66]
  library.
* **α-CROWN** (sometimes referred to as optimized CROWN or optimized LiRPA) is used in the
  Fast-and-Complete verifier ([Xu et al., ICLR 2021][67]), which jointly optimizes intermediate
  layer bounds and final layer bounds in CROWN via variable α. α-CROWN typically has greater power
  than LP since LP cannot cheaply tighten intermediate layer bounds.
* **β-CROWN** ([Wang et al., NeurIPS 2021][68]) incorporates ReLU split constraints in branch and
  bound (BaB) into the CROWN bound propagation procedure via an additional optimizable parameter β.
  The combination of efficient and GPU-accelerated bound propagation with branch and bound produces
  a powerful and scalable neural network verifier.
* **BaB-Attack** ([Zhang et al., ICML 2022][69]) is a strong falsifier (adversarial attack) based on
  branch and bound, which can find adversarial examples for hard instances where gradient or
  input-space-search based methods cannot succeed.
* **GCP-CROWN** ([Zhang et al., NeurIPS 2022][70]) enables the use of general cutting planes methods
  for neural network verification in a GPU-accelerated and very efficient bound propagation
  framework. Cutting planes can significantly strengthen bound tightness.
* **INVPROP** ([Kotha et al., NeurIPS 2023][71]) handles constraints on the outputs of neural
  networks which enables tight and provable bounds on the preimage of a neural network. We
  demonstrated several applications, including OOD detection, backward reachability analysis for
  NN-controlled systems, and tightening bounds for robustness verification.
* **BICCOS** ([Zhou et al., NeurIPS 2024][72]) generates effective cutting planes during
  branch-and-bound to tighten verification bounds. The cutting plane generation process is efficient
  and scalable and does not require a MIP solver.
* **GenBaB** ([Shi et al., TACAS 2025][73]) enables branch-and-bound based verification for general
  nonlinear functions, achieving significant improvements on verifying neural networks with non-ReLU
  nonlinearties (such as Transformers), and enabling new applications that contain complicated
  nonlinear functions on the output of neural networks, such as [ML for AC Optimal Power Flow][74].
* **Clip-and-Verify** ([Zhou et al., NeurIPS 2025][75]) efficiently handles linear constraints and
  can significantly reduce the number of subproblems handled during BaB. It consistently tightens
  bounds across multiple benchmarks, significantly accelerating challenging verification tasks from
  research on [provably stable neural network control systems][76].
`@article{zhang2018efficient,
  title={Efficient Neural Network Robustness Certification with General Activation Functions},
  author={Zhang, Huan and Weng, Tsui-Wei and Chen, Pin-Yu and Hsieh, Cho-Jui and Daniel, Luca},
  journal={Advances in Neural Information Processing Systems},
  volume={31},
  pages={4939--4948},
  year={2018},
  url={https://arxiv.org/pdf/1811.00866.pdf}
}

@article{xu2020automatic,
  title={Automatic perturbation analysis for scalable certified robustness and beyond},
  author={Xu, Kaidi and Shi, Zhouxing and Zhang, Huan and Wang, Yihan and Chang, Kai-Wei and Huang, 
Minlie and Kailkhura, Bhavya and Lin, Xue and Hsieh, Cho-Jui},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}

@article{salman2019convex,
  title={A Convex Relaxation Barrier to Tight Robustness Verification of Neural Networks},
  author={Salman, Hadi and Yang, Greg and Zhang, Huan and Hsieh, Cho-Jui and Zhang, Pengchuan},
  journal={Advances in Neural Information Processing Systems},
  volume={32},
  pages={9835--9846},
  year={2019}
}

@inproceedings{xu2021fast,
    title={{Fast and Complete}: Enabling Complete Neural Network Verification with Rapid and Massive
ly Parallel Incomplete Verifiers},
    author={Kaidi Xu and Huan Zhang and Shiqi Wang and Yihan Wang and Suman Jana and Xue Lin and Cho
-Jui Hsieh},
    booktitle={International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=nVZtXBI6LNn}
}

@article{wang2021beta,
  title={{Beta-CROWN}: Efficient bound propagation with per-neuron split constraints for complete an
d incomplete neural network verification},
  author={Wang, Shiqi and Zhang, Huan and Xu, Kaidi and Lin, Xue and Jana, Suman and Hsieh, Cho-Jui 
and Kolter, J Zico},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}

@InProceedings{zhang22babattack,
  title =        {A Branch and Bound Framework for Stronger Adversarial Attacks of {R}e{LU} Networks
},
  author =       {Zhang, Huan and Wang, Shiqi and Xu, Kaidi and Wang, Yihan and Jana, Suman and Hsie
h, Cho-Jui and Kolter, Zico},
  booktitle =    {Proceedings of the 39th International Conference on Machine Learning},
  volume =       {162},
  pages =        {26591--26604},
  year =         {2022},
}

@article{zhang2022general,
  title={General Cutting Planes for Bound-Propagation-Based Neural Network Verification},
  author={Zhang, Huan and Wang, Shiqi and Xu, Kaidi and Li, Linyi and Li, Bo and Jana, Suman and Hsi
eh, Cho-Jui and Kolter, J Zico},
  journal={Advances in Neural Information Processing Systems},
  year={2022}
}

@inproceedings{kotha2023provably,
 author = {Kotha, Suhas and Brix, Christopher and Kolter, J. Zico and Dvijotham, Krishnamurthy and Z
hang, Huan},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Oh and T. Neumann and A. Globerson and K. Saenko and M. Hardt and S. Levine},
 pages = {80270--80290},
 publisher = {Curran Associates, Inc.},
 title = {Provably Bounding Neural Network Preimages},
 url = {https://proceedings.neurips.cc/paper_files/paper/2023/file/fe061ec0ae03c5cf5b5323a2b9121bfd-
Paper-Conference.pdf},
 volume = {36},
 year = {2023}
}

@inproceedings{zhou2024scalable,
  title={Scalable Neural Network Verification with Branch-and-bound Inferred Cutting Planes},
  author={Zhou, Duo and Brix, Christopher and Hanasusanto, Grani A and Zhang, Huan},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024}
}

@inproceedings{shi2024genbab,
  title={Neural Network Verification with Branch-and-Bound for General Nonlinearities},
  author={Shi, Zhouxing and Jin, Qirui and Kolter, Zico and Jana, Suman and Hsieh, Cho-Jui and Zhang
, Huan},
  booktitle={International Conference on Tools and Algorithms for the Construction and Analysis of S
ystems},
  year={2025}
}

@inproceedings{zhou2025clip,
  title={Clip-and-Verify: Linear Constraint-Driven Domain Clipping for Accelerating Neural Network V
erification},
  author={Zhou, Duo and Chavez, Jorge and Chen, Hesun and Hanasusanto, Grani A and Zhang, Huan},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025}
}
`

## Developers and Copyright

Team leaders:

* Faculty: Huan Zhang ([huan@huan-zhang.com][77]), UIUC
* Student: Xiangru Zhong ([xiangru4@illinois.edu][78]), UIUC

Current developers (* indicates members of VNN-COMP 2025 team):

* *Duo Zhou ([duozhou2@illinois.edu][79]), UIUC
* *Keyi Shen ([keyis2@illinois.edu][80]), UIUC (graduated, now at Georgia Tech)
* *Hesun Chen ([hesunc2@illinois.edu][81]), UIUC
* *Haoyu Li ([haoyuli5@illinois.edu][82]), UIUC
* *Ruize Gao ([ruizeg2@illinois.edu][83]), UIUC
* *Hao Cheng ([haoc539@illinois.edu][84]), UIUC
* Zhouxing Shi ([zhouxingshichn@gmail.com][85]), UCLA/UC Riverside
* Lei Huang ([leih5@illinois.edu][86]), UIUC
* Taobo Liao ([taobol2@illinois.edu][87]), UIUC
* Jorge Chavez ([jorgejc2@illinois.edu][88]), UIUC

Past developers:

* Hongji Xu ([hx84@duke.edu][89]), Duke University (intern with Prof. Huan Zhang)
* Christopher Brix ([brix@cs.rwth-aachen.de][90]), RWTH Aachen University
* Hao Chen ([haoc8@illinois.edu][91]), UIUC
* Keyu Lu ([keyulu2@illinois.edu][92]), UIUC
* Kaidi Xu ([kx46@drexel.edu][93]), Drexel University
* Sanil Chawla ([schawla7@illinois.edu][94]), UIUC
* Linyi Li ([linyi2@illinois.edu][95]), UIUC
* Zhuolin Yang ([zhuolin5@illinois.edu][96]), UIUC
* Zhuowen Yuan ([realzhuowen@gmail.com][97]), UIUC
* Qirui Jin ([qiruijin@umich.edu][98]), University of Michigan
* Shiqi Wang ([sw3215@columbia.edu][99]), Columbia University
* Yihan Wang ([yihanwang@ucla.edu][100]), UCLA
* Jinqi (Kathryn) Chen ([jinqic@cs.cmu.edu][101]), CMU

α,β-CROWN is currently supported in part by the National Science Foundation (NSF; award 2331967,
2525287), the AI2050 program at Schmidt Science, the Virtual Institute for Scientific Software
(VISS) at Georgia Tech, the University Research Program at Toyota Research Institute (TRI), and a
Mathworks research award.

The team acknowledges the financial and advisory support (2021 - 2023) from Prof. Zico Kolter
([zkolter@cs.cmu.edu][102]), Prof. Cho-Jui Hsieh ([chohsieh@cs.ucla.edu][103]), Prof. Suman Jana
([suman@cs.columbia.edu][104]), Prof. Bo Li ([lbo@illinois.edu][105]), and Prof. Xue Lin
([xue.lin@northeastern.edu][106]) during the years 2021 - 2023.

Our library is released under the BSD 3-Clause license. A copy of the license is included
[here][107].

[1]: http://github.com/Verified-Intelligence/auto_LiRPA
[2]: https://arxiv.org/pdf/2404.07956
[3]: https://sites.google.com/view/vnn2021
[4]: https://sites.google.com/view/vnn2022
[5]: https://sites.google.com/view/vnn2023
[6]: https://sites.google.com/view/vnn2024
[7]: https://sites.google.com/view/vnn2025
[8]: https://docs.google.com/presentation/d/1oM3NqqU03EUqgQVc3bGK2ENgHa57u-W6Q63Vflkv000/edit#slide=
id.ge4496ad360_14_21
[9]: https://arxiv.org/abs/2109.00498
[10]: https://arxiv.org/pdf/2212.10376.pdf
[11]: https://github.com/ChristopherBrix/vnncomp2023_results/blob/main/SCORING/slides.pdf
[12]: https://arxiv.org/abs/2312.16760
[13]: https://docs.google.com/presentation/d/1RvZWeAdTfRC3bNtCqt84O6IIPoJBnF4jnsEvhTTxsPE/edit
[14]: https://www.arxiv.org/pdf/2412.19985
[15]: https://docs.google.com/presentation/d/1ep-hGGotgWQF6SA0JIpQ6nFqs2lXoyuLMM-bORzNvrQ/edit?slide
=id.p#slide=id.p
[16]: https://huan-zhang.com/
[17]: #developers-and-copyright
[18]: #publications
[19]: /Verified-Intelligence/alpha-beta-CROWN/blob/main/complete_verifier/docs/abcrown_api.md
[20]: https://sites.google.com/view/vnn2025
[21]: https://github.com/VNN-COMP/vnncomp2025_results/blob/main/SCORING-SMALL-TOL/latex/main.pdf
[22]: https://github.com/Verified-Intelligence/Two-Stage_Neural_Controller_Training
[23]: https://openreview.net/pdf?id=HuSSR12Yot
[24]: https://arxiv.org/pdf/1811.00866.pdf
[25]: https://arxiv.org/pdf/2002.12920.pdf
[26]: https://arxiv.org/pdf/2011.13824.pdf
[27]: https://arxiv.org/pdf/2103.06624.pdf
[28]: https://arxiv.org/pdf/2405.21063.pdf
[29]: https://arxiv.org/pdf/2208.05740.pdf
[30]: https://proceedings.mlr.press/v162/zhang22ae/zhang22ae.pdf
[31]: https://arxiv.org/pdf/1711.07356.pdf
[32]: https://arxiv.org/pdf/2302.01404.pdf
[33]: https://openreview.net/pdf?id=FwhM1Zpyft
[34]: https://github.com/Verified-Intelligence/auto_LiRPA
[35]: https://arxiv.org/pdf/2002.12920.pdf
[36]: /Verified-Intelligence/alpha-beta-CROWN/blob/main/complete_verifier/exp_configs
[37]: https://github.com/Verified-Intelligence/Lyapunov_Stable_NN_Controllers
[38]: https://github.com/Verified-Intelligence/Two-Stage_Neural_Controller_Training
[39]: https://huggingface.co/datasets/zhouxingshi/GenBaB
[40]: /Verified-Intelligence/alpha-beta-CROWN/blob/main/complete_verifier/docs/abcrown_usage.md#guid
e-on-algorithm-selection
[41]: https://docs.conda.io/en/latest/miniconda.html
[42]: https://github.com/Verified-Intelligence/auto_LiRPA
[43]: /Verified-Intelligence/alpha-beta-CROWN/blob/main/complete_verifier/environment.yaml
[44]: /Verified-Intelligence/alpha-beta-CROWN/blob/main/complete_verifier/environment.yaml
[45]: https://pytorch.org/get-started
[46]: /Verified-Intelligence/alpha-beta-CROWN/blob/main/complete_verifier/docs/vnn_comp.md#installat
ion
[47]: /Verified-Intelligence/alpha-beta-CROWN/blob/main/complete_verifier/docs/vnn_comp.md
[48]: /Verified-Intelligence/alpha-beta-CROWN/blob/main/complete_verifier/docs/abcrown_api.md
[49]: /Verified-Intelligence/alpha-beta-CROWN/blob/main/complete_verifier/docs/abcrown_api.md
[50]: /Verified-Intelligence/alpha-beta-CROWN/blob/main/complete_verifier/exp_configs/tutorial_examp
les/cifar_resnet_2b.yaml
[51]: /Verified-Intelligence/alpha-beta-CROWN/blob/main/complete_verifier/docs/abcrown_usage.md
[52]: /Verified-Intelligence/alpha-beta-CROWN/blob/main/complete_verifier/exp_configs
[53]: https://arxiv.org/pdf/1811.00866.pdf
[54]: https://arxiv.org/pdf/2002.12920.pdf
[55]: https://arxiv.org/pdf/2011.13824.pdf
[56]: https://arxiv.org/pdf/2103.06624.pdf
[57]: https://arxiv.org/pdf/2405.21063.pdf
[58]: https://arxiv.org/pdf/2208.05740.pdf
[59]: https://openreview.net/pdf?id=FwhM1Zpyft
[60]: https://arxiv.org/pdf/2302.01404.pdf
[61]: https://arxiv.org/pdf/1902.08722
[62]: https://proceedings.mlr.press/v162/zhang22ae/zhang22ae.pdf
[63]: https://arxiv.org/pdf/1811.00866.pdf
[64]: https://arxiv.org/pdf/1902.08722
[65]: https://arxiv.org/pdf/2002.12920.pdf
[66]: https://github.com/Verified-Intelligence/auto_LiRPA
[67]: https://arxiv.org/pdf/2011.13824.pdf
[68]: https://arxiv.org/pdf/2103.06624.pdf
[69]: https://proceedings.mlr.press/v162/zhang22ae/zhang22ae.pdf
[70]: https://arxiv.org/pdf/2208.05740.pdf
[71]: https://arxiv.org/pdf/2302.01404.pdf
[72]: https://openreview.net/pdf?id=FwhM1Zpyft
[73]: https://arxiv.org/pdf/2405.21063.pdf
[74]: https://github.com/AI4OPT/ml4acopf_benchmark
[75]: https://openreview.net/pdf?id=HuSSR12Yot
[76]: https://github.com/Verified-Intelligence/Two-Stage_Neural_Controller_Training
[77]: mailto:huan@huan-zhang.com
[78]: mailto:xiangru4@illinois.edu
[79]: mailto:duozhou2@illinois.edu
[80]: mailto:keyis2@illinois.edu
[81]: mailto:hesunc2@illinois.edu
[82]: mailto:haoyuli5@illinois.edu
[83]: mailto:ruizeg2@illinois.edu
[84]: mailto:haoc539@illinois.edu
[85]: mailto:zhouxingshichn@gmail.com
[86]: mailto:leih5@illinois.edu
[87]: mailto:taobol2@illinois.edu
[88]: mailto:jorgejc2@illinois.edu
[89]: mailto:hx84@duke.edu
[90]: mailto:brix@cs.rwth-aachen.de
[91]: mailto:haoc8@illinois.edu
[92]: mailto:keyulu2@illinois.edu
[93]: mailto:kx46@drexel.edu
[94]: mailto:schawla7@illinois.edu
[95]: mailto:linyi2@illinois.edu
[96]: mailto:zhuolin5@illinois.edu
[97]: mailto:realzhuowen@gmail.com
[98]: mailto:qiruijin@umich.edu
[99]: mailto:sw3215@columbia.edu
[100]: mailto:yihanwang@ucla.edu
[101]: mailto:jinqic@cs.cmu.edu
[102]: mailto:zkolter@cs.cmu.edu
[103]: mailto:chohsieh@cs.ucla.edu
[104]: mailto:suman@cs.columbia.edu
[105]: mailto:lbo@illinois.edu
[106]: mailto:xue.lin@northeastern.edu
[107]: /Verified-Intelligence/alpha-beta-CROWN/blob/main/LICENSE
