# Documentation for [auto_LiRPA][1][¶][2]


## Introduction[¶][3]

`auto_LiRPA` is a library for automatically deriving and computing bounds with linear relaxation
based perturbation analysis (LiRPA) (e.g. [CROWN][4] and [DeepPoly][5]) for neural networks, which
is a useful tool for formal robustness verification. We generalize existing LiRPA algorithms for
feed-forward neural networks to a graph algorithm on general computational graphs, defined by
PyTorch. Additionally, our implementation is also automatically **differentiable**, allowing
optimizing network parameters to shape the bounds into certain specifications (e.g., certified
defense). You can find [a video ▶️ introduction here][6].

Our library supports the following algorithms:

* Backward mode LiRPA bound propagation ([CROWN][7]/[DeepPoly][8])
* Backward mode LiRPA bound propagation with optimized bounds ([α-CROWN][9])
* Backward mode LiRPA bound propagation with split constraints ([β-CROWN][10] for ReLU, and
  [GenBaB][11] for general nonlinear functions)
* Generalized backward mode LiRPA bound propagation with general cutting plane constraints
  ([GCP-CROWN][12])
* Backward mode LiRPA bound propagation with bounds tightened using output constraints
  ([INVPROP][13])
* Generalized backward mode LiRPA bound propagation for higher-order computational graphs ([Shi et
  al., 2022][14])
* Forward mode LiRPA bound propagation ([Xu et al., 2020][15])
* Forward mode LiRPA bound propagation with optimized bounds (similar to [α-CROWN][16])
* Interval bound propagation ([IBP][17])
* Hybrid approaches, e.g., Forward+Backward, IBP+Backward ([CROWN-IBP][18]), [α,β-CROWN][19]
  ([alpha-beta-CROWN][20])
* MIP/LP formulation of neural networks

Our library allows automatic bound derivation and computation for general computational graphs, in a
similar manner that gradients are obtained in modern deep learning frameworks – users only define
the computation in a forward pass, and `auto_LiRPA` traverses through the computational graph and
derives bounds for any nodes on the graph. With `auto_LiRPA` we free users from deriving and
implementing LiPRA for most common tasks, and they can simply apply LiPRA as a tool for their own
applications. This is especially useful for users who are not experts of LiRPA and cannot derive
these bounds manually (LiRPA is significantly more complicated than backpropagation).

## Usage[¶][21]

* [Installation][22]
* [Quick Start][23]
* [More Working Examples][24]
* [API Usage][25]
* [Custom Operators][26]
* [Reproducing our NeurIPS 2020 paper][27]

# [auto_LiRPA][28]

### Navigation

* [Installation][29]
* [Quick Start][30]
* [Examples][31]
* [API Usage][32]
* [Custom Operators][33]
* [Reproducing Our NeurIPS 2020 Paper][34]

### Related Topics

* [Documentation overview][35]
  
  * Next: [Installation][36]

### Quick search

©2020-2025, [auto-LiRPA authors][37]. | Powered by [Sphinx 7.4.7][38] & [Alabaster 0.7.16][39] |
[Page source][40]

[1]: https://github.com/Verified-Intelligence/auto_LiRPA
[2]: #documentation-for-auto-lirpa
[3]: #introduction
[4]: https://arxiv.org/pdf/1811.00866.pdf
[5]: https://files.sri.inf.ethz.ch/website/papers/DeepPoly.pdf
[6]: http://PaperCode.cc/AutoLiRPA-Video
[7]: https://arxiv.org/pdf/1811.00866.pdf
[8]: https://files.sri.inf.ethz.ch/website/papers/DeepPoly.pdf
[9]: https://arxiv.org/pdf/2011.13824.pdf
[10]: https://arxiv.org/pdf/2103.06624.pdf
[11]: https://arxiv.org/pdf/2405.21063
[12]: https://arxiv.org/pdf/2208.05740.pdf
[13]: https://arxiv.org/pdf/2302.01404.pdf
[14]: https://arxiv.org/abs/2210.07394
[15]: https://arxiv.org/pdf/2002.12920
[16]: https://arxiv.org/pdf/2011.13824.pdf
[17]: https://arxiv.org/pdf/1810.12715.pdf
[18]: https://arxiv.org/pdf/1906.06316.pdf
[19]: https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
[20]: https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
[21]: #usage
[22]: installation.html
[23]: quick-start.html
[24]: examples.html
[25]: api.html
[26]: custom_op.html
[27]: paper.html
[28]: #
[29]: installation.html
[30]: quick-start.html
[31]: examples.html
[32]: api.html
[33]: custom_op.html
[34]: paper.html
[35]: #
[36]: installation.html
[37]: https://github.com/Verified-Intelligence/auto_LiRPA#developers-and-copyright
[38]: https://www.sphinx-doc.org/
[39]: https://alabaster.readthedocs.io
[40]: _sources/index.rst.txt
