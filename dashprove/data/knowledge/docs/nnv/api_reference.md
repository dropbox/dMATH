# NNV

Matlab Toolbox for Neural Network Verification

This toolbox implements reachability methods for analyzing neural networks and control systems with
neural network controllers in the area of autonomous cyber-physical systems (CPS).

## Related tools and software

This toolbox makes use of the neural network model transformation tool ([nnmt][1]) and for
closed-loop systems analysis, the hybrid systems model transformation and translation tool
([HyST][2]), and the COntinuous Reachability Analyzer ([CORA][3]).

## Execution without installation:

NNV can be executed online without installing Matlab or other dependencies through [CodeOcean][4]
via the following CodeOcean capsules:

*Latest*

* CAV 2023 Tool Paper: [https://doi.org/10.24433/CO.0803700.v1][5]

*Previous*

* CAV 2020 ImageStar paper version: [https://doi.org/10.24433/CO.3351375.v1][6]
* CAV 2020 Tool paper version: [https://doi.org/10.24433/CO.0221760.v1][7]
* Earliest version: [https://doi.org/10.24433/CO.1314285.v1][8]

# Installation:

1. Clone or download the NNV toolbox from ([https://github.com/verivital/nnv][9])
   
   Note: to operate correctly, nnv depends on other tools (CORA, NNMT, HyST, onnx2nnv), which are
   included as git submodules. As such, you must clone recursively, e.g., with the following:
   
   `git clone --recursive https://github.com/verivital/nnv.git
   `
2. If running in Ubuntu, install MATLAB and proceed to run the provided installation script (then,
   skip to step 6).
   
   `chmod +x install_ubuntu.sh
   ./install_ubuntu.sh
   `
3. For MacOS and Windows, please install MATLAB (2023a or newer) with at least the following
   toolboxes:
   
   * Computer Vision
   * Control Systems
   * Deep Learning
   * Image Processing
   * Optimization
   * Parallel Computing
   * Statistics and Machine Learning
   * Symbolic Math
   * System Identification
4. Install the following support package [Deep Learning Toolbox Converter for ONNX Model Format][10]
   
   `Note: Support packages can be installed in MATLAB's HOME tab > Add-Ons > Get Add-ons, search for
    the support package using the Add-on Explorer and click on the Install button.
   `
5. Open MATLAB, then go to the directory where NNV exists on your machine, then run the `install.m`
   script located at /nnv/
   
   Note: if you restart Matlab, rerun either install.m or startup_nnv.m, which will add the
   necessary dependencies to the path; you alternatively can run `savepath` after installation to
   avoid this step after restarting Matlab, but this may require administrative privileges
6. Optional installation packages
   
   a. To run verification for convolutional neural networks (CNNs) on VGG16/VGG19, additional
   support packages must be installed:
   
   * [VGG16][11]
   * [VGG19][12]
   
   b) To run MATLAB's neural network verification comparison, an additional support package is
   needed (used in CAV'2023 submission):
   
   * [Deep Learning Toolbox Verification Library][13]
   
   c) To load models from other deep learning frameworks, please install the additional support
   packages:
   
   * TensorFlow and Keras: [Deep Learning Toolbox Converter for TensorFlow Models][14]
   * PyTorch: [Deep Learning Toolbox Converter for PyTorch Models][15]
   
   d) To use Conformal Prediction (CP) verification, set up a Python virtual environment:
   
   # From the NNV root directory:
   python -m venv .venv
   # Windows: .venv\Scripts\activate
   # macOS/Linux: source .venv/bin/activate
   pip install -r requirement.txt
   
   See [PYTHON_SETUP.md][16] for detailed instructions.

## Uninstallation:

Open MATLAB, then go to the `/code/nnv/` folder and execute the `uninstall.m` script.

# What's New in NNV 3.0

NNV 3.0 introduces major new verification capabilities:

* **VolumeStar**: Verification of video and 3D volumetric data (medical images)
* **FairNNV**: Formal verification of neural network fairness
* **Probabilistic Verification**: Scalable conformal prediction-based analysis
* **Weight Perturbation Analysis**: Verification under quantization/hardware errors
* **Time-Dependent Networks**: Variable-length time series verification
* **Malware Detection**: New cybersecurity verification domain

See [README_NNV3_CONTRIBUTIONS.md][17] for full details on NNV 3.0 features.

## Documentation

──────────────────────────────────┬──────────────────────────────────────
Document                          │Description                           
──────────────────────────────────┼──────────────────────────────────────
[README_NNV3_CONTRIBUTIONS.md][18]│NNV 3.0 new features and contributions
──────────────────────────────────┼──────────────────────────────────────
[README_TEST.md][19]              │Testing documentation and coverage    
──────────────────────────────────┼──────────────────────────────────────
[code/nnv/examples/README.md][20] │Examples navigation guide             
──────────────────────────────────┼──────────────────────────────────────
[code/nnv/examples/QuickStart/][21│Getting started examples              
]                                 │                                      
──────────────────────────────────┼──────────────────────────────────────
[code/nnv/examples/Tutorial/][22] │Step-by-step tutorials                
──────────────────────────────────┴──────────────────────────────────────

# Getting started with NNV

### Quick Start

**New to NNV?** Start with the [QuickStart examples][23] for installation verification and your
first neural network verification.

cd examples/QuickStart
test_installation      % Verify your setup
simple_verification    % Your first verification

**Troubleshooting?** Run the diagnostic tool:

check_nnv_setup()

### [Tutorial][24]

To get started with NNV, let's take a look at a tutorial containing examples demonstrating:

**NN**

* Robustness verification on the MNIST dataset.
  
  * Includes model training and several verification examples.
* Robustness verification on the GTSRB dataset.
  
  * Includes model training and robustness verification.
* Comparisons of exact (sound and complete) and approximate (sound and incomplete) methods using
  Star sets
  
  * Visualize the size difference on the output sets and the computation times for each method.
* Robustness analysis of a malware classifier (BODMAS Dataset).

**NNCS**

* Reachability analysis of an inverted pendulum.
* Safety verification example of an Adaptive Cruise Control (ACC) system.
* Safety verification of an Automated Emergency Braking System

And more! Please go to the [tutorial description][25] for more details!

### Examples

In addition to the examples from the tutorial, there are more examples in the 'code/nnv/examples/'
folder, including:

**Semantic Segmentation**

* [Robustness analysis of semantic segmentation NNs][26]

**Recurrent Neural Networks**

* [Robustness analysis of RNNs][27]

**Neural Ordinary Differential Equations**

* [Reachability analysis of neural ODEs][28]

And more other [NN][29] and [NNCS][30] examples.

### Tests

To run all the tests, one can run the following command from 'code/nnv/tests/' folder:

`runtests(pwd, 'IncludeSubfolders', true);`

### Paper Publications and Competitions

All the code for the publication using NNV, including competitions that NNV has participated in
(e.g. [VNNCOMP][31] and [ARCH-COMP][32]) can be found in the [examples/Submissions][33] folder. For
a selected subset of publications, [tags][34] were created. To reproduce those results, please
download NNV using the corresponding tag or use the corresponding CodeOcean capsules.

## Contributors

* [Hoang-Dung Tran][35]
* [Diego Manzanas Lopez][36]
* [Neelanjana Pal][37]
* [Weiming Xiang][38]
* [Stanley Bak][39]
* [Patrick Musau][40]
* [Xiaodong Yang][41]
* [Luan Viet Nguyen][42]
* [Taylor T. Johnson][43]

## References

The methods implemented in NNV are based upon or used in the following papers:

* Diego Manzanas Lopez, Sung Woo Choi, Hoang-Dung Tran, Taylor T. Johnson, "NNV 2.0: The Neural
  Network Verification Tool". In: Enea, C., Lal, A. (eds) Computer Aided Verification. CAV 2023.
  Lecture Notes in Computer Science, vol 13965. Springer, Cham.
  [[https://doi.org/10.1007/978-3-031-37703-7_19][44]]
* Anne M Tumlin, Diego Manzanas Lopez, Preston Robinette, Yuying Zhao, Tyler Derr, and Taylor T
  Johnson. "FairNNV: The Neural Network Verification Tool For Certifying Fairness. In Proceedings of
  the 5th ACM International Conference on AI in Finance (ICAIF '24)". Association for Computing
  Machinery, New York, NY, USA, 36–44. [[https://doi.org/10.1145/3677052.3698677][45]]
* Navid Hashemi, Samuel Sasaki, Ipek Oguz, Meiyi Ma, Taylor T. Johnson, "Scaling Data-Driven
  Probabilistic Robustness Analysis for Semantic Segmentation Neural Networks", 38th Conference on
  Neural Information Processing Systems (NeurIPS), 2025.
* Samuel Sasaki, Diego Manzanas Lopez, Preston K. Robinette, Taylor T. Johnson, "Robustness
  Verification of Video Classification Neural Networks", IEEE/ACM 13th International Conference on
  Formal Methods in Software Engineering (FormaliSE), 2025.
  [[https://doi.org/10.1109/FormaliSE66629.2025.00009][46]]
* Lucas C. Cordeiro, Matthew L. Daggitt, Julien Girard-Satabin, Omri Isac, Taylor T. Johnson, Guy
  Katz, Ekaterina Komendantskaya, Augustin Lemesle, Edoardo Manino, Artjoms Sinkarovs, Haoze Wu,
  "Neural Network Verification is a Programming Language Challenge", 34th European Symposium on
  Programming (ESOP), 2025. [[https://doi.org/10.1007/978-3-031-91118-7_9][47]]
* Muhammad Usama Zubair, Taylor T. Johnson, Kanad Basu, Waseem Abbas, "Verification of Neural
  Network Robustness Against Weight Perturbations Using Star Sets", IEEE Conference on Artificial
  Intelligence (CAI), 2025. [[https://doi.org/10.1109/CAI64502.2025.00117][48]]
* Diego Manzanas Lopez, Samuel Sasaki, Taylor T. Johnson, "NNV: A Star Set Reachability Approach
  (Competition Contribution)", 7th Workshop on Formal Methods for ML-Enabled Autonomous Systems
  (SAIV), 2025. [[https://doi.org/10.1007/978-3-031-99991-8_15][49]]
* Taylor T. Johnson, Diego Manzanas Lopez and Hoang-Dung. Tran, "Tutorial: Safe, Secure, and
  Trustworthy Artificial Intelligence (AI) via Formal Verification of Neural Networks and Autonomous
  Cyber-Physical Systems (CPS) with NNV," 2024 54th Annual IEEE/IFIP International Conference on
  Dependable Systems and Networks - Supplemental Volume (DSN-S), Brisbane, Australia, 2024, pp.
  65-66, [[https://doi.org/10.1109/DSN-S60304.2024.00027][50]]
* Preston K. Robinette, Diego Manzanas Lopez, Serena Serbinowska, Kevin Leach, and Taylor T Johnson.
  "Case Study: Neural Network Malware Detection Verification for Feature and Image Datasets". In
  Proceedings of the 2024 IEEE/ACM 12th International Conference on Formal Methods in Software
  Engineering (FormaliSE) (FormaliSE '24). Association for Computing Machinery, New York, NY, USA,
  127–137. [[https://doi.org/10.1145/3644033.3644372][51]]
* Hoang-Dung Tran, Diego Manzanas Lopez, and Taylor Johnson. "Tutorial: Neural Network and
  Autonomous Cyber-Physical Systems Formal Verification for Trustworthy AI and Safe Autonomy". In
  Proceedings of the International Conference on Embedded Software (EMSOFT '23). Association for
  Computing Machinery, New York, NY, USA, 1–2. [[https://doi.org/10.1145/3607890.3608454][52]]
* Neelanjana Pal, Diego Manzanas Lopez, Taylor T Johnson, "Robustness Verification of Deep Neural
  Networks using Star-Based Reachability Analysis with Variable-Length Time Series Input", to be
  presented at FMICS 2023. [[https://arxiv.org/pdf/2307.13907.pdf][53]]
* Mykhailo Ivashchenko, Sung Woo Choi, Luan Viet Nguyen, Hoang-Dung Tran, "Verifying Binary Neural
  Networks on Continuous Input Space using Star Reachability," 2023 IEEE/ACM 11th International
  Conference on Formal Methods in Software Engineering (FormaliSE), Melbourne, Australia, 2023, pp.
  7-17, [[https://doi.org/10.1109/FormaliSE58978.2023.00009][54]]
* Hoang Dung Tran, Sung Woo Choi, Xiaodong Yang, Tomoya Yamaguchi, Bardh Hoxha, and Danil Prokhorov.
  "Verification of Recurrent Neural Networks with Star Reachability". In Proceedings of the 26th ACM
  International Conference on Hybrid Systems: Computation and Control (HSCC '23). Association for
  Computing Machinery, New York, NY, USA, Article 6, 1–13.
  [[https://doi.org/10.1145/3575870.3587128][55]]
* Diego Manzanas Lopez, Taylor T. Johnson, Stanley Bak, Hoang-Dung Tran, Kerianne Hobbs, "Evaluation
  of Neural Network Verification Methods for Air to Air Collision Avoidance", In AIAA Journal of Air
  Transportation (JAT), 2022 [[http://www.taylortjohnson.com/research/lopez2022jat.pdf][56]]
* Diego Manzanas Lopez, Patrick Musau, Nathaniel Hamilton, Taylor T. Johnson, "Reachability Analysis
  of a General Class of Neural Ordinary Differential Equations", In 20th International Conference on
  Formal Modeling and Analysis of Timed Systems (FORMATS), 2022
  [[http://www.taylortjohnson.com/research/lopez2022formats.pdf][57]]
* Hoang-Dung Tran, Neelanjana Pal, Patrick Musau, Xiaodong Yang, Nathaniel P. Hamilton, Diego
  Manzanas Lopez, Stanley Bak, Taylor T. Johnson, "Robustness Verification of Semantic Segmentation
  Neural Networks using Relaxed Reachability", In 33rd International Conference on Computer-Aided
  Verification (CAV), Springer, 2021. [[http://www.taylortjohnson.com/research/tran2021cav.pdf][58]]
* Hoang-Dung Tran, Patrick Musau, Diego Manzanas Lopez, Xiaodong Yang, Luan Viet Nguyen, Weiming
  Xiang, Taylor T.Johnson, "NNV: A Tool for Verification of Deep Neural Networks and
  Learning-Enabled Autonomous Cyber-Physical Systems", 32nd International Conference on
  Computer-Aided Verification (CAV), 2020.
  [[http://taylortjohnson.com/research/tran2020cav_tool.pdf][59]]
* Hoang-Dung Tran, Stanley Bak, Weiming Xiang, Taylor T. Johnson, "Towards Verification of Large
  Convolutional Neural Networks Using ImageStars", 32nd International Conference on Computer-Aided
  Verification (CAV), 2020. [[http://taylortjohnson.com/research/tran2020cav.pdf][60]]
* Stanley Bak, Hoang-Dung Tran, Kerianne Hobbs, Taylor T. Johnson, "Improved Geometric Path
  Enumeration for Verifying ReLU Neural Networks", In 32nd International Conference on
  Computer-Aided Verification (CAV), 2020.
  [[http://www.taylortjohnson.com/research/bak2020cav.pdf][61]]
* Hoang-Dung Tran, Weiming Xiang, Taylor T. Johnson, "Verification Approaches for Learning-Enabled
  Autonomous Cyber-Physical Systems", The IEEE Design & Test 2020.
  [[http://www.taylortjohnson.com/research/tran2020dandt.pdf][62]]
* Hoang-Dung Tran, Patrick Musau, Diego Manzanas Lopez, Xiaodong Yang, Luan Viet Nguyen, Weiming
  Xiang, Taylor T.Johnson, "Star-Based Reachability Analysis for Deep Neural Networks", The 23rd
  International Symposium on Formal Methods (FM), Porto, Portugal, 2019, Acceptance Rate 30%. .
  [[http://taylortjohnson.com/research/tran2019fm.pdf][63]]
* Hoang-Dung Tran, Feiyang Cei, Diego Manzanas Lopez, Taylor T.Johnson, Xenofon Koutsoukos, "Safety
  Verification of Cyber-Physical Systems with Reinforcement Learning Control", The International
  Conference on Embedded Software (EMSOFT), New York, October 2019. Acceptance Rate 25%.
  [[http://taylortjohnson.com/research/tran2019emsoft.pdf][64]]
* Hoang-Dung Tran, Patrick Musau, Diego Manzanas Lopez, Xiaodong Yang, Luan Viet Nguyen, Weiming
  Xiang, Taylor T.Johnson, "Parallelizable Reachability Analysis Algorithms for FeedForward Neural
  Networks", In 7th International Conference on Formal Methods in Software Engineering (FormaLISE),
  27, May 2019 in Montreal, Canada, Acceptance Rate 28%.
  [[http://taylortjohnson.com/research/tran2019formalise.pdf][65]]
* Diego Manzanas Lopez, Patrick Musau, Hoang-Dung Tran, Taylor T.Johnson, "Verification of
  Closed-loop Systems with Neural Network Controllers (Benchmark Proposal)", The 6th International
  Workshop on Applied Verification of Continuous and Hybrid Systems (ARCH2019). Montreal, Canada,
  2019. [[http://taylortjohnson.com/research/lopez2019arch.pdf][66]]
* Weiming Xiang, Hoang-Dung Tran, Taylor T. Johnson, "Output Reachable Set Estimation and
  Verification for Multi-Layer Neural Networks", In IEEE Transactions on Neural Networks and
  Learning Systems (TNNLS), 2018, March.
  [[http://taylortjohnson.com/research/xiang2018tnnls.pdf][67]]
* Weiming Xiang, Hoang-Dung Tran, Taylor T. Johnson, "Reachable Set Computation and Safety
  Verification for Neural Networks with ReLU Activations", In In Submission, IEEE, 2018, September.
  [[http://www.taylortjohnson.com/research/xiang2018tcyb.pdf][68]]
* Weiming Xiang, Diego Manzanas Lopez, Patrick Musau, Taylor T. Johnson, "Reachable Set Estimation
  and Verification for Neural Network Models of Nonlinear Dynamic Systems", In Unmanned System
  Technologies: Safe, Autonomous and Intelligent Vehicles, Springer, 2018, September.
  [[http://www.taylortjohnson.com/research/xiang2018ust.pdf][69]]
* Reachability Analysis and Safety Verification for Neural Network Control Systems, Weiming Xiang,
  Taylor T. Johnson [[https://arxiv.org/abs/1805.09944][70]]
* Weiming Xiang, Patrick Musau, Ayana A. Wild, Diego Manzanas Lopez, Nathaniel Hamilton, Xiaodong
  Yang, Joel Rosenfeld, Taylor T. Johnson, "Verification for Machine Learning, Autonomy, and Neural
  Networks Survey," October 2018, [[https://arxiv.org/abs/1810.01989][71]]
* Specification-Guided Safety Verification for Feedforward Neural Networks, Weiming Xiang,
  Hoang-Dung Tran, Taylor T. Johnson [[https://arxiv.org/abs/1812.06161][72]]

#### VNN-COMP Competition Reports

* Christopher Brix, Stanley Bak, Taylor T. Johnson, Haoze Wu, "The Fifth International Verification
  of Neural Networks Competition (VNN-COMP 2024): Summary and Results", arXiv:2412.19985, 2024.
  [[https://doi.org/10.48550/arXiv.2412.19985][73]]
* Christopher Brix, Stanley Bak, Changliu Liu, Taylor T. Johnson, "The Fourth International
  Verification of Neural Networks Competition (VNN-COMP 2023): Summary and Results",
  arXiv:2312.16760, 2023. [[https://arxiv.org/abs/2312.16760][74]]
* Mark Niklas Müller, Christopher Brix, Stanley Bak, Changliu Liu, Taylor T. Johnson, "The Third
  International Verification of Neural Networks Competition (VNN-COMP 2022): Summary and Results",
  arXiv:2212.10376, 2022. [[https://doi.org/10.48550/arXiv.2212.10376][75]]
* Stanley Bak, Changliu Liu, Taylor T. Johnson, "The Second International Verification of Neural
  Networks Competition (VNN-COMP 2021): Summary and Results", arXiv:2109.00498, 2021.
  [[https://arxiv.org/abs/2109.00498][76]]
* Christopher Brix, Mark Niklas Müller, Stanley Bak, Taylor T. Johnson, Changliu Liu, "First Three
  Years of the International Verification of Neural Networks Competition (VNN-COMP)", Int J Softw
  Tools Technol Transfer 25, 329-339, 2023. [[https://doi.org/10.1007/s10009-023-00703-4][77]]

#### ARCH-COMP AINNCS Category Reports

* Diego Manzanas Lopez, Matthias Althoff, Luis Benet, Marcelo Forets, Taylor T. Johnson, et al.,
  "ARCH-COMP24 Category Report: Artificial Intelligence and Neural Network Control Systems (AINNCS)
  for Continuous and Hybrid Systems Plants", ARCH 2024.
  [[https://easychair.org/publications/paper/WsgX][78]]
* Diego Manzanas Lopez, et al., "ARCH-COMP23 Category Report: Artificial Intelligence and Neural
  Network Control Systems (AINNCS) for Continuous and Hybrid Systems Plants", ARCH 2023.
  [[https://easychair.org/publications/paper/Vfq4b][79]]
* Diego Manzanas Lopez, et al., "ARCH-COMP22 Category Report: Artificial Intelligence and Neural
  Network Control Systems (AINNCS) for Continuous and Hybrid Systems Plants", ARCH 2022.
  [[https://easychair.org/publications/paper/C1J8][80]]
* Diego Manzanas Lopez, et al., "ARCH-COMP21 Category Report: Artificial Intelligence and Neural
  Network Control Systems (AINNCS) for Continuous and Hybrid Systems Plants", ARCH 2021.
  [[https://easychair.org/publications/paper/Jq4h][81]]
* Diego Manzanas Lopez, et al., "ARCH-COMP20 Category Report: Artificial Intelligence and Neural
  Network Control Systems (AINNCS) for Continuous and Hybrid Systems Plants", ARCH 2020.
  [[https://easychair.org/publications/paper/Jvwg][82]]

#### Cite

`@inproceedings{nnv2_cav2023,
author = {Lopez, Diego Manzanas and Choi, Sung Woo and Tran, Hoang-Dung and Johnson, Taylor T.},
title = {NNV 2.0: The Neural Network Verification Tool},
year = {2023},
isbn = {978-3-031-37702-0},
publisher = {Springer-Verlag},
address = {Berlin, Heidelberg},
url = {https://doi.org/10.1007/978-3-031-37703-7_19},
doi = {10.1007/978-3-031-37703-7_19},
abstract = {This manuscript presents the updated version of the Neural Network Verification (NNV) to
ol. NNV is a formal verification software tool for deep learning models and cyber-physical systems w
ith neural network components. NNV was first introduced as a verification framework for feedforward 
and convolutional neural networks, as well as for neural network control systems. Since then, numero
us works have made significant improvements in the verification of new deep learning models, as well
 as tackling some of the scalability issues that may arise when verifying complex models. In this ne
w version of NNV, we introduce verification support for multiple deep learning models, including neu
ral ordinary differential equations, semantic segmentation networks and recurrent neural networks, a
s well as a collection of reachability methods that aim to reduce the computation cost of reachabili
ty analysis of complex neural networks. We have also added direct support for standard input verific
ation formats in the community such as VNNLIB (verification properties), and ONNX (neural networks) 
formats. We present a collection of experiments in which NNV verifies safety and robustness properti
es of feedforward, convolutional, semantic segmentation and recurrent neural networks, as well as ne
ural ordinary differential equations and neural network control systems. Furthermore, we demonstrate
 the capabilities of NNV against a commercially available product in a collection of benchmarks from
 control systems, semantic segmentation, image classification, and time-series data.},
booktitle = {Computer Aided Verification: 35th International Conference, CAV 2023, Paris, France, Ju
ly 17–22, 2023, Proceedings, Part II},
pages = {397–412},
numpages = {16},
keywords = {neural networks, cyber-physical systems, verification, tool},
location = {Paris, France}
}
`
`@inproceedings{nnv_cav2020,
author = {Tran, Hoang-Dung and Yang, Xiaodong and Manzanas Lopez, Diego and Musau, Patrick and Nguye
n, Luan Viet and Xiang, Weiming and Bak, Stanley and Johnson, Taylor T.},
title = {NNV: The Neural Network Verification Tool for Deep Neural Networks and Learning-Enabled Cyb
er-Physical Systems},
year = {2020},
isbn = {978-3-030-53287-1},
publisher = {Springer-Verlag},
address = {Berlin, Heidelberg},
url = {https://doi.org/10.1007/978-3-030-53288-8_1},
doi = {10.1007/978-3-030-53288-8_1},
abstract = {This paper presents the Neural Network Verification (NNV) software tool, a set-based ver
ification framework for deep neural networks (DNNs) and learning-enabled cyber-physical systems (CPS
). The crux of NNV is a collection of reachability algorithms that make use of a variety of set repr
esentations, such as polyhedra, star sets, zonotopes, and abstract-domain representations. NNV suppo
rts both exact (sound and complete) and over-approximate (sound) reachability algorithms for verifyi
ng safety and robustness properties of feed-forward neural networks (FFNNs) with various activation 
functions. For learning-enabled CPS, such as closed-loop control systems incorporating neural networ
ks, NNV provides exact and over-approximate reachability analysis schemes for linear plant models an
d FFNN controllers with piecewise-linear activation functions, such as ReLUs. For similar neural net
work control systems (NNCS) that instead have nonlinear plant models, NNV supports over-approximate 
analysis by combining the star set analysis used for FFNN controllers with zonotope-based analysis f
or nonlinear plant dynamics building on CORA. We evaluate NNV using two real-world case studies: the
 first is safety verification of ACAS Xu networks, and the second deals with the safety verification
 of a deep learning-based adaptive cruise control system.},
booktitle = {Computer Aided Verification: 32nd International Conference, CAV 2020, Los Angeles, CA, 
USA, July 21–24, 2020, Proceedings, Part I},
pages = {3–17},
numpages = {15},
keywords = {Autonomy, Verification, Cyber-physical systems, Machine learning, Neural networks},
location = {Los Angeles, CA, USA}
}
`

### Acknowledgements

This work is supported in part by AFOSR, DARPA, NSF.

### Contact

For any questions related to NNV, please add them to the issues or contact [Diego Manzanas
Lopez][83] or [Hoang Dung Tran][84].

[1]: https://github.com/verivital/nnmt
[2]: https://github.com/verivital/hyst
[3]: https://github.com/TUMcps/CORA
[4]: https://www.codeocean.com
[5]: https://doi.org/10.24433/CO.0803700.v1
[6]: https://doi.org/10.24433/CO.3351375.v1
[7]: https://doi.org/10.24433/CO.0221760.v1
[8]: https://doi.org/10.24433/CO.1314285.v1
[9]: https://github.com/verivital/nnv
[10]: https://www.mathworks.com/matlabcentral/fileexchange/67296-deep-learning-toolbox-converter-for
-onnx-model-format
[11]: https://www.mathworks.com/matlabcentral/fileexchange/61733-deep-learning-toolbox-model-for-vgg
-16-network
[12]: https://www.mathworks.com/help/deeplearning/ref/vgg19.html
[13]: https://www.mathworks.com/matlabcentral/fileexchange/118735-deep-learning-toolbox-verification
-library
[14]: https://www.mathworks.com/matlabcentral/fileexchange/64649-deep-learning-toolbox-converter-for
-tensorflow-models
[15]: https://www.mathworks.com/matlabcentral/fileexchange/111925-deep-learning-toolbox-converter-fo
r-pytorch-models
[16]: /verivital/nnv/blob/master/PYTHON_SETUP.md
[17]: /verivital/nnv/blob/master/README_NNV3_CONTRIBUTIONS.md
[18]: /verivital/nnv/blob/master/README_NNV3_CONTRIBUTIONS.md
[19]: /verivital/nnv/blob/master/README_TEST.md
[20]: /verivital/nnv/blob/master/code/nnv/examples/README.md
[21]: /verivital/nnv/blob/master/code/nnv/examples/QuickStart
[22]: /verivital/nnv/blob/master/code/nnv/examples/Tutorial
[23]: /verivital/nnv/blob/master/code/nnv/examples/QuickStart
[24]: /verivital/nnv/blob/master/code/nnv/examples/Tutorial
[25]: /verivital/nnv/blob/master/code/nnv/examples/Tutorial/readme.md
[26]: /verivital/nnv/blob/master/code/nnv/examples/NN/SemanticSegmentation/M2NIST
[27]: /verivital/nnv/blob/master/code/nnv/examples/NN/RNN
[28]: /verivital/nnv/blob/master/code/nnv/examples/NN/NeuralODEs
[29]: /verivital/nnv/blob/master/code/nnv/examples/NN
[30]: /verivital/nnv/blob/master/code/nnv/examples/NNCS
[31]: https://sites.google.com/view/vnn2023
[32]: https://cps-vo.org/group/ARCH/FriendlyCompetition
[33]: /verivital/nnv/blob/master/code/nnv/examples/Submission
[34]: https://github.com/verivital/nnv/tags
[35]: https://sites.google.com/view/v2a2/
[36]: https://mldiego.github.io/
[37]: https://scholar.google.com/citations?user=3j_f-ewAAAAJ&hl=en
[38]: https://xiangweiming.github.io/
[39]: http://stanleybak.com/
[40]: https://pmusau17.github.io/
[41]: https://scholar.google.com/citations?user=xe3Jr7EAAAAJ&hl=en
[42]: https://luanvietnguyen.github.io
[43]: http://www.taylortjohnson.com
[44]: https://doi.org/10.1007/978-3-031-37703-7_19
[45]: https://doi.org/10.1145/3677052.3698677
[46]: https://doi.org/10.1109/FormaliSE66629.2025.00009
[47]: https://doi.org/10.1007/978-3-031-91118-7_9
[48]: https://doi.org/10.1109/CAI64502.2025.00117
[49]: https://doi.org/10.1007/978-3-031-99991-8_15
[50]: https://doi.org/10.1109/DSN-S60304.2024.00027
[51]: https://doi.org/10.1145/3644033.3644372
[52]: https://doi.org/10.1145/3607890.3608454
[53]: https://arxiv.org/pdf/2307.13907.pdf
[54]: https://doi.org/10.1109/FormaliSE58978.2023.00009
[55]: https://doi.org/10.1145/3575870.3587128
[56]: http://www.taylortjohnson.com/research/lopez2022jat.pdf
[57]: http://www.taylortjohnson.com/research/lopez2022formats.pdf
[58]: http://www.taylortjohnson.com/research/tran2021cav.pdf
[59]: http://taylortjohnson.com/research/tran2020cav_tool.pdf
[60]: http://taylortjohnson.com/research/tran2020cav.pdf
[61]: http://www.taylortjohnson.com/research/bak2020cav.pdf
[62]: http://www.taylortjohnson.com/research/tran2020dandt.pdf
[63]: http://taylortjohnson.com/research/tran2019fm.pdf
[64]: http://taylortjohnson.com/research/tran2019emsoft.pdf
[65]: http://taylortjohnson.com/research/tran2019formalise.pdf
[66]: http://taylortjohnson.com/research/lopez2019arch.pdf
[67]: http://taylortjohnson.com/research/xiang2018tnnls.pdf
[68]: http://www.taylortjohnson.com/research/xiang2018tcyb.pdf
[69]: http://www.taylortjohnson.com/research/xiang2018ust.pdf
[70]: https://arxiv.org/abs/1805.09944
[71]: https://arxiv.org/abs/1810.01989
[72]: https://arxiv.org/abs/1812.06161
[73]: https://doi.org/10.48550/arXiv.2412.19985
[74]: https://arxiv.org/abs/2312.16760
[75]: https://doi.org/10.48550/arXiv.2212.10376
[76]: https://arxiv.org/abs/2109.00498
[77]: https://doi.org/10.1007/s10009-023-00703-4
[78]: https://easychair.org/publications/paper/WsgX
[79]: https://easychair.org/publications/paper/Vfq4b
[80]: https://easychair.org/publications/paper/C1J8
[81]: https://easychair.org/publications/paper/Jq4h
[82]: https://easychair.org/publications/paper/Jvwg
[83]: mailto:diego.manzanas.lopez@vanderbilt.edu
[84]: mailto:trhoangdung@gmail.com
