[[Build Status]][1]

# nnenum - The Neural Network Enumeration Tool

**nnenum** (pronounced *en-en-en-um*) is a high-performance neural network verification tool.
Multiple levels of abstraction are used to quickly verify ReLU networks without sacrificing
completeness. Analysis combines three types of zonotopes with star set (triangle)
overapproximations, and uses [efficient parallelized ReLU case splitting][2]. The tool is written in
Python 3, uses GLPK for LP solving and directly accepts [ONNX][3] network files and `vnnlib`
specifications as input. The [ImageStar trick][4] allows sets to be quickly propagated through all
layers supported by the [ONNX runtime][5], such as convolutional layers with arbitrary parameters.

The tool is written by Stanley Bak ([homepage][6], [twitter][7]).

### Getting Started

The `Dockerfile` shows how to install all the dependencies (mostly python packages) and set up the
environment. The tool loads neural networks directly from ONNX files and properties to check from
`vnnlib` files. For example, try running:

`python3 -m nnenum.nnenum examples/acasxu/data/ACASXU_run2a_3_3_batch_2000.onnx examples/acasxu/data
/prop_9.vnnlib
`

You can see a few more examples in `run_tests.sh`.

### VNN 2020 Neural Network Verification Competition (VNN-COMP) Version

The nnenum tool performed well in VNN-COMP 2020, being the only tool to verify all the ACAS-Xu
benchmarks (each in under 10 seconds). The version used for the competition as well as model files
and scripts to run the compeition benchmarks are in the `vnn2020` branch.

### CAV 2020 Paper Version

The CAV 2020 paper ["Improved Geometric Path Enumeration for Verifying ReLU Neural Networks"][8] by
S. Bak, H.D Tran, K. Hobbs and T. T. Johnson corresponds to optimizations integrated into the exact
analysis mode of nnenum, which also benefits overapproximative analysis. The paper version and
repeatability evaluation package instructions are available [here][9].

### Citing nnenum

The following citations can be used for nnenum:

`@inproceedings{bak2021nfm,
  title={nnenum: Verification of ReLU Neural Networks with Optimized Abstraction Refinement},
  author={Bak, Stanley},
  booktitle={NASA Formal Methods Symposium},
  pages={19--36},
  year={2021},
  organization={Springer}
}
`
`@inproceedings{bak2020cav,
  title={Improved Geometric Path Enumeration for Verifying ReLU Neural Networks},
  author={Bak, Stanley and Tran, Hoang-Dung and Hobbs, Kerianne and Johnson, Taylor T.},
  booktitle={Proceedings of the 32nd International Conference on Computer Aided Verification},
  year={2020},
  organization={Springer}
}
`

[1]: https://app.travis-ci.com/github/stanleybak/nnenum
[2]: http://stanleybak.com/papers/bak2020cav.pdf
[3]: https://github.com/onnx/onnx
[4]: https://arxiv.org/abs/2004.05511
[5]: https://github.com/microsoft/onnxruntime
[6]: http://stanleybak.com
[7]: https://twitter.com/StanleyBak
[8]: http://stanleybak.com/papers/bak2020cav.pdf
[9]: http://stanleybak.com/papers/bak2020cav_repeatability.zip
