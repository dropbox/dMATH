* »
* Maraboupy
* [ View page source][1]

# Maraboupy[¶][2]

Maraboupy is the python interface for Marabou, an SMT-based neural network verification tool. Neural
networks can be supplied in tensorflow, ONNX, or NNet formats, and piecewise-linear constraints can
be added to the network variables to encode a desired property for verification. Maraboupy uses
pybind11 to solve the neural network query using the C++ implementation of Marabou. In general, the
solver will return either UNSAT if no assignment of values to all network variables can satisfy all
equations and constraints, or SAT if a satisfying assignment exists.

This documentation explains how to setup Maraboupy, shows examples using Maraboupy, provides API
documentation, and explains how to contribute to Maraboupy.

Setup[¶][3]

* [Installation][4]
* [Testing][5]
* [Troubleshooting][6]

Examples[¶][7]

* [NNet Example][8]
* [Tensorflow Example][9]
* [ONNX Example][10]
* [MarabouCore Example][11]
* [DNC Example][12]
* [Disjunction Constraint Example][13]

API Documentation[¶][14]

* [Marabou][15]
* [MarabouNetwork][16]
* [MarabouNetworkNNet][17]
* [MarabouNetworkTF][18]
* [MarabouNetworkONNX][19]
* [MarabouUtils][20]
* [MarabouCore][21]

Developer's Guide[¶][22]

* [Contributing to Marabou][23]
* [Pull Requests][24]
* [Coding Style Guidelines][25]
* [Tests][26]
* [Examples][27]
* [Documentation][28]
[Next ][29]

© Copyright 2020, The Marabou Team

Built with [Sphinx][30] using a [theme][31] provided by [Read the Docs][32].

[1]: _sources/index.rst.txt
[2]: #maraboupy
[3]: #setup
[4]: Setup/0_Installation.html
[5]: Setup/1_Testing.html
[6]: Setup/2_Troubleshooting.html
[7]: #examples
[8]: Examples/0_NNetExample.html
[9]: Examples/1_TensorflowExample.html
[10]: Examples/2_ONNXExample.html
[11]: Examples/3_MarabouCoreExample.html
[12]: Examples/4_DncExample.html
[13]: Examples/5_DisjunctionConstraintExample.html
[14]: #api
[15]: API/0_Marabou.html
[16]: API/1_MarabouNetwork.html
[17]: API/2_MarabouNetworkNNet.html
[18]: API/3_MarabouNetworkTF.html
[19]: API/4_MarabouNetworkONNX.html
[20]: API/5_MarabouUtils.html
[21]: API/6_MarabouCore.html
[22]: #develop
[23]: Develop/0_Guide.html
[24]: Develop/1_PullRequests.html
[25]: Develop/2_CodingStyle.html
[26]: Develop/3_Tests.html
[27]: Develop/4_Examples.html
[28]: Develop/5_Documentation.html
[29]: Setup/0_Installation.html
[30]: http://sphinx-doc.org/
[31]: https://github.com/rtfd/sphinx_rtd_theme
[32]: https://readthedocs.org
