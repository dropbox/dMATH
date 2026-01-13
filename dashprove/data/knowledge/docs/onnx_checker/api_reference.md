# API Reference[¶][1]

Tip

The [ir-py project][2] provides alternative Pythonic APIs for creating and manipulating ONNX models
without interaction with Protobuf.

## Versioning[¶][3]

The following example shows how to retrieve onnx version, the onnx opset, the IR version. Every new
major release increments the opset version (see [Opset Version][4]).

from onnx import __version__, IR_VERSION
from onnx.defs import onnx_opset_version
print(f"onnx.__version__={__version__!r}, opset={onnx_opset_version()}, IR_VERSION={IR_VERSION}")
onnx.__version__='1.21.0', opset=26, IR_VERSION=13

The intermediate representation (IR) specification is the abstract model for graphs and operators
and the concrete format that represents them. Adding a structure, modifying one them increases the
IR version.

The opset version increases when an operator is added or removed or modified. A higher opset means a
longer list of operators and more options to implement an ONNX functions. An operator is usually
modified because it supports more input and output type, or an attribute becomes an input.

## Data Structures[¶][5]

Every ONNX object is defined based on a [protobuf message][6] and has a name ended with suffix
`Proto`. For example, [NodeProto][7] defines an operator, [TensorProto][8] defines a tensor. Next
page lists all of them.

* [Protos][9]
* [Serialization][10]

## Functions[¶][11]

An ONNX model can be directly from the classes described in previous section but it is faster to
create and verify a model with the following helpers.

* [onnx.backend][12]
* [onnx.checker][13]
* [onnx.compose][14]
* [onnx.defs][15]
* [onnx.external_data_helper][16]
* [onnx.helper][17]
* [onnx.hub][18]
* [onnx.inliner][19]
* [onnx.model_container][20]
* [onnx.numpy_helper][21]
* [onnx.parser][22]
* [onnx.printer][23]
* [onnx.reference][24]
* [onnx.shape_inference][25]
* [onnx.tools][26]
* [onnx.utils][27]
* [onnx.version_converter][28]

[1]: #api-reference
[2]: https://github.com/onnx/ir-py
[3]: #versioning
[4]: defs.html#l-api-opset-version
[5]: #data-structures
[6]: https://googleapis.dev/python/protobuf/latest/google/protobuf/message.html
[7]: classes.html#l-nodeproto
[8]: classes.html#l-tensorproto
[9]: classes.html
[10]: serialization.html
[11]: #functions
[12]: backend.html
[13]: checker.html
[14]: compose.html
[15]: defs.html
[16]: external_data_helper.html
[17]: helper.html
[18]: hub.html
[19]: inliner.html
[20]: model_container.html
[21]: numpy_helper.html
[22]: parser.html
[23]: printer.html
[24]: reference.html
[25]: shape_inference.html
[26]: tools.html
[27]: utils.html
[28]: version_converter.html
