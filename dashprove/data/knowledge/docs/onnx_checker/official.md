# onnx.checker[¶][1]

## CheckerContext[¶][2]

* onnx.checker.DEFAULT_CONTEXT[¶][3]*
  alias of <onnx.onnx_cpp2py_export.checker.CheckerContext object>

## The `onnx.checker` module[¶][4]

Graph utilities for checking whether an ONNX proto message is legal.

* *exception *onnx.checker.ValidationError[¶][5]*
  Bases: [`Exception`][6]

* onnx.checker.check_attribute(*attr: ~onnx.onnx_ml_pb2.AttributeProto*, *ctx:
~onnx.onnx_cpp2py_export.checker.CheckerContext = <onnx.onnx_cpp2py_export.checker.CheckerContext
object>*, *lexical_scope_ctx: ~onnx.onnx_cpp2py_export.checker.LexicalScopeContext =
<onnx.onnx_cpp2py_export.checker.LexicalScopeContext object>*) → [None][7][[source]][8][¶][9]*

* onnx.checker.check_function(*function: ~onnx.onnx_ml_pb2.FunctionProto*, *ctx:
~onnx.onnx_cpp2py_export.checker.CheckerContext | None = None*, *lexical_scope_ctx:
~onnx.onnx_cpp2py_export.checker.LexicalScopeContext =
<onnx.onnx_cpp2py_export.checker.LexicalScopeContext object>*) → [None][10][[source]][11][¶][12]*

* onnx.checker.check_graph(*graph: ~onnx.onnx_ml_pb2.GraphProto*, *ctx:
~onnx.onnx_cpp2py_export.checker.CheckerContext = <onnx.onnx_cpp2py_export.checker.CheckerContext
object>*, *lexical_scope_ctx: ~onnx.onnx_cpp2py_export.checker.LexicalScopeContext =
<onnx.onnx_cpp2py_export.checker.LexicalScopeContext object>*) → [None][13][[source]][14][¶][15]*

* onnx.checker.check_model(*model: [ModelProto][16] | [str][17] | [bytes][18] | [PathLike][19]*,
*full_check: [bool][20] = False*, *skip_opset_compatibility_check: [bool][21] = False*,
*check_custom_domain: [bool][22] = False*) → [None][23][[source]][24][¶][25]*
  Check the consistency of a model.
  
  An exception will be raised if the model’s ir_version is not set properly or is higher than
  checker’s ir_version, or if the model has duplicate keys in metadata_props.
  
  If IR version >= 3, the model must specify opset_import. If IR version < 3, the model cannot have
  any opset_import specified.
  
  *Parameters:*
    * **model** – Model to check. If model is a path, the function checks model path first. If the
      model bytes size is larger than 2GB, function should be called using model path.
    * **full_check** – If True, the function also runs shape inference check.
    * **skip_opset_compatibility_check** – If True, the function skips the check for opset
      compatibility.
    * **check_custom_domain** – If True, the function will check all domains. Otherwise only check
      built-in domains.

* onnx.checker.check_node(*node: ~onnx.onnx_ml_pb2.NodeProto*, *ctx:
~onnx.onnx_cpp2py_export.checker.CheckerContext = <onnx.onnx_cpp2py_export.checker.CheckerContext
object>*, *lexical_scope_ctx: ~onnx.onnx_cpp2py_export.checker.LexicalScopeContext =
<onnx.onnx_cpp2py_export.checker.LexicalScopeContext object>*) → [None][26][[source]][27][¶][28]*

* onnx.checker.check_sparse_tensor(*sparse: ~onnx.onnx_ml_pb2.SparseTensorProto*, *ctx:
~onnx.onnx_cpp2py_export.checker.CheckerContext = <onnx.onnx_cpp2py_export.checker.CheckerContext
object>*) → [None][29][[source]][30][¶][31]*

* onnx.checker.check_tensor(*tensor: ~onnx.onnx_ml_pb2.TensorProto*, *ctx:
~onnx.onnx_cpp2py_export.checker.CheckerContext = <onnx.onnx_cpp2py_export.checker.CheckerContext
object>*) → [None][32][[source]][33][¶][34]*

* onnx.checker.check_value_info(*value_info: ~onnx.onnx_ml_pb2.ValueInfoProto*, *ctx:
~onnx.onnx_cpp2py_export.checker.CheckerContext = <onnx.onnx_cpp2py_export.checker.CheckerContext
object>*) → [None][35][[source]][36][¶][37]*

[1]: #onnx-checker
[2]: #checkercontext
[3]: #onnx.checker.DEFAULT_CONTEXT
[4]: #module-onnx.checker
[5]: #onnx.checker.ValidationError
[6]: https://docs.python.org/3/library/exceptions.html#Exception
[7]: https://docs.python.org/3/library/constants.html#None
[8]: ../_modules/onnx/checker.html#check_attribute
[9]: #onnx.checker.check_attribute
[10]: https://docs.python.org/3/library/constants.html#None
[11]: ../_modules/onnx/checker.html#check_function
[12]: #onnx.checker.check_function
[13]: https://docs.python.org/3/library/constants.html#None
[14]: ../_modules/onnx/checker.html#check_graph
[15]: #onnx.checker.check_graph
[16]: serialization.html#onnx.ModelProto
[17]: https://docs.python.org/3/library/stdtypes.html#str
[18]: https://docs.python.org/3/library/stdtypes.html#bytes
[19]: https://docs.python.org/3/library/os.html#os.PathLike
[20]: https://docs.python.org/3/library/functions.html#bool
[21]: https://docs.python.org/3/library/functions.html#bool
[22]: https://docs.python.org/3/library/functions.html#bool
[23]: https://docs.python.org/3/library/constants.html#None
[24]: ../_modules/onnx/checker.html#check_model
[25]: #onnx.checker.check_model
[26]: https://docs.python.org/3/library/constants.html#None
[27]: ../_modules/onnx/checker.html#check_node
[28]: #onnx.checker.check_node
[29]: https://docs.python.org/3/library/constants.html#None
[30]: ../_modules/onnx/checker.html#check_sparse_tensor
[31]: #onnx.checker.check_sparse_tensor
[32]: https://docs.python.org/3/library/constants.html#None
[33]: ../_modules/onnx/checker.html#check_tensor
[34]: #onnx.checker.check_tensor
[35]: https://docs.python.org/3/library/constants.html#None
[36]: ../_modules/onnx/checker.html#check_value_info
[37]: #onnx.checker.check_value_info
