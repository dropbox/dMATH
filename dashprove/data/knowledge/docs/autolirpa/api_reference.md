# API Usage[¶][1]

* *class *auto_LiRPA.BoundedModule(*model*, *global_input*, *bound_opts=None*, *device='auto'*,
*verbose=False*, *custom_ops=None*)[[source]][2][¶][3]*
  Bounded module with support for automatically computing bounds.
  
  *Args:*
    model (nn.Module): The original model to be wrapped by BoundedModule.
    
    global_input (tuple): A dummy input to the original model. The shape of the dummy input should
    be consistent with the actual input to the model except for the batch dimension.
    
    bound_opts (dict): Options for bounds. See [Bound Options][4].
    
    device (str or torch.device): Device of the bounded module. If ‘auto’, the device will be
    automatically inferred from the device of parameters in the original model or the dummy input.
    
    custom_ops (dict): A dictionary of custom operators. The dictionary maps operator names to their
    corresponding bound classes (subclasses of Bound).
  
  * forward(*self*, **x*, *final_node_name=None*, *clear_forward_only=False*,
  *reset_perturbed_nodes=True*)[[source]][5][¶][6]*
    Standard forward computation for the network.
    
    *Args:*
      x (tuple or None): Input to the model.
      
      final_node_name (str, optional): The name of the final node in the model. The value on the
      corresponding node will be returned.
      
      clear_forward_only (bool, default False): Whether only standard forward values stored on the
      nodes should be cleared. If True, only standard forward values stored on the nodes will be
      cleared. Otherwise, bound information on the nodes will also be cleared.
      
      reset_perturbed_nodes (bool, default True): Mark all perturbed nodes with input perturbations.
      When set to True, it may accidentally clear all .perturbed properties for intermediate nodes.
    *Returns:*
      output: The output of the model, or if final_node_name is not None, return the value on the
      corresponding node instead.
  
  * compute_bounds(*self*, *x=None*, *aux=None*, *C=None*, *method='backward'*, *IBP=False*,
  *forward=False*, *bound_lower=True*, *bound_upper=True*, *reuse_ibp=False*, *reuse_alpha=False*,
  *return_A=False*, *needed_A_dict=None*, *final_node_name=None*, *average_A=False*,
  *interm_bounds=None*, *reference_bounds=None*, *intermediate_constr=None*, *alpha_idx=None*,
  *aux_reference_bounds=None*, *need_A_only=False*, *cutter=None*, *decision_thresh=None*,
  *update_mask=None*, *ibp_nodes=None*, *cache_bounds=False*)[[source]][7][¶][8]*
    Main function for computing bounds.
    
    *Args:*
      x (tuple or None): Input to the model. If it is None, the input from the last forward or
      compute_bounds call is reused. Otherwise: the number of elements in the tuple should be equal
      to the number of input nodes in the model, and each element in the tuple corresponds to the
      value for each input node respectively. It should look similar as the global_input argument
      when used for creating a BoundedModule.
      
      aux (object, optional): Auxliary information that can be passed to Perturbation classes for
      initializing and concretizing bounds, e.g., additional information for supporting synonym word
      subsitution perturbaiton.
      
      C (Tensor): The specification matrix that can map the output of the model with an additional
      linear layer. This is usually used for maping the logits output of the model to classification
      margins.
      
      *method (str): The main method for bound computation. Choices:*
        * IBP: purely use Interval Bound Propagation (IBP) bounds.
        * CROWN-IBP: use IBP to compute intermediate bounds,
        
        but use CROWN (backward mode LiRPA) to compute the bounds of the final node. * CROWN: purely
        use CROWN to compute bounds for intermediate nodes and the final node. * Forward: purely use
        forward mode LiRPA. * Forward+Backward: use forward mode LiRPA for intermediate nodes, but
        further use CROWN for the final node. * CROWN-Optimized or alpha-CROWN: use CROWN, and also
        optimize the linear relaxation parameters for activations. * forward-optimized: use forward
        bounds with optimized linear relaxation. * dynamic-forward: use dynamic forward bound
        propagation where new input variables may be dynamically introduced for nonlinearities. *
        dynamic-forward+backward: use dynamic forward mode for intermediate nodes, but use CROWN for
        the final node.
      
      IBP (bool, optional): If True, use IBP to compute the bounds of intermediate nodes. It can be
      automatically set according to method.
      
      forward (bool, optional): If True, use the forward mode bound propagation to compute the
      bounds of intermediate nodes. It can be automatically set according to method.
      
      bound_lower (bool, default True): If True, the lower bounds of the output needs to be
      computed.
      
      bound_upper (bool, default True): If True, the upper bounds of the output needs to be
      computed.
      
      reuse_ibp (bool, optional): If True and method is None, reuse the previously saved IBP bounds.
      
      final_node_name (str, optional): Set the final node in the computational graph for bound
      computation. By default, the final node of the originally built computational graph is used.
      
      return_A (bool, optional): If True, return linear coefficients in bound propagation (A
      tensors) with needed_A_dict set.
      
      needed_A_dict (dict, optional): A dictionary specifying linear coefficients (A tensors) that
      are needed and should be returned. Each key in the dictionary is the name of a starting node
      in backward bound propagation, with a list as the value for the key, which specifies the names
      of the ending nodes in backward bound propagation, and the linear coefficients of the starting
      node w.r.t. the specified ending nodes are returned. By default, it is empty.
      
      reuse_alpha (bool, optional): If True, reuse previously saved alpha values when they are not
      being optimized.
      
      decision_thresh (float, optional): In CROWN-optimized mode, we will use this decision_thresh
      to dynamically optimize those domains that <= the threshold.
      
      interm_bounds: A dictionary of 2-element tuple/list containing lower and upper bounds for
      intermediate layers. The dictionary keys should include the names of the layers whose bounds
      should be set without recomputation. The layer names can be viewed by setting environment
      variable AUTOLIRPA_DEBUG=1. The values of each dictionary elements are (lower_bounds,
      upper_bounds) where “lower_bounds” and “upper_bounds” are two tensors with the same shape as
      the output shape of this layer. If you only need to set intermediate layer bounds for certain
      layers, then just include these layers’ names in the dictionary.
      
      reference_bounds: Format is similar to “interm_bounds”. However, these bounds are only used as
      a reference, and the bounds for intermediate layers will still be computed (e.g., using CROWN,
      IBP or other specified methods). The computed bounds will be compared to “reference_bounds”
      and the tighter one between the two will be used.
      
      aux_reference_bounds: Format is similar to intermediate layer bounds. However, these bounds
      are only used for determine which neurons are stable and which neurons are unstable for ReLU
      networks. Unstable neurons’ intermediate layer bounds will be recomputed.
      
      cache_bounds: If True, the currently set lower and upper bounds will not be deleted, but
      cached for use by the INVPROP algorithm. This should not be set by the user, but only in
      _get_optimized_bounds.
    *Returns:*
      bound (tuple): When return_A is False, return a tuple of the computed lower bound and upper
      bound. When return_A is True, return a tuple of lower bound, upper bound, and A dictionary.
  
  * save_intermediate(*self*, *save_path=None*)[[source]][9][¶][10]*
    A function for saving intermediate bounds.
    
    Please call this function after compute_bounds, or it will output IBP bounds by default.
    
    *Args:*
      save_path (str, default None): If None, the intermediate bounds will not be saved, or it will
      be saved at the designated path.
    *Returns:*
      save_dict (dict): Return a dictionary of lower and upper bounds, with the key being the name
      of the layer.

* *class *auto_LiRPA.bound_ops.Bound(*attr=None*, *inputs=None*, *output_index=0*,
*options=None*)[[source]][11][¶][12]*
  Base class for supporting the bound computation of an operator. Please see examples at
  auto_LiRPA/operators.
  
  *Args:*
    attr (dict): Attributes of the operator.
    
    inputs (list): A list of input nodes.
    
    output_index (int): The index in the output if the operator has multiple outputs. Usually
    output_index=0.
    
    options (dict): Bound options.
  
  Be sure to run super().__init__(attr, inputs, output_index, options, device) first in the __init__
  function.
  
  * forward(*self*, **x*)[[source]][13][¶][14]*
    Function for standard/clean forward.
    
    *Args:*
      x: A list of input values. The length of the list is equal to the number of input nodes.
    *Returns:*
      output (Tensor): The standard/clean output of this node.
  
  * interval_propagate(*self*, **v*)[[source]][15][¶][16]*
    Function for interval bound propagation (IBP) computation.
    
    There is a default function self.default_interval_propagate(*v) in the base class, which can be
    used if the operator is *monotonic*. To use it, set self.use_default_ibp = True in the __init__
    function, and the implementation of this function can be skipped.
    
    *Args:*
      v: A list of the interval bound of input nodes. Generally, for each element v[i], v[i][0] is
      the lower interval bound, and v[i][1] is the upper interval bound.
    *Returns:*
      bound: The interval bound of this node, in a same format as v[i].
  
  * bound_forward(*self*, *dim_in*, **x*)[[source]][17][¶][18]*
    Function for forward mode bound propagation.
    
    Forward mode LiRPA computs a LinearBound instance representing the linear bound for each
    involved node. Major attributes of LinearBound include lw, uw, lb, ub, lower, and upper.
    
    lw and uw are coefficients of linear bounds w.r.t. model input. Their shape is (batch_size,
    dim_in, *standard_shape), where dim_in is the total dimension of perturbed input nodes of the
    model, and standard_shape is the shape of the standard/clean output. lb and ub are bias terms of
    linear bounds, and their shape is equal to the shape of standard/clean output. lower and upper
    are concretized lower and upper bounds that will be computed later in BoundedModule.
    
    *Args:*
      dim_in (int): Total dimension of perturbed input nodes of the model.
      
      x: A list of the linear bound of input nodes. Each element in x is a LinearBound instance.
    *Returns:*
      bound (LinearBound): The linear bound of this node.
  
  * bound_backward(*self*, *last_lA*, *last_uA*, **x*, ***kwargs*)[[source]][19][¶][20]*
    Function for backward mode bound propagation.
    
    *Args:*
      last_lA (Tensor): A matrix for lower bound computation propagated to this node. It can be None
      if lower bound is not needed.
      
      last_uA (Tensor): A matrix for upper bound computation propagated to this node. It can be None
      if upper bound is not needed.
      
      x: A list of input nodes, with x[i].lower and x[i].upper that can be used as pre-activation
      bounds.
    *Returns:*
      A: A list of A matrices for the input nodes. Each element is a tuple (lA, uA).
      
      lbias (Tensor): The bias term for lower bound computation, introduced by the linear relaxation
      of this node. .
      
      ubias (Tensor): The bias term for upper bound computation, introduced by the linear relaxation
      of this node.

* *class *auto_LiRPA.perturbations.Perturbation[[source]][21][¶][22]*
  Base class for a perturbation specification. Please see examples at auto_LiRPA/perturbations.py.
  
  Examples:
  
  * PerturbationLpNorm: Lp-norm (p>=1) perturbation.
  * PerturbationL0Norm: L0-norm perturbation.
  * PerturbationSynonym: Synonym substitution perturbation for NLP.
  
  * concretize(*self*, *x*, *A*, *sign=-1*, *aux=None*)[[source]][23][¶][24]*
    Concretize bounds according to the perturbation specification.
    
    *Args:*
      x (Tensor): Input before perturbation.
      
      A (Tensor) : A matrix from LiRPA computation.
      
      sign (-1 or +1): If -1, concretize for lower bound; if +1, concretize for upper bound.
      
      aux (object, optional): Auxilary information for concretization.
    *Returns:*
      bound (Tensor): concretized bound with the shape equal to the clean output.
  
  * init(*self*, *x*, *aux=None*, *forward=False*)[[source]][25][¶][26]*
    Initialize bounds before LiRPA computation.
    
    *Args:*
      x (Tensor): Input before perturbation.
      
      aux (object, optional): Auxilary information.
      
      forward (bool): It indicates whether forward mode LiRPA is involved.
    *Returns:*
      bound (LinearBound): Initialized bounds.
      
      center (Tensor): Center of perturbation. It can simply be x, or some other value.
      
      aux (object, optional): Auxilary information. Bound initialization may modify or add auxilary
      information.

## Indices and tables[¶][27]

* [Index][28]
* [Search Page][29]

# [auto_LiRPA][30]

### Navigation

* [Installation][31]
* [Quick Start][32]
* [Examples][33]
* [API Usage][34]
  
  * [`BoundedModule`][35]
    
    * [`BoundedModule.forward()`][36]
    * [`BoundedModule.compute_bounds()`][37]
    * [`BoundedModule.save_intermediate()`][38]
  * [`Bound`][39]
    
    * [`Bound.forward()`][40]
    * [`Bound.interval_propagate()`][41]
    * [`Bound.bound_forward()`][42]
    * [`Bound.bound_backward()`][43]
  * [`Perturbation`][44]
    
    * [`Perturbation.concretize()`][45]
    * [`Perturbation.init()`][46]
  * [Indices and tables][47]
* [Custom Operators][48]
* [Reproducing Our NeurIPS 2020 Paper][49]

### Related Topics

* [Documentation overview][50]
  
  * Previous: [Examples][51]
  * Next: [Custom Operators][52]

### Quick search

©2020-2025, [auto-LiRPA authors][53]. | Powered by [Sphinx 7.4.7][54] & [Alabaster 0.7.16][55] |
[Page source][56]

[1]: #api-usage
[2]: https://github.com/Verified-Intelligence/auto_LiRPA/blob/HEAD/doc//home/docs/checkouts/readthed
ocs.org/user_builds/auto-lirpa/checkouts/latest/doc/../auto_LiRPA/bound_general.py#L40-L1575
[3]: #auto_LiRPA.BoundedModule
[4]: bound_opts.html
[5]: https://github.com/Verified-Intelligence/auto_LiRPA/blob/HEAD/doc//home/docs/checkouts/readthed
ocs.org/user_builds/auto-lirpa/checkouts/latest/doc/../auto_LiRPA/bound_general.py#L495-L523
[6]: #auto_LiRPA.BoundedModule.BoundedModule.forward
[7]: https://github.com/Verified-Intelligence/auto_LiRPA/blob/HEAD/doc//home/docs/checkouts/readthed
ocs.org/user_builds/auto-lirpa/checkouts/latest/doc/../auto_LiRPA/bound_general.py#L1135-L1412
[8]: #auto_LiRPA.BoundedModule.BoundedModule.compute_bounds
[9]: https://github.com/Verified-Intelligence/auto_LiRPA/blob/HEAD/doc//home/docs/checkouts/readthed
ocs.org/user_builds/auto-lirpa/checkouts/latest/doc/../auto_LiRPA/bound_general.py#L1414-L1446
[10]: #auto_LiRPA.BoundedModule.BoundedModule.save_intermediate
[11]: https://github.com/Verified-Intelligence/auto_LiRPA/blob/HEAD/doc//home/docs/checkouts/readthe
docs.org/user_builds/auto-lirpa/checkouts/latest/doc/../auto_LiRPA/operators/base.py#L105-L635
[12]: #auto_LiRPA.bound_ops.Bound
[13]: https://github.com/Verified-Intelligence/auto_LiRPA/blob/HEAD/doc//home/docs/checkouts/readthe
docs.org/user_builds/auto-lirpa/checkouts/latest/doc/../auto_LiRPA/operators/base.py#L314-L324
[14]: #auto_LiRPA.bound_ops.Bound.Bound.forward
[15]: https://github.com/Verified-Intelligence/auto_LiRPA/blob/HEAD/doc//home/docs/checkouts/readthe
docs.org/user_builds/auto-lirpa/checkouts/latest/doc/../auto_LiRPA/operators/base.py#L326-L345
[16]: #auto_LiRPA.bound_ops.Bound.Bound.interval_propagate
[17]: https://github.com/Verified-Intelligence/auto_LiRPA/blob/HEAD/doc//home/docs/checkouts/readthe
docs.org/user_builds/auto-lirpa/checkouts/latest/doc/../auto_LiRPA/operators/base.py#L363-L389
[18]: #auto_LiRPA.bound_ops.Bound.Bound.bound_forward
[19]: https://github.com/Verified-Intelligence/auto_LiRPA/blob/HEAD/doc//home/docs/checkouts/readthe
docs.org/user_builds/auto-lirpa/checkouts/latest/doc/../auto_LiRPA/operators/base.py#L394-L412
[20]: #auto_LiRPA.bound_ops.Bound.Bound.bound_backward
[21]: https://github.com/Verified-Intelligence/auto_LiRPA/blob/HEAD/doc//home/docs/checkouts/readthe
docs.org/user_builds/auto-lirpa/checkouts/latest/doc/../auto_LiRPA/perturbations.py#L27-L84
[22]: #auto_LiRPA.perturbations.Perturbation
[23]: https://github.com/Verified-Intelligence/auto_LiRPA/blob/HEAD/doc//home/docs/checkouts/readthe
docs.org/user_builds/auto-lirpa/checkouts/latest/doc/../auto_LiRPA/perturbations.py#L47-L63
[24]: #auto_LiRPA.perturbations.Perturbation.Perturbation.concretize
[25]: https://github.com/Verified-Intelligence/auto_LiRPA/blob/HEAD/doc//home/docs/checkouts/readthe
docs.org/user_builds/auto-lirpa/checkouts/latest/doc/../auto_LiRPA/perturbations.py#L65-L84
[26]: #auto_LiRPA.perturbations.Perturbation.Perturbation.init
[27]: #indices-and-tables
[28]: genindex.html
[29]: search.html
[30]: index.html
[31]: installation.html
[32]: quick-start.html
[33]: examples.html
[34]: #
[35]: #auto_LiRPA.BoundedModule
[36]: #auto_LiRPA.BoundedModule.BoundedModule.forward
[37]: #auto_LiRPA.BoundedModule.BoundedModule.compute_bounds
[38]: #auto_LiRPA.BoundedModule.BoundedModule.save_intermediate
[39]: #auto_LiRPA.bound_ops.Bound
[40]: #auto_LiRPA.bound_ops.Bound.Bound.forward
[41]: #auto_LiRPA.bound_ops.Bound.Bound.interval_propagate
[42]: #auto_LiRPA.bound_ops.Bound.Bound.bound_forward
[43]: #auto_LiRPA.bound_ops.Bound.Bound.bound_backward
[44]: #auto_LiRPA.perturbations.Perturbation
[45]: #auto_LiRPA.perturbations.Perturbation.Perturbation.concretize
[46]: #auto_LiRPA.perturbations.Perturbation.Perturbation.init
[47]: #indices-and-tables
[48]: custom_op.html
[49]: paper.html
[50]: index.html
[51]: examples.html
[52]: custom_op.html
[53]: https://github.com/Verified-Intelligence/auto_LiRPA#developers-and-copyright
[54]: https://www.sphinx-doc.org/
[55]: https://alabaster.readthedocs.io
[56]: _sources/api.rst.txt
