Evaluate documentation

Main classes

# Evaluate

🏡 View all docsAWS Trainium & InferentiaAccelerateArgillaAutoTrainBitsandbytesChat UIDataset
viewerDatasetsDeploying on AWSDiffusersDistilabelEvaluateGoogle CloudGoogle TPUsGradioHubHub Python
LibraryHuggingface.jsInference Endpoints (dedicated)Inference
ProvidersKernelsLeRobotLeaderboardsLightevalMicrosoft AzureOptimumPEFTSafetensorsSentence
TransformersTRLTasksText Embeddings InferenceText Generation
InferenceTokenizersTrackioTransformersTransformers.jssmolagentstimm
Search documentation
mainv0.4.6v0.3.0v0.2.3v0.1.2 EN
Get started
[🤗 Evaluate ][1]
Tutorials
[Installation ][2][A quick tour ][3]
How-to guides
[Choosing the right metric ][4][Adding new evaluations ][5][Using the evaluator ][6][Using the
evaluator with custom pipelines ][7][Creating an EvaluationSuite ][8]
Using 🤗 Evaluate with other ML frameworks
[Transformers ][9][Keras and Tensorflow ][10][scikit-learn ][11]
Conceptual guides
[Types of evaluations ][12][Considerations for model evaluation ][13]
Reference
[Main classes ][14][Loading methods ][15][Saving methods ][16][Hub methods ][17][Evaluator classes
][18][Visualization methods ][19][Logging methods ][20]
[Hugging Face's logo]
Join the Hugging Face community

and get access to the augmented documentation experience

Collaborate on models, datasets and Spaces
Faster examples with accelerated inference
Switch between documentation themes
[Sign Up][21]

to get started

# Main classes

## EvaluationModuleInfo

The base class `EvaluationModuleInfo` implements a the logic for the subclasses `MetricInfo`,
`ComparisonInfo`, and `MeasurementInfo`.

### class evaluate.EvaluationModuleInfo

[< source >][22]

( description: str citation: str features: typing.Union[datasets.features.features.Features,
typing.List[datasets.features.features.Features]] inputs_description: str = <factory> homepage: str
= <factory> license: str = <factory> codebase_urls: typing.List[str] = <factory> reference_urls:
typing.List[str] = <factory> streamable: bool = False format: typing.Optional[str] = None
module_type: str = 'metric' module_name: typing.Optional[str] = None config_name:
typing.Optional[str] = None experiment_id: typing.Optional[str] = None )

Base class to store information about an evaluation used for `MetricInfo`, `ComparisonInfo`, and
`MeasurementInfo`.

`EvaluationModuleInfo` documents an evaluation, including its name, version, and features. See the
constructor arguments and properties for a full list.

Note: Not all fields are known on construction and may be updated later.

#### from_directory

[< source >][23]

( metric_info_dir )

Parameters

* **metric_info_dir** (`str`) — The directory containing the `metric_info` JSON file. This should be
  the root directory of a specific metric version.

Create `EvaluationModuleInfo` from the JSON file in `metric_info_dir`.

Example:

Copied
>>> my_metric = EvaluationModuleInfo.from_directory("/path/to/directory/")

#### write_to_directory

[< source >][24]

( metric_info_dir )

Parameters

* **metric_info_dir** (`str`) — The directory to save `metric_info_dir` to.

Write `EvaluationModuleInfo` as JSON to `metric_info_dir`. Also save the license separately in
LICENSE.

Example:

Copied
>>> my_metric.info.write_to_directory("/path/to/directory/")

### class evaluate.MetricInfo

[< source >][25]

( description: str citation: str features: typing.Union[datasets.features.features.Features,
typing.List[datasets.features.features.Features]] inputs_description: str = <factory> homepage: str
= <factory> license: str = <factory> codebase_urls: typing.List[str] = <factory> reference_urls:
typing.List[str] = <factory> streamable: bool = False format: typing.Optional[str] = None
module_type: str = 'metric' module_name: typing.Optional[str] = None config_name:
typing.Optional[str] = None experiment_id: typing.Optional[str] = None )

Information about a metric.

`EvaluationModuleInfo` documents a metric, including its name, version, and features. See the
constructor arguments and properties for a full list.

Note: Not all fields are known on construction and may be updated later.

### class evaluate.ComparisonInfo

[< source >][26]

( description: str citation: str features: typing.Union[datasets.features.features.Features,
typing.List[datasets.features.features.Features]] inputs_description: str = <factory> homepage: str
= <factory> license: str = <factory> codebase_urls: typing.List[str] = <factory> reference_urls:
typing.List[str] = <factory> streamable: bool = False format: typing.Optional[str] = None
module_type: str = 'comparison' module_name: typing.Optional[str] = None config_name:
typing.Optional[str] = None experiment_id: typing.Optional[str] = None )

Information about a comparison.

`EvaluationModuleInfo` documents a comparison, including its name, version, and features. See the
constructor arguments and properties for a full list.

Note: Not all fields are known on construction and may be updated later.

### class evaluate.MeasurementInfo

[< source >][27]

( description: str citation: str features: typing.Union[datasets.features.features.Features,
typing.List[datasets.features.features.Features]] inputs_description: str = <factory> homepage: str
= <factory> license: str = <factory> codebase_urls: typing.List[str] = <factory> reference_urls:
typing.List[str] = <factory> streamable: bool = False format: typing.Optional[str] = None
module_type: str = 'measurement' module_name: typing.Optional[str] = None config_name:
typing.Optional[str] = None experiment_id: typing.Optional[str] = None )

Information about a measurement.

`EvaluationModuleInfo` documents a measurement, including its name, version, and features. See the
constructor arguments and properties for a full list.

Note: Not all fields are known on construction and may be updated later.

## EvaluationModule

The base class `EvaluationModule` implements a the logic for the subclasses `Metric`, `Comparison`,
and `Measurement`.

### class evaluate.EvaluationModule

[< source >][28]

( config_name: typing.Optional[str] = None keep_in_memory: bool = False cache_dir:
typing.Optional[str] = None num_process: int = 1 process_id: int = 0 seed: typing.Optional[int] =
None experiment_id: typing.Optional[str] = None hash: str = None max_concurrent_cache_files: int =
10000 timeout: typing.Union[int, float] = 100 **kwargs )

Parameters

* **config_name** (`str`) — This is used to define a hash specific to a module computation script
  and prevents the module’s data to be overridden when the module loading script is modified.
* **keep_in_memory** (`bool`) — Keep all predictions and references in memory. Not possible in
  distributed settings.
* **cache_dir** (`str`) — Path to a directory in which temporary prediction/references data will be
  stored. The data directory should be located on a shared file-system in distributed setups.
* **num_process** (`int`) — Specify the total number of nodes in a distributed settings. This is
  useful to compute module in distributed setups (in particular non-additive modules like F1).
* **process_id** (`int`) — Specify the id of the current process in a distributed setup (between 0
  and num_process-1) This is useful to compute module in distributed setups (in particular
  non-additive metrics like F1).
* **seed** (`int`, optional) — If specified, this will temporarily set numpy’s random seed when
  [compute()][29] is run.
* **experiment_id** (`str`) — A specific experiment id. This is used if several distributed
  evaluations share the same file system. This is useful to compute module in distributed setups (in
  particular non-additive metrics like F1).
* **hash** (`str`) — Used to identify the evaluation module according to the hashed file contents.
* **max_concurrent_cache_files** (`int`) — Max number of concurrent module cache files (default
  `10000`).
* **timeout** (`Union[int, float]`) — Timeout in second for distributed setting synchronization.

A `EvaluationModule` is the base class and common API for metrics, comparisons, and measurements.

#### add

[< source >][30]

( prediction = None reference = None **kwargs )

Parameters

* **prediction** (`list/array/tensor`, *optional*) — Predictions.
* **reference** (`list/array/tensor`, *optional*) — References.

Add one prediction and reference for the evaluation module’s stack.

Example:

Copied
>>> import evaluate
>>> accuracy = evaluate.load("accuracy")
>>> accuracy.add(references=[0,1], predictions=[1,0])

#### add_batch

[< source >][31]

( predictions = None references = None **kwargs )

Parameters

* **predictions** (`list/array/tensor`, *optional*) — Predictions.
* **references** (`list/array/tensor`, *optional*) — References.

Add a batch of predictions and references for the evaluation module’s stack.

Example:

Copied
>>> import evaluate
>>> accuracy = evaluate.load("accuracy")
>>> for refs, preds in zip([[0,1],[0,1]], [[1,0],[0,1]]):
...     accuracy.add_batch(references=refs, predictions=preds)

#### compute

[< source >][32]

( predictions = None references = None **kwargs ) → `dict` or `None`

Parameters

* **predictions** (`list/array/tensor`, *optional*) — Predictions.
* **references** (`list/array/tensor`, *optional*) — References.
* ****kwargs** (optional) — Keyword arguments that will be forwarded to the evaluation module
  [compute()][33] method (see details in the docstring).

Returns

`dict` or `None`


* Dictionary with the results if this evaluation module is run on the main process (`process_id ==
  0`).
* `None` if the evaluation module is not run on the main process (`process_id != 0`).

Compute the evaluation module.

Usage of positional arguments is not allowed to prevent mistakes.

Copied
>>> import evaluate
>>> accuracy =  evaluate.load("accuracy")
>>> accuracy.compute(predictions=[0, 1, 1, 0], references=[0, 1, 0, 1])

#### download_and_prepare

[< source >][34]

( download_config: typing.Optional[datasets.download.download_config.DownloadConfig] = None
dl_manager: typing.Optional[datasets.download.download_manager.DownloadManager] = None )

Parameters

* **download_config** (`DownloadConfig`, *optional*) — Specific download configuration parameters.
* **dl_manager** (`DownloadManager`, *optional*) — Specific download manager to use.

Downloads and prepares evaluation module for reading.

Example:

Copied
>>> import evaluate

### class evaluate.Metric

[< source >][35]

( config_name: typing.Optional[str] = None keep_in_memory: bool = False cache_dir:
typing.Optional[str] = None num_process: int = 1 process_id: int = 0 seed: typing.Optional[int] =
None experiment_id: typing.Optional[str] = None hash: str = None max_concurrent_cache_files: int =
10000 timeout: typing.Union[int, float] = 100 **kwargs )

Parameters

* **config_name** (`str`) — This is used to define a hash specific to a metric computation script
  and prevents the metric’s data to be overridden when the metric loading script is modified.
* **keep_in_memory** (`bool`) — Keep all predictions and references in memory. Not possible in
  distributed settings.
* **cache_dir** (`str`) — Path to a directory in which temporary prediction/references data will be
  stored. The data directory should be located on a shared file-system in distributed setups.
* **num_process** (`int`) — Specify the total number of nodes in a distributed settings. This is
  useful to compute metrics in distributed setups (in particular non-additive metrics like F1).
* **process_id** (`int`) — Specify the id of the current process in a distributed setup (between 0
  and num_process-1) This is useful to compute metrics in distributed setups (in particular
  non-additive metrics like F1).
* **seed** (`int`, *optional*) — If specified, this will temporarily set numpy’s random seed when
  [compute()][36] is run.
* **experiment_id** (`str`) — A specific experiment id. This is used if several distributed
  evaluations share the same file system. This is useful to compute metrics in distributed setups
  (in particular non-additive metrics like F1).
* **max_concurrent_cache_files** (`int`) — Max number of concurrent metric cache files (default
  `10000`).
* **timeout** (`Union[int, float]`) — Timeout in second for distributed setting synchronization.

A Metric is the base class and common API for all metrics.

### class evaluate.Comparison

[< source >][37]

( config_name: typing.Optional[str] = None keep_in_memory: bool = False cache_dir:
typing.Optional[str] = None num_process: int = 1 process_id: int = 0 seed: typing.Optional[int] =
None experiment_id: typing.Optional[str] = None hash: str = None max_concurrent_cache_files: int =
10000 timeout: typing.Union[int, float] = 100 **kwargs )

Parameters

* **config_name** (`str`) — This is used to define a hash specific to a comparison computation
  script and prevents the comparison’s data to be overridden when the comparison loading script is
  modified.
* **keep_in_memory** (`bool`) — Keep all predictions and references in memory. Not possible in
  distributed settings.
* **cache_dir** (`str`) — Path to a directory in which temporary prediction/references data will be
  stored. The data directory should be located on a shared file-system in distributed setups.
* **num_process** (`int`) — Specify the total number of nodes in a distributed settings. This is
  useful to compute comparisons in distributed setups (in particular non-additive comparisons).
* **process_id** (`int`) — Specify the id of the current process in a distributed setup (between 0
  and num_process-1) This is useful to compute comparisons in distributed setups (in particular
  non-additive comparisons).
* **seed** (`int`, *optional*) — If specified, this will temporarily set numpy’s random seed when
  [compute()][38] is run.
* **experiment_id** (`str`) — A specific experiment id. This is used if several distributed
  evaluations share the same file system. This is useful to compute comparisons in distributed
  setups (in particular non-additive comparisons).
* **max_concurrent_cache_files** (`int`) — Max number of concurrent comparison cache files (default
  `10000`).
* **timeout** (`Union[int, float]`) — Timeout in second for distributed setting synchronization.

A Comparison is the base class and common API for all comparisons.

### class evaluate.Measurement

[< source >][39]

( config_name: typing.Optional[str] = None keep_in_memory: bool = False cache_dir:
typing.Optional[str] = None num_process: int = 1 process_id: int = 0 seed: typing.Optional[int] =
None experiment_id: typing.Optional[str] = None hash: str = None max_concurrent_cache_files: int =
10000 timeout: typing.Union[int, float] = 100 **kwargs )

Parameters

* **config_name** (`str`) — This is used to define a hash specific to a measurement computation
  script and prevents the measurement’s data to be overridden when the measurement loading script is
  modified.
* **keep_in_memory** (`bool`) — Keep all predictions and references in memory. Not possible in
  distributed settings.
* **cache_dir** (`str`) — Path to a directory in which temporary prediction/references data will be
  stored. The data directory should be located on a shared file-system in distributed setups.
* **num_process** (`int`) — Specify the total number of nodes in a distributed settings. This is
  useful to compute measurements in distributed setups (in particular non-additive measurements).
* **process_id** (`int`) — Specify the id of the current process in a distributed setup (between 0
  and num_process-1) This is useful to compute measurements in distributed setups (in particular
  non-additive measurements).
* **seed** (`int`, *optional*) — If specified, this will temporarily set numpy’s random seed when
  [compute()][40] is run.
* **experiment_id** (`str`) — A specific experiment id. This is used if several distributed
  evaluations share the same file system. This is useful to compute measurements in distributed
  setups (in particular non-additive measurements).
* **max_concurrent_cache_files** (`int`) — Max number of concurrent measurement cache files (default
  `10000`).
* **timeout** (`Union[int, float]`) — Timeout in second for distributed setting synchronization.

A Measurement is the base class and common API for all measurements.

## CombinedEvaluations

The `combine` function allows to combine multiple `EvaluationModule`s into a single
`CombinedEvaluations`.

#### evaluate.combine

[< source >][41]

( evaluations force_prefix = False )

Parameters

* **evaluations** (`Union[list, dict]`) — A list or dictionary of evaluation modules. The modules
  can either be passed as strings or loaded `EvaluationModule`s. If a dictionary is passed its keys
  are the names used and the values the modules. The names are used as prefix in case there are name
  overlaps in the returned results of each module or if `force_prefix=True`.
* **force_prefix** (`bool`, *optional*, defaults to `False`) — If `True` all scores from the modules
  are prefixed with their name. If a dictionary is passed the keys are used as name otherwise the
  module’s name.

Combines several metrics, comparisons, or measurements into a single `CombinedEvaluations` object
that can be used like a single evaluation module.

If two scores have the same name, then they are prefixed with their module names. And if two modules
have the same name, please use a dictionary to give them different names, otherwise an integer id is
appended to the prefix.

Examples:

Copied
>>> import evaluate
>>> accuracy = evaluate.load("accuracy")
>>> f1 = evaluate.load("f1")
>>> clf_metrics = combine(["accuracy", "f1"])

### class evaluate.CombinedEvaluations

[< source >][42]

( evaluation_modules force_prefix = False )

#### add

[< source >][43]

( prediction = None reference = None **kwargs )

Parameters

* **predictions** (`list/array/tensor`, *optional*) — Predictions.
* **references** (`list/array/tensor`, *optional*) — References.

Add one prediction and reference for each evaluation module’s stack.

Example:

Copied
>>> import evaluate
>>> accuracy = evaluate.load("accuracy")
>>> f1 = evaluate.load("f1")
>>> clf_metrics = combine(["accuracy", "f1"])
>>> for ref, pred in zip([0,1,0,1], [1,0,0,1]):
...     clf_metrics.add(references=ref, predictions=pred)

#### add_batch

[< source >][44]

( predictions = None references = None **kwargs )

Parameters

* **predictions** (`list/array/tensor`, *optional*) — Predictions.
* **references** (`list/array/tensor`, *optional*) — References.

Add a batch of predictions and references for each evaluation module’s stack.

Example:

Copied
>>> import evaluate
>>> accuracy = evaluate.load("accuracy")
>>> f1 = evaluate.load("f1")
>>> clf_metrics = combine(["accuracy", "f1"])
>>> for refs, preds in zip([[0,1],[0,1]], [[1,0],[0,1]]):
...     clf_metrics.add(references=refs, predictions=preds)

#### compute

[< source >][45]

( predictions = None references = None **kwargs ) → `dict` or `None`

Parameters

* **predictions** (`list/array/tensor`, *optional*) — Predictions.
* **references** (`list/array/tensor`, *optional*) — References.
* ****kwargs** (*optional*) — Keyword arguments that will be forwarded to the evaluation module
  [compute()][46] method (see details in the docstring).

Returns

`dict` or `None`


* Dictionary with the results if this evaluation module is run on the main process (`process_id ==
  0`).
* `None` if the evaluation module is not run on the main process (`process_id != 0`).

Compute each evaluation module.

Usage of positional arguments is not allowed to prevent mistakes.

Example:

Copied
>>> import evaluate
>>> accuracy = evaluate.load("accuracy")
>>> f1 = evaluate.load("f1")
>>> clf_metrics = combine(["accuracy", "f1"])
>>> clf_metrics.compute(predictions=[0,1], references=[1,1])
{'accuracy': 0.5, 'f1': 0.6666666666666666}
[< > Update on GitHub][47]
[←Considerations for model evaluation][48] [Loading methods→][49]
[Main classes][50] [EvaluationModuleInfo][51] [EvaluationModule][52] [CombinedEvaluations][53]

[1]: /docs/evaluate/index
[2]: /docs/evaluate/installation
[3]: /docs/evaluate/a_quick_tour
[4]: /docs/evaluate/choosing_a_metric
[5]: /docs/evaluate/creating_and_sharing
[6]: /docs/evaluate/base_evaluator
[7]: /docs/evaluate/custom_evaluator
[8]: /docs/evaluate/evaluation_suite
[9]: /docs/evaluate/transformers_integrations
[10]: /docs/evaluate/keras_integrations
[11]: /docs/evaluate/sklearn_integrations
[12]: /docs/evaluate/types_of_evaluations
[13]: /docs/evaluate/considerations
[14]: /docs/evaluate/package_reference/main_classes
[15]: /docs/evaluate/package_reference/loading_methods
[16]: /docs/evaluate/package_reference/saving_methods
[17]: /docs/evaluate/package_reference/hub_methods
[18]: /docs/evaluate/package_reference/evaluator_classes
[19]: /docs/evaluate/package_reference/visualization_methods
[20]: /docs/evaluate/package_reference/logging_methods
[21]: /join
[22]: https://github.com/huggingface/evaluate/blob/v0.4.6/src/evaluate/info.py#L35
[23]: https://github.com/huggingface/evaluate/blob/v0.4.6/src/evaluate/info.py#L92
[24]: https://github.com/huggingface/evaluate/blob/v0.4.6/src/evaluate/info.py#L72
[25]: https://github.com/huggingface/evaluate/blob/v0.4.6/src/evaluate/info.py#L122
[26]: https://github.com/huggingface/evaluate/blob/v0.4.6/src/evaluate/info.py#L135
[27]: https://github.com/huggingface/evaluate/blob/v0.4.6/src/evaluate/info.py#L148
[28]: https://github.com/huggingface/evaluate/blob/v0.4.6/src/evaluate/module.py#L149
[29]: /docs/evaluate/v0.4.6/en/package_reference/main_classes#evaluate.EvaluationModule.compute
[30]: https://github.com/huggingface/evaluate/blob/v0.4.6/src/evaluate/module.py#L548
[31]: https://github.com/huggingface/evaluate/blob/v0.4.6/src/evaluate/module.py#L488
[32]: https://github.com/huggingface/evaluate/blob/v0.4.6/src/evaluate/module.py#L415
[33]: /docs/evaluate/v0.4.6/en/package_reference/main_classes#evaluate.EvaluationModule.compute
[34]: https://github.com/huggingface/evaluate/blob/v0.4.6/src/evaluate/module.py#L677
[35]: https://github.com/huggingface/evaluate/blob/v0.4.6/src/evaluate/module.py#L782
[36]: /docs/evaluate/v0.4.6/en/package_reference/main_classes#evaluate.EvaluationModule.compute
[37]: https://github.com/huggingface/evaluate/blob/v0.4.6/src/evaluate/module.py#L812
[38]: /docs/evaluate/v0.4.6/en/package_reference/main_classes#evaluate.EvaluationModule.compute
[39]: https://github.com/huggingface/evaluate/blob/v0.4.6/src/evaluate/module.py#L842
[40]: /docs/evaluate/v0.4.6/en/package_reference/main_classes#evaluate.EvaluationModule.compute
[41]: https://github.com/huggingface/evaluate/blob/v0.4.6/src/evaluate/module.py#L1008
[42]: https://github.com/huggingface/evaluate/blob/v0.4.6/src/evaluate/module.py#L872
[43]: https://github.com/huggingface/evaluate/blob/v0.4.6/src/evaluate/module.py#L895
[44]: https://github.com/huggingface/evaluate/blob/v0.4.6/src/evaluate/module.py#L920
[45]: https://github.com/huggingface/evaluate/blob/v0.4.6/src/evaluate/module.py#L944
[46]: /docs/evaluate/v0.4.6/en/package_reference/main_classes#evaluate.EvaluationModule.compute
[47]: https://github.com/huggingface/evaluate/blob/main/docs/source/package_reference/main_classes.m
dx
[48]: /docs/evaluate/considerations
[49]: /docs/evaluate/package_reference/loading_methods
[50]: #main-classes
[51]: #evaluate.EvaluationModuleInfo
[52]: #evaluate.EvaluationModule
[53]: #evaluate.combine
