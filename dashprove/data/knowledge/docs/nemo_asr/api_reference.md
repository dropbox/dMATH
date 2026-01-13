# NeMo Core APIs[#][1]

## Base class for all NeMo models[#][2]

## Base Neural Module class[#][3]

## Base Mixin classes[#][4]

## Base Connector classes[#][5]

* *class *nemo.core.connectors.save_restore_connector.SaveRestoreConnector[#][6]*
  Bases: `object`
  
  Connector for saving and restoring models.
  
  * save_to(
  
    *model: nemo_classes.ModelPT*,
    *save_path: str*,
  )[#][7]*
    Saves model instance (weights and configuration) into .nemo file. You can use “restore_from”
    method to fully restore instance from .nemo file.
    
    *.nemo file is an archive (tar.gz) with the following:*
      *model_config.yaml - model configuration in .yaml format.*
        You can deserialize this into cfg argument for model’s constructor
      
      model_wights.ckpt - model checkpoint
    
    *Parameters:*
      * **model** – ModelPT object to be saved.
      * **save_path** – Path to .nemo file where model instance should be saved
    *Returns:*
      *Path to .nemo file where model instance was saved (same as save_path argument) or None if not
      rank 0*
        The path can be a directory if the flag pack_nemo_file is set to False.
    *Return type:*
      str
  
  * load_config_and_state_dict(
  
    *calling_cls*,
    *restore_path: str*,
    *override_config_path: omegaconf.OmegaConf | str | None = None*,
    *map_location: torch.device | None = None*,
    *strict: bool = True*,
    *return_config: bool = False*,
    *trainer: lightning.pytorch.trainer.trainer.Trainer | None = None*,
    *validate_access_integrity: bool = True*,
  )[#][8]*
    Restores model instance (weights and configuration) into .nemo file
    
    *Parameters:*
      * **restore_path** – path to .nemo file from which model should be instantiated
      * **override_config_path** – path to a yaml config that will override the internal config file
        or an OmegaConf / DictConfig object representing the model config.
      * **map_location** – Optional torch.device() to map the instantiated model to a device. By
        default (None), it will select a GPU if available, falling back to CPU otherwise.
      * **strict** – Passed to load_state_dict. By default True
      * **return_config** – If set to true, will return just the underlying config of the restored
        model as an OmegaConf DictConfig object without instantiating the model.
    
    Example
    
    `` model = nemo.collections.asr.models.EncDecCTCModel.restore_from('asr.nemo') assert
    isinstance(model, nemo.collections.asr.models.EncDecCTCModel) ``
    
    *Returns:*
      An instance of type cls or its underlying config (if return_config is set).
  
  * modify_state_dict(*conf*, *state_dict*)[#][9]*
    Utility method that allows to modify the state dict before loading parameters into a model.
    :param conf: A model level OmegaConf object. :param state_dict: The state dict restored from the
    checkpoint.
    
    *Returns:*
      A potentially modified state dict.
  
  * load_instance_with_state_dict(
  
    *instance*,
    *state_dict*,
    *strict*,
  )[#][10]*
    Utility method that loads a model instance with the (potentially modified) state dict.
    
    *Parameters:*
      * **instance** – ModelPT subclass instance.
      * **state_dict** – The state dict (which may have been modified)
      * **strict** – Bool, whether to perform strict checks when loading the state dict.
  
  * restore_from(
  
    *calling_cls*,
    *restore_path: str*,
    *override_config_path: omegaconf.OmegaConf | str | None = None*,
    *map_location: torch.device | None = None*,
    *strict: bool = True*,
    *return_config: bool = False*,
    *trainer: lightning.pytorch.trainer.trainer.Trainer | None = None*,
    *validate_access_integrity: bool = True*,
  )[#][11]*
    Restores model instance (weights and configuration) into .nemo file
    
    *Parameters:*
      * **restore_path** – path to .nemo file from which model should be instantiated
      * **override_config_path** – path to a yaml config that will override the internal config file
        or an OmegaConf / DictConfig object representing the model config.
      * **map_location** – Optional torch.device() to map the instantiated model to a device. By
        default (None), it will select a GPU if available, falling back to CPU otherwise.
      * **strict** – Passed to load_state_dict. By default True
      * **return_config** – If set to true, will return just the underlying config of the restored
        model as an OmegaConf DictConfig object without instantiating the model.
      * **trainer** – An optional Trainer object, passed to the model constructor.
    
    Example
    
    `` model = nemo.collections.asr.models.EncDecCTCModel.restore_from('asr.nemo') assert
    isinstance(model, nemo.collections.asr.models.EncDecCTCModel) ``
    
    *Returns:*
      An instance of type cls or its underlying config (if return_config is set).
  
  * extract_state_dict_from(
  
    *restore_path: str*,
    *save_dir: str*,
    *split_by_module: bool = False*,
  )[#][12]*
    Extract the state dict(s) from a provided .nemo tarfile and save it to a directory.
    
    *Parameters:*
      * **restore_path** – path to .nemo file from which state dict(s) should be extracted
      * **save_dir** – directory in which the saved state dict(s) should be stored
      * **split_by_module** – bool flag, which determins whether the output checkpoint should be for
        the entire Model, or the individual module’s that comprise the Model
    
    Example
    
    To convert the .nemo tarfile into a single Model level PyTorch checkpoint :: state_dict =
    nemo.collections.asr.models.EncDecCTCModel.extract_state_dict_from(‘asr.nemo’, ‘./asr_ckpts’)
    
    To restore a model from a Model level checkpoint :: model =
    nemo.collections.asr.models.EncDecCTCModel(cfg) # or any other method of restoration
    model.load_state_dict(torch.load(“./asr_ckpts/model_weights.ckpt”))
    
    To convert the .nemo tarfile into multiple Module level PyTorch checkpoints :: state_dict =
    nemo.collections.asr.models.EncDecCTCModel.extract_state_dict_from(
    
    > ‘asr.nemo’, ‘./asr_ckpts’, split_by_module=True
    
    )
    
    To restore a module from a Module level checkpoint :: model =
    nemo.collections.asr.models.EncDecCTCModel(cfg) # or any other method of restoration
    
    # load the individual components
    model.preprocessor.load_state_dict(torch.load(“./asr_ckpts/preprocessor.ckpt”))
    model.encoder.load_state_dict(torch.load(“./asr_ckpts/encoder.ckpt”))
    model.decoder.load_state_dict(torch.load(“./asr_ckpts/decoder.ckpt”))
    
    *Returns:*
      The state dict that was loaded from the original .nemo checkpoint
  
  * register_artifact(
  
    *model*,
    *config_path: str*,
    *src: str*,
    *verify_src_exists: bool = True*,
  )[#][13]*
    Register model artifacts with this function. These artifacts (files) will be included inside
    .nemo file when model.save_to(“mymodel.nemo”) is called.
    
    How it works:
    
    1. *It always returns existing absolute path which can be used during Model constructor call*
         EXCEPTION: src is None or “” in which case nothing will be done and src will be returned
    2. It will add (config_path, model_utils.ArtifactItem()) pair to self.artifacts
       
       > If "src" is local existing path:
       >     then it will be returned in absolute path form
       > elif "src" starts with "nemo_file:unique_artifact_name":
       >     .nemo will be untarred to a temporary folder location and an actual existing path will 
       > be returned
       > else:
       >     an error will be raised.
    
    WARNING: use .register_artifact calls in your models’ constructors. The returned path is not
    guaranteed to exist after you have exited your model’s constructor.
    
    *Parameters:*
      * **model** – ModelPT object to register artifact for.
      * **config_path** (*str*) – Artifact key. Usually corresponds to the model config.
      * **src** (*str*) – Path to artifact.
      * **verify_src_exists** (*bool*) – If set to False, then the artifact is optional and
        register_artifact will return None even if src is not found. Defaults to True.
    *Returns:*
      *If src is not None or empty it always returns absolute path which is guaranteed to exists
      during model*
        instance life
    *Return type:*
      str
  
  * *property *model_config_yaml*: str*[#][14]*
    Get the path to the model config yaml file.
  
  * *property *model_weights_ckpt*: str*[#][15]*
    Get the path to the model weights checkpoint file.
  
  * *property *model_extracted_dir*: str | None*[#][16]*
    Get the path to the model extracted directory.
  
  * *property *pack_nemo_file*: bool*[#][17]*
    Get the flag for packing a nemo file.

## Base Mixin Classes[#][18]

* *class *nemo.core.classes.mixins.access_mixins.AccessMixin[#][19]*
  Bases: `ABC`
  
  Allows access to output of intermediate layers of a model
  
  * register_accessible_tensor(*name*, *tensor*)[#][20]*
    Register tensor for later use.
  
  * *classmethod *get_module_registry(*module: torch.nn.Module*)[#][21]*
    Extract all registries from named submodules, return dictionary where the keys are the flattened
    module names, the values are the internal registry of each such module.
  
  * reset_registry(*registry_key: str | None = None*)[#][22]*
    Reset the registries of all named sub-modules
  
  * *property *access_cfg[#][23]*
    Returns: The global access config shared across all access mixin modules.

* *class *nemo.core.classes.mixins.hf_io_mixin.HuggingFaceFileIO[#][24]*
  Bases: `ABC`
  
  Mixin that provides Hugging Face file IO functionality for NeMo models. It is usually implemented
  as a mixin to ModelPT.
  
  This mixin provides the following functionality: - search_huggingface_models(): Search the hub
  programmatically via some model filter. - push_to_hf_hub(): Push a model to the hub.
  
  * *classmethod *get_hf_model_filter() → Dict[str, Any][#][25]*
    Generates a filter for HuggingFace models.
    
    Additionaly includes default values of some metadata about results returned by the Hub.
    
    *Metadata:*
      resolve_card_info: Bool flag, if set, returns the model card metadata. Default: False.
      limit_results: Optional int, limits the number of results returned.
    
    *Returns:*
      A dict representing the arguments passable to huggingface list_models().
  
  * *classmethod *search_huggingface_models(
  
    *model_filter: Dict[str, Any] | None = None*,
  ) → Iterable[huggingface_hub.hf_api.ModelInfo][#][26]*
    Should list all pre-trained models available via Hugging Face Hub.
    
    The following metadata can be passed via the model_filter for additional results. Metadata:
    
    > resolve_card_info: Bool flag, if set, returns the model card metadata. Default: False.
    > 
    > limit_results: Optional int, limits the number of results returned.
    
    # You can replace <DomainSubclass> with any subclass of ModelPT.
    from nemo.core import ModelPT
    
    # Get default filter dict
    filt = <DomainSubclass>.get_hf_model_filter()
    
    # Make any modifications to the filter as necessary
    filt['language'] = [...]
    filt['task'] = ...
    filt['tags'] = [...]
    
    # Add any metadata to the filter as needed (kwargs to list_models)
    filt['limit'] = 5
    
    # Obtain model info
    model_infos = <DomainSubclass>.search_huggingface_models(model_filter=filt)
    
    # Browse through cards and select an appropriate one
    card = model_infos[0]
    
    # Restore model using `modelId` of the card.
    model = ModelPT.from_pretrained(card.modelId)
    
    *Parameters:*
      **model_filter** – Optional Dictionary (for Hugging Face Hub kwargs) that filters the returned
      list of compatible model cards, and selects all results from each filter. Users can then use
      model_card.modelId in from_pretrained() to restore a NeMo Model.
    *Returns:*
      A list of ModelInfo entries.
  
  * push_to_hf_hub(
  
    *repo_id: str*,
    ***,
    *pack_nemo_file: bool = True*,
    *model_card: huggingface_hub.ModelCard | None | object | str = None*,
    *commit_message: str = 'Push model using huggingface_hub.'*,
    *private: bool = False*,
    *api_endpoint: str | None = None*,
    *token: str | None = None*,
    *branch: str | None = None*,
    *allow_patterns: List[str] | str | None = None*,
    *ignore_patterns: List[str] | str | None = None*,
    *delete_patterns: List[str] | str | None = None*,
  )[#][27]*
    Upload model checkpoint to the Hub.
    
    Use allow_patterns and ignore_patterns to precisely filter which files should be pushed to the
    hub. Use delete_patterns to delete existing remote files in the same commit. See [upload_folder]
    reference for more details.
    
    *Parameters:*
      * **repo_id** (str) – ID of the repository to push to (example: “username/my-model”).
      * **pack_nemo_file** (bool, *optional*, defaults to True) – Whether to pack the model
        checkpoint and configuration into a single .nemo file. If set to false, uploads the contents
        of the directory containing the model checkpoint and configuration plus additional
        artifacts.
      * **model_card** (ModelCard, *optional*) – Model card to upload with the model. If None, will
        use the model card template provided by the class itself via generate_model_card(). Any
        object that implements str(obj) can be passed here. Two keyword replacements are passed to
        generate_model_card(): model_name and repo_id. If the model card generates a string, and it
        contains {model_name} or {repo_id}, they will be replaced with the actual values.
      * **commit_message** (str, *optional*) – Message to commit while pushing.
      * **private** (bool, *optional*, defaults to False) – Whether the repository created should be
        private.
      * **api_endpoint** (str, *optional*) – The API endpoint to use when pushing the model to the
        hub.
      * **token** (str, *optional*) – The token to use as HTTP bearer authorization for remote
        files. By default, it will use the token cached when running huggingface-cli login.
      * **branch** (str, *optional*) – The git branch on which to push the model. This defaults to
        “main”.
      * **allow_patterns** (List[str] or str, *optional*) – If provided, only files matching at
        least one pattern are pushed.
      * **ignore_patterns** (List[str] or str, *optional*) – If provided, files matching any of the
        patterns are not pushed.
      * **delete_patterns** (List[str] or str, *optional*) – If provided, remote files matching any
        of the patterns will be deleted from the repo.
    *Returns:*
      The url of the uploaded HF repo.

## Neural Type checking[#][28]

* *class *nemo.core.classes.common.typecheck(

  *input_types: [TypeState][29] | Dict[str, [NeuralType][30]] = TypeState.UNINITIALIZED*,
  *output_types: [TypeState][31] | Dict[str, [NeuralType][32]] = TypeState.UNINITIALIZED*,
  *ignore_collections: bool = False*,
)[#][33]*
  Bases: `object`
  
  A decorator which performs input-output neural type checks, and attaches neural types to the
  output of the function that it wraps.
  
  Requires that the class inherit from `Typing` in order to perform type checking, and will raise an
  error if that is not the case.
  
  # Usage (Class level type support)
  
  @typecheck()
  def fn(self, arg1, arg2, ...):
      ...
  
  # Usage (Function level type support)
  
  @typecheck(input_types=..., output_types=...)
  def fn(self, arg1, arg2, ...):
      ...
  
  Points to be noted:
  
  1. The brackets () in @typecheck() are necessary.
     
     > You will encounter a TypeError: __init__() takes 1 positional argument but X were given
     > without those brackets.
  2. The function can take any number of positional arguments during definition.
     
     > When you call this function, all arguments must be passed using kwargs only.
  
  * __call__(*wrapped*)[#][34]*
    Call self as a function.
  
  * *class *TypeState(*value*)[#][35]*
    Bases: `Enum`
    
    Placeholder to denote the default value of type information provided. If the constructor of this
    decorator is used to override the class level type definition, this enum value indicate that
    types will be overridden.
  
  * wrapped_call(
  
    *wrapped*,
    *instance: Typing*,
    *args*,
    *kwargs*,
  )[#][36]*
    Wrapper method that can be used on any function of a class that implements `Typing`. By default,
    it will utilize the input_types and output_types properties of the class inheriting Typing.
    
    Local function level overrides can be provided by supplying dictionaries as arguments to the
    decorator.
    
    *Parameters:*
      * **input_types** – Union[TypeState, Dict[str, NeuralType]]. By default, uses the global
        input_types.
      * **output_types** – Union[TypeState, Dict[str, NeuralType]]. By default, uses the global
        output_types.
      * **ignore_collections** – Bool. Determines if container types should be asserted for depth
        checks, or if depth checks are skipped entirely.
  
  * *static *set_typecheck_enabled(*enabled: bool = True*)[#][37]*
    Global method to enable/disable typechecking.
    
    *Parameters:*
      **enabled** – bool, when True will enable typechecking.
  
  * *static *disable_checks()[#][38]*
    Context manager that temporarily disables type checking within its context.
  
  * *static *set_semantic_check_enabled(*enabled: bool = True*)[#][39]*
    Global method to enable/disable semantic typechecking.
    
    *Parameters:*
      **enabled** – bool, when True will enable semantic typechecking.
  
  * *static *disable_semantic_checks()[#][40]*
    Context manager that temporarily disables semantic type checking within its context.

## Neural Type classes[#][41]

* *class *nemo.core.neural_types.NeuralType(

  *axes: Any | None = None*,
  *elements_type: Any | None = None*,
  *optional: bool = False*,
)[#][42]*
  Bases: `object`
  
  This is the main class which would represent neural type concept. It is used to represent *the
  types* of inputs and outputs.
  
  *Parameters:*
    * **axes** (*Optional**[**Tuple**]*) – a tuple of AxisTypes objects representing the semantics
      of what varying each axis means You can use a short, string-based form here. For example:
      (‘B’, ‘C’, ‘H’, ‘W’) would correspond to an NCHW format frequently used in computer vision.
      (‘B’, ‘T’, ‘D’) is frequently used for signal processing and means [batch, time,
      dimension/channel].
    * **elements_type** ([*ElementType*][43]) – an instance of ElementType class representing the
      semantics of what is stored inside the tensor. For example: logits (LogitsType), log
      probabilities (LogprobType), etc.
    * **optional** (*bool*) – By default, this is false. If set to True, it would means that input
      to the port of this type can be optional.
  
  * compare(
  
    *second*,
  ) → [NeuralTypeComparisonResult][44][#][45]*
    Performs neural type comparison of self with second. When you chain two modules’ inputs/outputs
    via __call__ method, this comparison will be called to ensure neural type compatibility.
  
  * compare_and_raise_error(
  
    *parent_type_name*,
    *port_name*,
    *second_object*,
  )[#][46]*
    Method compares definition of one type with another and raises an error if not compatible.

* *class *nemo.core.neural_types.axes.AxisType(

  *kind: AxisKindAbstract*,
  *size: int | None = None*,
  *is_list=False*,
)[#][47]*
  Bases: `object`
  
  This class represents axis semantics and (optionally) it’s dimensionality :param kind: what kind
  of axis it is? For example Batch, Height, etc. :type kind: AxisKindAbstract :param size: specify
  if the axis should have a fixed size. By default it is set to None and you :type size: int,
  optional :param typically do not want to set it for Batch and Time: :param is_list: whether this
  is a list or a tensor axis :type is_list: bool, default=False

* *class *nemo.core.neural_types.elements.ElementType[#][48]*
  Bases: `ABC`
  
  Abstract class defining semantics of the tensor elements. We are relying on Python for inheritance
  checking
  
  * *property *type_parameters*: Dict[str, Any]*[#][49]*
    Override this property to parametrize your type. For example, you can specify ‘storage’ type
    such as float, int, bool with ‘dtype’ keyword. Another example, is if you want to represent a
    signal with a particular property (say, sample frequency), then you can put sample_freq->value
    in there. When two types are compared their type_parameters must match.
  
  * *property *fields[#][50]*
    This should be used to logically represent tuples/structures. For example, if you want to
    represent a bounding box (x, y, width, height) you can put a tuple with names (‘x’, y’, ‘w’,
    ‘h’) in here. Under the hood this should be converted to the last tesnor dimension of fixed size
    = len(fields). When two types are compared their fields must match.

* *class *nemo.core.neural_types.comparison.NeuralTypeComparisonResult(*value*)[#][51]*
  Bases: `Enum`
  
  The result of comparing two neural type objects for compatibility. When comparing A.compare_to(B):

## Experiment manager[#][52]

* *class *nemo.utils.exp_manager.exp_manager(

  *trainer: lightning.pytorch.Trainer*,
  *cfg: omegaconf.DictConfig | Dict | None = None*,
)[#][53]*
  Bases:
  
  exp_manager is a helper function used to manage folders for experiments. It follows the pytorch
  lightning paradigm of exp_dir/model_or_experiment_name/version. If the lightning trainer has a
  logger, exp_manager will get exp_dir, name, and version from the logger. Otherwise it will use the
  exp_dir and name arguments to create the logging directory. exp_manager also allows for explicit
  folder creation via explicit_log_dir.
  
  The version can be a datetime string or an integer. Datestime version can be disabled if
  use_datetime_version is set to False. It optionally creates TensorBoardLogger, WandBLogger,
  DLLogger, MLFlowLogger, ClearMLLogger, ModelCheckpoint objects from pytorch lightning. It copies
  sys.argv, and git information if available to the logging directory. It creates a log file for
  each process to log their output into.
  
  exp_manager additionally has a resume feature (resume_if_exists) which can be used to continuing
  training from the constructed log_dir. When you need to continue the training repeatedly (like on
  a cluster which you need multiple consecutive jobs), you need to avoid creating the version
  folders. Therefore from v1.0.0, when resume_if_exists is set to True, creating the version folders
  is ignored.
  
  *Parameters:*
    * **trainer** (*lightning.pytorch.Trainer*) – The lightning trainer.
    * **cfg** (*DictConfig**, **dict*) –
      
      Can have the following keys:
      
      * *explicit_log_dir (str, Path): Can be used to override exp_dir/name/version folder*
          creation. Defaults to None, which will use exp_dir, name, and version to construct the
          logging directory.
      * *exp_dir (str, Path): The base directory to create the logging directory.*
          Defaults to None, which logs to ./nemo_experiments.
      * *name (str): The name of the experiment. Defaults to None which turns into “default”*
          via name = name or “default”.
      * *version (str): The version of the experiment. Defaults to None which uses either a*
          datetime string or lightning’s TensorboardLogger system of using version_{int}.
      * *use_datetime_version (bool): Whether to use a datetime string for version.*
          Defaults to True.
      * *resume_if_exists (bool): Whether this experiment is resuming from a previous run.*
          If True, it sets trainer._checkpoint_connector._ckpt_path so that the trainer should
          auto-resume. exp_manager will move files under log_dir to log_dir/run_{int}. Defaults to
          False. From v1.0.0, when resume_if_exists is True, we would not create version folders to
          make it easier to find the log folder for next runs.
      * *resume_past_end (bool): exp_manager errors out if resume_if_exists is True*
          and a checkpoint matching `*end.ckpt` indicating a previous training run fully completed.
          This behaviour can be disabled, in which case the `*end.ckpt` will be loaded by setting
          resume_past_end to True. Defaults to False.
      * *resume_ignore_no_checkpoint (bool): exp_manager errors out if resume_if_exists is True*
          and no checkpoint could be found. This behaviour can be disabled, in which case
          exp_manager will print a message and continue without restoring, by setting
          resume_ignore_no_checkpoint to True. Defaults to False.
      * *resume_from_checkpoint (str): Can be used to specify a path to a specific checkpoint*
          file to load from. This will override any checkpoint found when resume_if_exists is True.
          Defaults to None.
      * *create_tensorboard_logger (bool): Whether to create a tensorboard logger and attach it*
          to the pytorch lightning trainer. Defaults to True.
      * *summary_writer_kwargs (dict): A dictionary of kwargs that can be passed to lightning’s*
          TensorboardLogger class. Note that log_dir is passed by exp_manager and cannot exist in
          this dict. Defaults to None.
      * *create_wandb_logger (bool): Whether to create a Weights and Baises logger and attach it*
          to the pytorch lightning trainer. Defaults to False.
      * *wandb_logger_kwargs (dict): A dictionary of kwargs that can be passed to lightning’s*
          WandBLogger class. Note that name and project are required parameters if
          create_wandb_logger is True. Defaults to None.
      * *create_mlflow_logger (bool): Whether to create an MLFlow logger and attach it to the*
          pytorch lightning training. Defaults to False
      * mlflow_logger_kwargs (dict): optional parameters for the MLFlow logger
      * *create_dllogger_logger (bool): Whether to create an DLLogger logger and attach it to the*
          pytorch lightning training. Defaults to False
      * dllogger_logger_kwargs (dict): optional parameters for the DLLogger logger
      * *create_clearml_logger (bool): Whether to create an ClearML logger and attach it to the*
          pytorch lightning training. Defaults to False
      * clearml_logger_kwargs (dict): optional parameters for the ClearML logger
      * *create_checkpoint_callback (bool): Whether to create a ModelCheckpoint callback and*
          attach it to the pytorch lightning trainer. The ModelCheckpoint saves the top 3 models
          with the best “val_loss”, the most recent checkpoint under `*last.ckpt`, and the final
          checkpoint after training completes under `*end.ckpt`. Defaults to True.
      * *create_early_stopping_callback (bool): Flag to decide if early stopping should be used*
          to stop training. Default is False. See EarlyStoppingParams dataclass above.
      * *create_preemption_callback (bool): Flag to decide whether to enable preemption callback*
          to save checkpoints and exit training immediately upon preemption. Default is True.
      * *create_straggler_detection_callback (bool): Use straggler detection callback.*
          Default is False.
      * create_fault_tolerance_callback (bool): Use fault tolerance callback. Default is False.
      * *files_to_copy (list): A list of files to copy to the experiment logging directory.*
          Defaults to None which copies no files.
      * *log_local_rank_0_only (bool): Whether to only create log files for local rank 0.*
          Defaults to False. Set this to True if you are using DDP with many GPUs and do not want
          many log files in your exp dir.
      * *log_global_rank_0_only (bool): Whether to only create log files for global rank 0.*
          Defaults to False. Set this to True if you are using DDP with many GPUs and do not want
          many log files in your exp dir.
      * *max_time (str): The maximum wall clock time *per run*. This is intended to be used on*
          clusters where you want a checkpoint to be saved after this specified time and be able to
          resume from that checkpoint. Defaults to None.
      * *seconds_to_sleep (float): seconds to sleep non rank 0 processes for. Used to give*
          enough time for rank 0 to initialize
      * *train_time_interval (timedelta): pass an object of timedelta to save the model every*
          timedelta. Defaults to None. (use _target_ with hydra to achieve this)
  *Returns:*
    *The final logging directory where logging files are saved. Usually the concatenation of*
      exp_dir, name, and version.
  *Return type:*
    log_dir (Path)

* *class *nemo.utils.exp_manager.ExpManagerConfig(

  *explicit_log_dir: str | None = None*,
  *exp_dir: str | None = None*,
  *name: str | None = None*,
  *version: str | None = None*,
  *use_datetime_version: bool | None = True*,
  *resume_if_exists: bool | None = False*,
  *resume_past_end: bool | None = False*,
  *resume_ignore_no_checkpoint: bool | None = False*,
  *resume_from_checkpoint: str | None = None*,
  *create_tensorboard_logger: bool | None = True*,
  *summary_writer_kwargs: ~typing.Dict[~typing.Any*,
  *~typing.Any] | None = None*,
  *create_wandb_logger: bool | None = False*,
  *wandb_logger_kwargs: ~typing.Dict[~typing.Any*,
  *~typing.Any] | None = None*,
  *create_mlflow_logger: bool | None = False*,
  *mlflow_logger_kwargs: ~nemo.utils.loggers.mlflow_logger.MLFlowParams | None = <factory>*,
  *create_dllogger_logger: bool | None = False*,
  *dllogger_logger_kwargs: ~nemo.utils.loggers.dllogger.DLLoggerParams | None = <factory>*,
  *create_clearml_logger: bool | None = False*,
  *clearml_logger_kwargs: ~nemo.utils.loggers.clearml_logger.ClearMLParams | None = <factory>*,
  *create_neptune_logger: bool | None = False*,
  *neptune_logger_kwargs: ~typing.Dict[~typing.Any*,
  *~typing.Any] | None = None*,
  *create_checkpoint_callback: bool | None = True*,
  *checkpoint_callback_params: ~nemo.utils.exp_manager.CallbackParams | None = <factory>*,
  *create_early_stopping_callback: bool | None = False*,
  *create_ipl_epoch_stopper_callback: bool | None = False*,
  *early_stopping_callback_params: ~nemo.utils.exp_manager.EarlyStoppingParams | None = <factory>*,
  *ipl_epoch_stopper_callback_params: ~nemo.utils.exp_manager.IPLEpochStopperParams | None =
  <factory>*,
  *create_preemption_callback: bool | None = True*,
  *files_to_copy: ~typing.List[str] | None = None*,
  *log_step_timing: bool | None = True*,
  *log_delta_step_timing: bool | None = False*,
  *step_timing_kwargs: ~nemo.utils.exp_manager.StepTimingParams | None = <factory>*,
  *log_local_rank_0_only: bool | None = False*,
  *log_global_rank_0_only: bool | None = False*,
  *disable_validation_on_resume: bool | None = True*,
  *ema: ~nemo.utils.exp_manager.EMAParams | None = <factory>*,
  *max_time_per_run: str | None = None*,
  *seconds_to_sleep: float = 5*,
  *create_straggler_detection_callback: bool | None = False*,
  *straggler_detection_params: ~nemo.utils.exp_manager.StragglerDetectionParams | None = <factory>*,
  *create_fault_tolerance_callback: bool | None = False*,
  *fault_tolerance: ~nemo.utils.exp_manager.FaultToleranceParams | None = <factory>*,
  *log_tflops_per_sec_per_gpu: bool | None = True*,
)[#][54]*
  Bases: `object`
  
  Experiment Manager config for validation of passed arguments.

## Exportable[#][55]

* *class *nemo.core.classes.exportable.Exportable[#][56]*
  Bases: `ABC`
  
  This Interface should be implemented by particular classes derived from nemo.core.NeuralModule or
  nemo.core.ModelPT. It gives these entities ability to be exported for deployment to formats such
  as ONNX.
  
  *Usage:*
    # exporting pre-trained model to ONNX file for deployment. model.eval() model.to(‘cuda’) # or
    to(‘cpu’) if you don’t have GPU
    
    model.export(‘mymodel.onnx’, [options]) # all arguments apart from output are optional.
  
  * export(
  
    *output: str*,
    *input_example=None*,
    *verbose=False*,
    *do_constant_folding=True*,
    *onnx_opset_version=None*,
    *check_trace: bool | List[torch.Tensor] = False*,
    *dynamic_axes=None*,
    *check_tolerance=0.01*,
    *export_modules_as_functions=False*,
    *keep_initializers_as_inputs=None*,
    *use_dynamo=False*,
  )[#][57]*
    Exports the model to the specified format. The format is inferred from the file extension of the
    output file.
    
    *Parameters:*
      * **output** (*str*) – Output file name. File extension be .onnx, .pt, or .ts, and is used to
        select export path of the model.
      * **input_example** (*list** or **dict*) – Example input to the model’s forward function. This
        is used to trace the model and export it to ONNX/TorchScript. If the model takes multiple
        inputs, then input_example should be a list of input examples. If the model takes named
        inputs, then input_example should be a dictionary of input examples.
      * **verbose** (*bool*) – If True, will print out a detailed description of the model’s export
        steps, along with the internal trace logs of the export process.
      * **do_constant_folding** (*bool*) – If True, will execute constant folding optimization on
        the model’s graph before exporting. This is ONNX specific.
      * **onnx_opset_version** (*int*) – The ONNX opset version to export the model to. If None,
        will use a reasonable default version.
      * **check_trace** (*bool*) – If True, will verify that the model’s output matches the output
        of the traced model, upto some tolerance.
      * **dynamic_axes** (*dict*) – A dictionary mapping input and output names to their dynamic
        axes. This is used to specify the dynamic axes of the model’s inputs and outputs. If the
        model takes multiple inputs, then dynamic_axes should be a list of dictionaries. If the
        model takes named inputs, then dynamic_axes should be a dictionary of dictionaries. If None,
        will use the dynamic axes of the input_example derived from the NeuralType of the input and
        output of the model.
      * **check_tolerance** (*float*) – The tolerance to use when checking the model’s output
        against the traced model’s output. This is only used if check_trace is True. Note the high
        tolerance is used because the traced model is not guaranteed to be 100% accurate.
      * **export_modules_as_functions** (*bool*) – If True, will export the model’s submodules as
        functions. This is ONNX specific.
      * **keep_initializers_as_inputs** (*bool*) – If True, will keep the model’s initializers as
        inputs in the onnx graph. This is ONNX specific.
      * **use_dynamo** (*bool*) – If True, use onnx.dynamo_export() instead of onnx.export(). This
        is ONNX specific.
    *Returns:*
      A tuple of two outputs. Item 0 in the output is a list of outputs, the outputs of each subnet
      exported. Item 1 in the output is a list of string descriptions. The description of each
      subnet exported can be used for logging purposes.
  
  * *property *disabled_deployment_input_names*: List[str]*[#][58]*
    Implement this method to return a set of input names disabled for export
  
  * *property *disabled_deployment_output_names*: List[str]*[#][59]*
    Implement this method to return a set of output names disabled for export
  
  * *property *supported_export_formats*: List[ExportFormat]*[#][60]*
    Implement this method to return a set of export formats supported. Default is all types.
  
  * get_export_subnet(*subnet=None*)[#][61]*
    Returns Exportable subnet model/module to export
  
  * list_export_subnets()[#][62]*
    Returns default set of subnet names exported for this model First goes the one receiving input
    (input_example)
  
  * get_export_config()[#][63]*
    Returns export_config dictionary
  
  * set_export_config(*args*)[#][64]*
    Sets/updates export_config dictionary

[1]: #nemo-core-apis
[2]: #base-class-for-all-nemo-models
[3]: #base-neural-module-class
[4]: #base-mixin-classes
[5]: #base-connector-classes
[6]: #nemo.core.connectors.save_restore_connector.SaveRestoreConnector
[7]: #nemo.core.connectors.save_restore_connector.SaveRestoreConnector.save_to
[8]: #nemo.core.connectors.save_restore_connector.SaveRestoreConnector.load_config_and_state_dict
[9]: #nemo.core.connectors.save_restore_connector.SaveRestoreConnector.modify_state_dict
[10]: #nemo.core.connectors.save_restore_connector.SaveRestoreConnector.load_instance_with_state_dic
t
[11]: #nemo.core.connectors.save_restore_connector.SaveRestoreConnector.restore_from
[12]: #nemo.core.connectors.save_restore_connector.SaveRestoreConnector.extract_state_dict_from
[13]: #nemo.core.connectors.save_restore_connector.SaveRestoreConnector.register_artifact
[14]: #nemo.core.connectors.save_restore_connector.SaveRestoreConnector.model_config_yaml
[15]: #nemo.core.connectors.save_restore_connector.SaveRestoreConnector.model_weights_ckpt
[16]: #nemo.core.connectors.save_restore_connector.SaveRestoreConnector.model_extracted_dir
[17]: #nemo.core.connectors.save_restore_connector.SaveRestoreConnector.pack_nemo_file
[18]: #id1
[19]: #nemo.core.classes.mixins.access_mixins.AccessMixin
[20]: #nemo.core.classes.mixins.access_mixins.AccessMixin.register_accessible_tensor
[21]: #nemo.core.classes.mixins.access_mixins.AccessMixin.get_module_registry
[22]: #nemo.core.classes.mixins.access_mixins.AccessMixin.reset_registry
[23]: #nemo.core.classes.mixins.access_mixins.AccessMixin.access_cfg
[24]: #nemo.core.classes.mixins.hf_io_mixin.HuggingFaceFileIO
[25]: #nemo.core.classes.mixins.hf_io_mixin.HuggingFaceFileIO.get_hf_model_filter
[26]: #nemo.core.classes.mixins.hf_io_mixin.HuggingFaceFileIO.search_huggingface_models
[27]: #nemo.core.classes.mixins.hf_io_mixin.HuggingFaceFileIO.push_to_hf_hub
[28]: #neural-type-checking
[29]: #nemo.core.classes.common.typecheck.TypeState
[30]: #nemo.core.neural_types.NeuralType
[31]: #nemo.core.classes.common.typecheck.TypeState
[32]: #nemo.core.neural_types.NeuralType
[33]: #nemo.core.classes.common.typecheck
[34]: #nemo.core.classes.common.typecheck.__call__
[35]: #nemo.core.classes.common.typecheck.TypeState
[36]: #nemo.core.classes.common.typecheck.wrapped_call
[37]: #nemo.core.classes.common.typecheck.set_typecheck_enabled
[38]: #nemo.core.classes.common.typecheck.disable_checks
[39]: #nemo.core.classes.common.typecheck.set_semantic_check_enabled
[40]: #nemo.core.classes.common.typecheck.disable_semantic_checks
[41]: #neural-type-classes
[42]: #nemo.core.neural_types.NeuralType
[43]: #nemo.core.neural_types.elements.ElementType
[44]: #nemo.core.neural_types.comparison.NeuralTypeComparisonResult
[45]: #nemo.core.neural_types.NeuralType.compare
[46]: #nemo.core.neural_types.NeuralType.compare_and_raise_error
[47]: #nemo.core.neural_types.axes.AxisType
[48]: #nemo.core.neural_types.elements.ElementType
[49]: #nemo.core.neural_types.elements.ElementType.type_parameters
[50]: #nemo.core.neural_types.elements.ElementType.fields
[51]: #nemo.core.neural_types.comparison.NeuralTypeComparisonResult
[52]: #experiment-manager
[53]: #nemo.utils.exp_manager.exp_manager
[54]: #nemo.utils.exp_manager.ExpManagerConfig
[55]: #exportable
[56]: #nemo.core.classes.exportable.Exportable
[57]: #nemo.core.classes.exportable.Exportable.export
[58]: #nemo.core.classes.exportable.Exportable.disabled_deployment_input_names
[59]: #nemo.core.classes.exportable.Exportable.disabled_deployment_output_names
[60]: #nemo.core.classes.exportable.Exportable.supported_export_formats
[61]: #nemo.core.classes.exportable.Exportable.get_export_subnet
[62]: #nemo.core.classes.exportable.Exportable.list_export_subnets
[63]: #nemo.core.classes.exportable.Exportable.get_export_config
[64]: #nemo.core.classes.exportable.Exportable.set_export_config
