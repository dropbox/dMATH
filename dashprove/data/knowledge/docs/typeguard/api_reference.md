* API reference
* [ View page source][1]

# API reference[][2]

## Type checking[][3]

* typeguard.check_type(*value*, *expected_type*, ***, *forward_ref_policy=ForwardRefPolicy.WARN*,
*typecheck_fail_callback=None*,
*collection_check_strategy=CollectionCheckStrategy.FIRST_ITEM*)[][4]*
  Ensure that `value` matches `expected_type`.
  
  The types from the [`typing`][5] module do not support [`isinstance()`][6] or [`issubclass()`][7]
  so a number of type specific checks are required. This function knows which checker to call for
  which type.
  
  This function wraps [`check_type_internal()`][8] in the following ways:
  
  * Respects type checking suppression ([`suppress_type_checks()`][9])
  * Forms a [`TypeCheckMemo`][10] from the current stack frame
  * Calls the configured type check fail callback if the check fails
  
  Note that this function is independent of the globally shared configuration in
  [`typeguard.config`][11]. This means that usage within libraries is safe from being affected
  configuration changes made by other libraries or by the integrating application. Instead,
  configuration options have the same default values as their corresponding fields in
  [`TypeCheckConfiguration`][12].
  
  *Parameters:*
    * **value** ([`object`][13]) – value to be checked against `expected_type`
    * **expected_type** ([`Any`][14]) – a class or generic type instance, or a tuple of such things
    * **forward_ref_policy** ([`ForwardRefPolicy`][15]) – see
      [`TypeCheckConfiguration.forward_ref_policy`][16]
    * **typecheck_fail_callback** ([`Optional`][17][typeguard.TypeCheckFailCallback]) – see
      :attr`TypeCheckConfiguration.typecheck_fail_callback`
    * **collection_check_strategy** ([`CollectionCheckStrategy`][18]) – see
      [`TypeCheckConfiguration.collection_check_strategy`][19]
  *Return type:*
    [`Any`][20]
  *Returns:*
    `value`, unmodified
  *Raises:*
    [**TypeCheckError**][21] – if there is a type mismatch

* @typeguard.typechecked(*target=None*, ***, *forward_ref_policy=<unset>*,
*typecheck_fail_callback=<unset>*, *collection_check_strategy=<unset>*,
*debug_instrumentation=<unset>*)[][22]*
  Instrument the target function to perform run-time type checking.
  
  This decorator recompiles the target function, injecting code to type check arguments, return
  values, yield values (excluding `yield from`) and assignments to annotated local variables.
  
  This can also be used as a class decorator. This will instrument all type annotated methods,
  including [`@classmethod`][23], [`@staticmethod`][24], and [`@property`][25] decorated methods in
  the class.
  
  Note
  
  When Python is run in optimized mode (`-O` or `-OO`, this decorator is a no-op). This is a feature
  meant for selectively introducing type checking into a code base where the checks aren’t meant to
  be run in production.
  
  *Parameters:*
    * **target** ([`Optional`][26][[`TypeVar`][27](`T_CallableOrType`, bound=
      [`Callable`][28][[`...`][29], [`Any`][30]])]) – the function or class to enable type checking
      for
    * **forward_ref_policy** ([`ForwardRefPolicy`][31] | [`Unset`][32]) – override for
      [`TypeCheckConfiguration.forward_ref_policy`][33]
    * **typecheck_fail_callback** ([`Union`][34][typeguard.TypeCheckFailCallback, [`Unset`][35]]) –
      override for [`TypeCheckConfiguration.typecheck_fail_callback`][36]
    * **collection_check_strategy** ([`CollectionCheckStrategy`][37] | [`Unset`][38]) – override for
      [`TypeCheckConfiguration.collection_check_strategy`][39]
    * **debug_instrumentation** ([`bool`][40] | [`Unset`][41]) – override for
      [`TypeCheckConfiguration.debug_instrumentation`][42]
  *Return type:*
    [`Any`][43]

## Import hook[][44]

* typeguard.install_import_hook(*packages=None*, ***, *cls=<class
'typeguard.TypeguardFinder'>*)[][45]*
  Install an import hook that instruments functions for automatic type checking.
  
  This only affects modules loaded **after** this hook has been installed.
  
  *Parameters:*
    * **packages** ([`Iterable`][46][[`str`][47]] | [`None`][48]) – an iterable of package names to
      instrument, or `None` to instrument all packages
    * **cls** ([`type`][49][[`TypeguardFinder`][50]]) – a custom meta path finder class
  *Return type:*
    [`ImportHookManager`][51]
  *Returns:*
    a context manager that uninstalls the hook on exit (or when you call `.uninstall()`)
  
  Added in version 2.6.

* *class *typeguard.TypeguardFinder(*packages*, *original_pathfinder*)[][52]*
  Wraps another path finder and instruments the module with [`@typechecked`][53] if
  [`should_instrument()`][54] returns `True`.
  
  Should not be used directly, but rather via [`install_import_hook()`][55].
  
  Added in version 2.6.
  
  * should_instrument(*module_name*)[][56]*
    Determine whether the module with the given name should be instrumented.
    
    *Parameters:*
      **module_name** ([`str`][57]) – full name of the module that is about to be imported (e.g.
      `xyz.abc`)
    *Return type:*
      [`bool`][58]

* *class *typeguard.ImportHookManager(*hook*)[][59]*
  A handle that can be used to uninstall the Typeguard import hook.
  
  * uninstall()[][60]*
    Uninstall the import hook.
    
    *Return type:*
      [`None`][61]

## Configuration[][62]

* typeguard.config*: [TypeCheckConfiguration][63]*[][64]*
  The global configuration object.
  
  Used by [`@typechecked`][65] and [`install_import_hook()`][66], and notably **not used** by
  [`check_type()`][67].

* *class *typeguard.TypeCheckConfiguration(*forward_ref_policy=ForwardRefPolicy.WARN*,
*typecheck_fail_callback=None*, *collection_check_strategy=CollectionCheckStrategy.FIRST_ITEM*,
*debug_instrumentation=False*)[][68]*
  > You can change Typeguard’s behavior with these settings.
  
  * typecheck_fail_callback*: Callable[[[TypeCheckError][69], [TypeCheckMemo][70]], Any]*[][71]*
    > Callable that is called when type checking fails.
    > 
    > Default: `None` (the [`TypeCheckError`][72] is raised directly)
  
  * forward_ref_policy*: [ForwardRefPolicy][73]*[][74]*
    > Specifies what to do when a forward reference fails to resolve.
    > 
    > Default: `WARN`
  
  * collection_check_strategy*: [CollectionCheckStrategy][75]*[][76]*
    > Specifies how thoroughly the contents of collections (list, dict, etc.) are type checked.
    > 
    > Default: `FIRST_ITEM`
  
  * debug_instrumentation*: [bool][77]*[][78]*
    > If set to `True`, the code of modules or functions instrumented by typeguard is printed to
    > `sys.stderr` after the instrumentation is done
    > 
    > Default: `False`

* *class *typeguard.CollectionCheckStrategy(*value*)[][79]*
  Specifies how thoroughly the contents of collections are type checked.
  
  This has an effect on the following built-in checkers:
  
  * `AbstractSet`
  * `Dict`
  * `List`
  * `Mapping`
  * `Set`
  * `Tuple[<type>, ...]` (arbitrarily sized tuples)
  
  Members:
  
  * `FIRST_ITEM`: check only the first item
  * `ALL_ITEMS`: check all items

* *class *typeguard.Unset[][80]*

* *class *typeguard.ForwardRefPolicy(*value*)[][81]*
  Defines how unresolved forward references are handled.
  
  Members:
  
  * `ERROR`: propagate the [`NameError`][82] when the forward reference lookup fails
  * `WARN`: emit a [`TypeHintWarning`][83] if the forward reference lookup fails
  * `IGNORE`: silently skip checks for unresolveable forward references

* typeguard.warn_on_error(*exc*, *memo*)[][84]*
  Emit a warning on a type mismatch.
  
  This is intended to be used as an error handler in
  [`TypeCheckConfiguration.typecheck_fail_callback`][85].
  
  *Return type:*
    [`None`][86]

## Custom checkers[][87]

* typeguard.check_type_internal(*value*, *annotation*, *memo*)[][88]*
  Check that the given object is compatible with the given type annotation.
  
  This function should only be used by type checker callables. Applications should use
  [`check_type()`][89] instead.
  
  *Parameters:*
    * **value** ([`Any`][90]) – the value to check
    * **annotation** ([`Any`][91]) – the type annotation to check against
    * **memo** ([`TypeCheckMemo`][92]) – a memo object containing configuration and information
      necessary for looking up forward references
  *Return type:*
    [`None`][93]

* typeguard.load_plugins()[][94]*
  Load all type checker lookup functions from entry points.
  
  All entry points from the `typeguard.checker_lookup` group are loaded, and the returned lookup
  functions are added to [`typeguard.checker_lookup_functions`][95].
  
  Note
  
  This function is called implicitly on import, unless the `TYPEGUARD_DISABLE_PLUGIN_AUTOLOAD`
  environment variable is present.
  
  *Return type:*
    [`None`][96]

* typeguard.checker_lookup_functions*: [list][97][Callable[[Any, Tuple[Any, ...], Tuple[Any, ...]],
Callable[[Any, Any, Tuple[Any, ...], [TypeCheckMemo][98]], Any] | [None][99]]]*[][100]*
  A list of callables that are used to look up a checker callable for an annotation.

* *class *typeguard.TypeCheckMemo(*globals*, *locals*, ***, *self_type=None*,
*config=TypeCheckConfiguration(forward_ref_policy=<ForwardRefPolicy.WARN: 2>*,
*typecheck_fail_callback=None*, *collection_check_strategy=<CollectionCheckStrategy.FIRST_ITEM: 1>*,
*debug_instrumentation=False)*)[][101]*
  Contains information necessary for type checkers to do their work.
  
  * globals*: [dict][102][[str][103], Any]*[][104]*
    > Dictionary of global variables to use for resolving forward references.
  
  * locals*: [dict][105][[str][106], Any]*[][107]*
    > Dictionary of local variables to use for resolving forward references.
  
  * self_type*: [type][108] | [None][109]*[][110]*
    > When running type checks within an instance method or class method, this is the class object
    > that the first argument (usually named `self` or `cls`) refers to.
  
  * config*: [TypeCheckConfiguration][111]*[][112]*
    > Contains the configuration for a particular set of type checking operations.

## Type check suppression[][113]

* @typeguard.typeguard_ignore[][114]*
  Decorator to indicate that annotations are not type hints.
  
  The argument must be a class or function; if it is a class, it applies recursively to all methods
  and classes defined in that class (but not to methods defined in its superclasses or subclasses).
  
  This mutates the function(s) or class(es) in place.

* typeguard.suppress_type_checks(*func=None*)[][115]*
  Temporarily suppress all type checking.
  
  This function has two operating modes, based on how it’s used:
  
  1. as a context manager (`with suppress_type_checks(): ...`)
  2. as a decorator (`@suppress_type_checks`)
  
  When used as a context manager, [`check_type()`][116] and any automatically instrumented functions
  skip the actual type checking. These context managers can be nested.
  
  When used as a decorator, all type checking is suppressed while the function is running.
  
  Type checking will resume once no more context managers are active and no decorated functions are
  running.
  
  Both operating modes are thread-safe.
  
  *Return type:*
    [`Union`][117][[`Callable`][118][[[`ParamSpec`][119](`P`)], [`TypeVar`][120](`T`)],
    [`ContextManager`][121][[`None`][122]]]

## Exceptions and warnings[][123]

* *exception *typeguard.InstrumentationWarning(*message*)[][124]*
  Emitted when there’s a problem with instrumenting a function for type checks.

* *exception *typeguard.TypeCheckError(*message*)[][125]*
  Raised by typeguard’s type checkers when a type mismatch is detected.

* *exception *typeguard.TypeCheckWarning(*message*)[][126]*
  Emitted by typeguard’s type checkers when a type mismatch is detected.

* *exception *typeguard.TypeHintWarning[][127]*
  A warning that is emitted when a type hint in string form could not be resolved to an actual type.
[ Previous][128] [Next ][129]

© Copyright 2015, Alex Grönholm.

Built with [Sphinx][130] using a [theme][131] provided by [Read the Docs][132].

[1]: _sources/api.rst.txt
[2]: #module-typeguard
[3]: #type-checking
[4]: #typeguard.check_type
[5]: https://docs.python.org/3/library/typing.html#module-typing
[6]: https://docs.python.org/3/library/functions.html#isinstance
[7]: https://docs.python.org/3/library/functions.html#issubclass
[8]: #typeguard.check_type_internal
[9]: #typeguard.suppress_type_checks
[10]: #typeguard.TypeCheckMemo
[11]: #typeguard.config
[12]: #typeguard.TypeCheckConfiguration
[13]: https://docs.python.org/3/library/functions.html#object
[14]: https://docs.python.org/3/library/typing.html#typing.Any
[15]: #typeguard.ForwardRefPolicy
[16]: #typeguard.TypeCheckConfiguration.forward_ref_policy
[17]: https://docs.python.org/3/library/typing.html#typing.Optional
[18]: #typeguard.CollectionCheckStrategy
[19]: #typeguard.TypeCheckConfiguration.collection_check_strategy
[20]: https://docs.python.org/3/library/typing.html#typing.Any
[21]: #typeguard.TypeCheckError
[22]: #typeguard.typechecked
[23]: https://docs.python.org/3/library/functions.html#classmethod
[24]: https://docs.python.org/3/library/functions.html#staticmethod
[25]: https://docs.python.org/3/library/functions.html#property
[26]: https://docs.python.org/3/library/typing.html#typing.Optional
[27]: https://docs.python.org/3/library/typing.html#typing.TypeVar
[28]: https://docs.python.org/3/library/typing.html#typing.Callable
[29]: https://docs.python.org/3/library/constants.html#Ellipsis
[30]: https://docs.python.org/3/library/typing.html#typing.Any
[31]: #typeguard.ForwardRefPolicy
[32]: #typeguard.Unset
[33]: #typeguard.TypeCheckConfiguration.forward_ref_policy
[34]: https://docs.python.org/3/library/typing.html#typing.Union
[35]: #typeguard.Unset
[36]: #typeguard.TypeCheckConfiguration.typecheck_fail_callback
[37]: #typeguard.CollectionCheckStrategy
[38]: #typeguard.Unset
[39]: #typeguard.TypeCheckConfiguration.collection_check_strategy
[40]: https://docs.python.org/3/library/functions.html#bool
[41]: #typeguard.Unset
[42]: #typeguard.TypeCheckConfiguration.debug_instrumentation
[43]: https://docs.python.org/3/library/typing.html#typing.Any
[44]: #import-hook
[45]: #typeguard.install_import_hook
[46]: https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable
[47]: https://docs.python.org/3/library/stdtypes.html#str
[48]: https://docs.python.org/3/library/constants.html#None
[49]: https://docs.python.org/3/library/functions.html#type
[50]: #typeguard.TypeguardFinder
[51]: #typeguard.ImportHookManager
[52]: #typeguard.TypeguardFinder
[53]: #typeguard.typechecked
[54]: #typeguard.TypeguardFinder.should_instrument
[55]: #typeguard.install_import_hook
[56]: #typeguard.TypeguardFinder.should_instrument
[57]: https://docs.python.org/3/library/stdtypes.html#str
[58]: https://docs.python.org/3/library/functions.html#bool
[59]: #typeguard.ImportHookManager
[60]: #typeguard.ImportHookManager.uninstall
[61]: https://docs.python.org/3/library/constants.html#None
[62]: #configuration
[63]: #typeguard.TypeCheckConfiguration
[64]: #typeguard.config
[65]: #typeguard.typechecked
[66]: #typeguard.install_import_hook
[67]: #typeguard.check_type
[68]: #typeguard.TypeCheckConfiguration
[69]: #typeguard.TypeCheckError
[70]: #typeguard.TypeCheckMemo
[71]: #typeguard.TypeCheckConfiguration.typecheck_fail_callback
[72]: #typeguard.TypeCheckError
[73]: #typeguard.ForwardRefPolicy
[74]: #typeguard.TypeCheckConfiguration.forward_ref_policy
[75]: #typeguard.CollectionCheckStrategy
[76]: #typeguard.TypeCheckConfiguration.collection_check_strategy
[77]: https://docs.python.org/3/library/functions.html#bool
[78]: #typeguard.TypeCheckConfiguration.debug_instrumentation
[79]: #typeguard.CollectionCheckStrategy
[80]: #typeguard.Unset
[81]: #typeguard.ForwardRefPolicy
[82]: https://docs.python.org/3/library/exceptions.html#NameError
[83]: #typeguard.TypeHintWarning
[84]: #typeguard.warn_on_error
[85]: #typeguard.TypeCheckConfiguration.typecheck_fail_callback
[86]: https://docs.python.org/3/library/constants.html#None
[87]: #custom-checkers
[88]: #typeguard.check_type_internal
[89]: #typeguard.check_type
[90]: https://docs.python.org/3/library/typing.html#typing.Any
[91]: https://docs.python.org/3/library/typing.html#typing.Any
[92]: #typeguard.TypeCheckMemo
[93]: https://docs.python.org/3/library/constants.html#None
[94]: #typeguard.load_plugins
[95]: #typeguard.checker_lookup_functions
[96]: https://docs.python.org/3/library/constants.html#None
[97]: https://docs.python.org/3/library/stdtypes.html#list
[98]: #typeguard.TypeCheckMemo
[99]: https://docs.python.org/3/library/constants.html#None
[100]: #typeguard.checker_lookup_functions
[101]: #typeguard.TypeCheckMemo
[102]: https://docs.python.org/3/library/stdtypes.html#dict
[103]: https://docs.python.org/3/library/stdtypes.html#str
[104]: #typeguard.TypeCheckMemo.globals
[105]: https://docs.python.org/3/library/stdtypes.html#dict
[106]: https://docs.python.org/3/library/stdtypes.html#str
[107]: #typeguard.TypeCheckMemo.locals
[108]: https://docs.python.org/3/library/functions.html#type
[109]: https://docs.python.org/3/library/constants.html#None
[110]: #typeguard.TypeCheckMemo.self_type
[111]: #typeguard.TypeCheckConfiguration
[112]: #typeguard.TypeCheckMemo.config
[113]: #type-check-suppression
[114]: #typeguard.typeguard_ignore
[115]: #typeguard.suppress_type_checks
[116]: #typeguard.check_type
[117]: https://docs.python.org/3/library/typing.html#typing.Union
[118]: https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable
[119]: https://docs.python.org/3/library/typing.html#typing.ParamSpec
[120]: https://docs.python.org/3/library/typing.html#typing.TypeVar
[121]: https://docs.python.org/3/library/typing.html#typing.ContextManager
[122]: https://docs.python.org/3/library/constants.html#None
[123]: #exceptions-and-warnings
[124]: #typeguard.InstrumentationWarning
[125]: #typeguard.TypeCheckError
[126]: #typeguard.TypeCheckWarning
[127]: #typeguard.TypeHintWarning
[128]: contributing.html
[129]: versionhistory.html
[130]: https://www.sphinx-doc.org/
[131]: https://github.com/readthedocs/sphinx_rtd_theme
[132]: https://readthedocs.org
