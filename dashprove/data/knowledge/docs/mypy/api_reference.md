# Welcome to mypy documentation![¶][1]

Mypy is a static type checker for Python.

Type checkers help ensure that you’re using variables and functions in your code correctly. With
mypy, add type hints ([**PEP 484**][2]) to your Python programs, and mypy will warn you when you use
those types incorrectly.

Python is a dynamic language, so usually you’ll only see errors in your code when you attempt to run
it. Mypy is a *static* checker, so it finds bugs in your programs without even running them!

Here is a small example to whet your appetite:

number = input("What is your favourite number?")
print("It is", number + 1)  # error: Unsupported operand types for + ("str" and "int")

Adding type hints for mypy does not interfere with the way your program would otherwise run. Think
of type hints as similar to comments! You can always use the Python interpreter to run your code,
even if mypy reports errors.

Mypy is designed with gradual typing in mind. This means you can add type hints to your code base
slowly and that you can always fall back to dynamic typing when static typing is not convenient.

Mypy has a powerful and easy-to-use type system, supporting features such as type inference,
generics, callable types, tuple types, union types, structural subtyping and more. Using mypy will
make your programs easier to understand, debug, and maintain.

Note

Although mypy is production ready, there may be occasional changes that break backward
compatibility. The mypy development team tries to minimize the impact of changes to user code. In
case of a major breaking change, mypy’s major version will be bumped.

## Contents[¶][3]

First steps

* [Getting started][4]
  
  * [Installing and running mypy][5]
  * [Dynamic vs static typing][6]
  * [Strict mode and configuration][7]
  * [More complex types][8]
  * [Local type inference][9]
  * [Types from libraries][10]
  * [Next steps][11]
* [Type hints cheat sheet][12]
  
  * [Variables][13]
  * [Useful built-in types][14]
  * [Functions][15]
  * [Classes][16]
  * [When you’re puzzled or when things are complicated][17]
  * [Standard “duck types”][18]
  * [Forward references][19]
  * [Decorators][20]
  * [Coroutines and asyncio][21]
* [Using mypy with an existing codebase][22]
  
  * [Start small][23]
  * [Run mypy consistently and prevent regressions][24]
  * [Ignoring errors from certain modules][25]
  * [Fixing errors related to imports][26]
  * [Prioritise annotating widely imported modules][27]
  * [Write annotations as you go][28]
  * [Automate annotation of legacy code][29]
  * [Introduce stricter options][30]
  * [Speed up mypy runs][31]

Type system reference

* [Built-in types][32]
  
  * [Simple types][33]
  * [Any type][34]
  * [Generic types][35]
* [Type inference and type annotations][36]
  
  * [Type inference][37]
  * [Explicit types for variables][38]
  * [Explicit types for collections][39]
  * [Compatibility of container types][40]
  * [Context in type inference][41]
  * [Silencing type errors][42]
* [Kinds of types][43]
  
  * [Class types][44]
  * [The Any type][45]
  * [Tuple types][46]
  * [Callable types (and lambdas)][47]
  * [Union types][48]
  * [Optional types and the None type][49]
  * [Type aliases][50]
  * [Named tuples][51]
  * [The type of class objects][52]
  * [Generators][53]
* [Class basics][54]
  
  * [Instance and class attributes][55]
  * [Annotating __init__ methods][56]
  * [Class attribute annotations][57]
  * [Overriding statically typed methods][58]
  * [Abstract base classes and multiple inheritance][59]
  * [Slots][60]
* [Annotation issues at runtime][61]
  
  * [String literal types and type comments][62]
  * [Future annotations import (PEP 563)][63]
  * [typing.TYPE_CHECKING][64]
  * [Class name forward references][65]
  * [Import cycles][66]
  * [Using classes that are generic in stubs but not at runtime][67]
  * [Using types defined in stubs but not at runtime][68]
  * [Using generic builtins][69]
  * [Using X | Y syntax for Unions][70]
  * [Using new additions to the typing module][71]
* [Protocols and structural subtyping][72]
  
  * [Predefined protocols][73]
  * [Simple user-defined protocols][74]
  * [Defining subprotocols and subclassing protocols][75]
  * [Invariance of protocol attributes][76]
  * [Recursive protocols][77]
  * [Using isinstance() with protocols][78]
  * [Callback protocols][79]
  * [Binding of types in protocol attributes][80]
  * [Predefined protocol reference][81]
* [Dynamically typed code][82]
  
  * [Operations on Any values][83]
  * [Any vs. object][84]
* [Type narrowing][85]
  
  * [Type narrowing expressions][86]
  * [Casts][87]
  * [User-Defined Type Guards][88]
  * [TypeIs][89]
  * [Limitations][90]
* [Duck type compatibility][91]
* [Stub files][92]
  
  * [Creating a stub][93]
  * [Stub file syntax][94]
  * [Using stub file syntax at runtime][95]
* [Generics][96]
  
  * [Defining generic classes][97]
  * [Defining subclasses of generic classes][98]
  * [Generic functions][99]
  * [Type variables with upper bounds][100]
  * [Generic methods and generic self][101]
  * [Automatic self types using typing.Self][102]
  * [Variance of generic types][103]
  * [Type variables with value restriction][104]
  * [Declaring decorators][105]
  * [Generic protocols][106]
  * [Generic type aliases][107]
  * [Differences between the new and old syntax][108]
  * [Generic class internals][109]
* [More types][110]
  
  * [The NoReturn type][111]
  * [NewTypes][112]
  * [Function overloading][113]
  * [Advanced uses of self-types][114]
  * [Typing async/await][115]
* [Literal types and Enums][116]
  
  * [Literal types][117]
  * [Enums][118]
* [TypedDict][119]
  
  * [Totality][120]
  * [Supported operations][121]
  * [Class-based syntax][122]
  * [Mixing required and non-required items][123]
  * [Read-only items][124]
  * [Unions of TypedDicts][125]
  * [Inline TypedDict types][126]
* [Final names, methods and classes][127]
  
  * [Final names][128]
  * [Final methods][129]
  * [Final classes][130]
* [Metaclasses][131]
  
  * [Defining a metaclass][132]
  * [Metaclass usage example][133]
  * [Gotchas and limitations of metaclass support][134]

Configuring and running mypy

* [Running mypy and managing imports][135]
  
  * [Specifying code to be checked][136]
  * [Reading a list of files from a file][137]
  * [Mapping file paths to modules][138]
  * [How mypy handles imports][139]
  * [Missing imports][140]
  * [How imports are found][141]
  * [Following imports][142]
* [The mypy command line][143]
  
  * [Specifying what to type check][144]
  * [Optional arguments][145]
  * [Config file][146]
  * [Import discovery][147]
  * [Platform configuration][148]
  * [Disallow dynamic typing][149]
  * [Untyped definitions and calls][150]
  * [None and Optional handling][151]
  * [Configuring warnings][152]
  * [Miscellaneous strictness flags][153]
  * [Configuring error messages][154]
  * [Incremental mode][155]
  * [Advanced options][156]
  * [Report generation][157]
  * [Enabling incomplete/experimental features][158]
  * [Miscellaneous][159]
* [The mypy configuration file][160]
  
  * [Config file format][161]
  * [Per-module and global options][162]
  * [Inverting option values][163]
  * [Example `mypy.ini`][164]
  * [Import discovery][165]
  * [Platform configuration][166]
  * [Disallow dynamic typing][167]
  * [Untyped definitions and calls][168]
  * [None and Optional handling][169]
  * [Configuring warnings][170]
  * [Suppressing errors][171]
  * [Miscellaneous strictness flags][172]
  * [Configuring error messages][173]
  * [Incremental mode][174]
  * [Advanced options][175]
  * [Report generation][176]
  * [Miscellaneous][177]
  * [Using a pyproject.toml file][178]
  * [Example `pyproject.toml`][179]
* [Inline configuration][180]
  
  * [Configuration comment format][181]
* [Mypy daemon (mypy server)][182]
  
  * [Basic usage][183]
  * [Daemon client commands][184]
  * [Additional daemon flags][185]
  * [Static inference of annotations][186]
  * [Statically inspect expressions][187]
* [Using installed packages][188]
  
  * [Using installed packages with mypy (PEP 561)][189]
  * [Creating PEP 561 compatible packages][190]
* [Extending and integrating mypy][191]
  
  * [Integrating mypy into another Python application][192]
  * [Extending mypy using plugins][193]
  * [Configuring mypy to use plugins][194]
  * [High-level overview][195]
  * [Current list of plugin hooks][196]
  * [Useful tools][197]
* [Automatic stub generation (stubgen)][198]
  
  * [Specifying what to stub][199]
  * [Specifying how to generate stubs][200]
  * [Additional flags][201]
* [Automatic stub testing (stubtest)][202]
  
  * [What stubtest does and does not do][203]
  * [Example][204]
  * [Usage][205]
  * [Allowlist][206]
  * [CLI][207]

Miscellaneous

* [Common issues and solutions][208]
  
  * [No errors reported for obviously wrong code][209]
  * [Spurious errors and locally silencing the checker][210]
  * [Ignoring a whole file][211]
  * [Issues with code at runtime][212]
  * [Mypy runs are slow][213]
  * [Types of empty collections][214]
  * [Redefinitions with incompatible types][215]
  * [Invariance vs covariance][216]
  * [Declaring a supertype as variable type][217]
  * [Complex type tests][218]
  * [Python version and system platform checks][219]
  * [Displaying the type of an expression][220]
  * [Silencing linters][221]
  * [Covariant subtyping of mutable protocol members is rejected][222]
  * [Dealing with conflicting names][223]
  * [Using a development mypy build][224]
  * [Variables vs type aliases][225]
  * [Incompatible overrides][226]
  * [Unreachable code][227]
  * [Narrowing and inner functions][228]
  * [Incorrect use of `Self`][229]
* [Supported Python features][230]
  
  * [Runtime definition of methods and functions][231]
* [Error codes][232]
  
  * [Silencing errors based on error codes][233]
  * [Enabling/disabling specific error codes globally][234]
  * [Per-module enabling/disabling error codes][235]
  * [Subcodes of error codes][236]
  * [Requiring error codes][237]
* [Error codes enabled by default][238]
  
  * [Check that attribute exists [attr-defined]][239]
  * [Check that attribute exists in each union item [union-attr]][240]
  * [Check that name is defined [name-defined]][241]
  * [Check that a variable is not used before it’s defined [used-before-def]][242]
  * [Check arguments in calls [call-arg]][243]
  * [Check argument types [arg-type]][244]
  * [Check calls to overloaded functions [call-overload]][245]
  * [Check validity of types [valid-type]][246]
  * [Check the validity of a class’s metaclass [metaclass]][247]
  * [Require annotation if variable type is unclear [var-annotated]][248]
  * [Check validity of overrides [override]][249]
  * [Check that function returns a value [return]][250]
  * [Check that functions don’t have empty bodies outside stubs [empty-body]][251]
  * [Check that return value is compatible [return-value]][252]
  * [Check types in assignment statement [assignment]][253]
  * [Check that assignment target is not a method [method-assign]][254]
  * [Check type variable values [type-var]][255]
  * [Check uses of various operators [operator]][256]
  * [Check indexing operations [index]][257]
  * [Check list items [list-item]][258]
  * [Check dict items [dict-item]][259]
  * [Check TypedDict items [typeddict-item]][260]
  * [Check TypedDict Keys [typeddict-unknown-key]][261]
  * [Check that type of target is known [has-type]][262]
  * [Check for an issue with imports [import]][263]
  * [Check that import target can be found [import-not-found]][264]
  * [Check that import target can be found [import-untyped]][265]
  * [Check that each name is defined once [no-redef]][266]
  * [Check that called function returns a value [func-returns-value]][267]
  * [Check instantiation of abstract classes [abstract]][268]
  * [Safe handling of abstract type object types [type-abstract]][269]
  * [Check that call to an abstract method via super is valid [safe-super]][270]
  * [Check the target of NewType [valid-newtype]][271]
  * [Check the return type of __exit__ [exit-return]][272]
  * [Check that naming is consistent [name-match]][273]
  * [Check that literal is used where expected [literal-required]][274]
  * [Check that overloaded functions have an implementation [no-overload-impl]][275]
  * [Check that coroutine return value is used [unused-coroutine]][276]
  * [Warn about top level await expressions [top-level-await]][277]
  * [Warn about await expressions used outside of coroutines [await-not-async]][278]
  * [Check types in assert_type [assert-type]][279]
  * [Check that function isn’t used in boolean context [truthy-function]][280]
  * [Check that string formatting/interpolation is type-safe [str-format]][281]
  * [Check for implicit bytes coercions [str-bytes-safe]][282]
  * [Check that overloaded functions don’t overlap [overload-overlap]][283]
  * [Check for overload signatures that cannot match [overload-cannot-match]][284]
  * [Notify about an annotation in an unchecked function [annotation-unchecked]][285]
  * [Decorator preceding property not supported [prop-decorator]][286]
  * [Report syntax errors [syntax]][287]
  * [ReadOnly key of a TypedDict is mutated [typeddict-readonly-mutated]][288]
  * [Check that `TypeIs` narrows types [narrowed-type-not-subtype]][289]
  * [String appears in a context which expects a TypeForm [maybe-unrecognized-str-typeform]][290]
  * [Miscellaneous checks [misc]][291]
* [Error codes for optional checks][292]
  
  * [Check that type arguments exist [type-arg]][293]
  * [Check that every function has an annotation [no-untyped-def]][294]
  * [Check that cast is not redundant [redundant-cast]][295]
  * [Check that methods do not have redundant Self annotations [redundant-self]][296]
  * [Check that comparisons are overlapping [comparison-overlap]][297]
  * [Check that no untyped functions are called [no-untyped-call]][298]
  * [Check that function does not return Any value [no-any-return]][299]
  * [Check that types have no Any components due to missing imports [no-any-unimported]][300]
  * [Check that statement or expression is unreachable [unreachable]][301]
  * [Check that imported or used feature is deprecated [deprecated]][302]
  * [Check that expression is redundant [redundant-expr]][303]
  * [Warn about variables that are defined only in some execution paths [possibly-undefined]][304]
  * [Check that expression is not implicitly true in boolean context [truthy-bool]][305]
  * [Check that iterable is not implicitly true in boolean context [truthy-iterable]][306]
  * [Check that `# type: ignore` include an error code [ignore-without-code]][307]
  * [Check that awaitable return value is used [unused-awaitable]][308]
  * [Check that `# type: ignore` comment is used [unused-ignore]][309]
  * [Check that `@override` is used when overriding a base class method [explicit-override]][310]
  * [Check that overrides of mutable attributes are safe [mutable-override]][311]
  * [Check that `reveal_type` is imported from typing or typing_extensions [unimported-reveal]][312]
  * [Check that explicit Any type annotations are not allowed [explicit-any]][313]
  * [Check that match statements match exhaustively [exhaustive-match]][314]
  * [Error if an untyped decorator makes a typed function effectively untyped
    [untyped-decorator]][315]
* [Additional features][316]
  
  * [Dataclasses][317]
  * [Data Class Transforms][318]
  * [The attrs package][319]
  * [Using a remote cache to speed up mypy runs][320]
  * [Extended Callable types][321]
* [Frequently Asked Questions][322]
  
  * [Why have both dynamic and static typing?][323]
  * [Would my project benefit from static typing?][324]
  * [Can I use mypy to type check my existing Python code?][325]
  * [Will static typing make my programs run faster?][326]
  * [Is mypy free?][327]
  * [Can I use duck typing with mypy?][328]
  * [I like Python and I have no need for static typing][329]
  * [How are mypy programs different from normal Python?][330]
  * [How is mypy different from Cython?][331]
  * [Does it run on PyPy?][332]
  * [Mypy is a cool project. Can I help?][333]
* [Mypy Release Notes][334]
  
  * [Next Release][335]
  * [Mypy 1.19][336]
  * [Mypy 1.18][337]
  * [Mypy 1.17][338]
  * [Mypy 1.16][339]
  * [Mypy 1.15][340]
  * [Mypy 1.14][341]
  * [Mypy 1.13][342]
  * [Mypy 1.12][343]
  * [Mypy 1.11][344]
  * [Mypy 1.10][345]
  * [Mypy 1.9][346]
  * [Mypy 1.8][347]
  * [Mypy 1.7][348]
  * [Mypy 1.6][349]
  * [Mypy 1.5][350]
  * [Mypy 1.4][351]
  * [Mypy 1.3][352]
  * [Mypy 1.2][353]
  * [Mypy 1.1.1][354]
  * [Mypy 1.0][355]
  * [Previous releases][356]

# Indices and tables[¶][357]

* [Index][358]
* [Search Page][359]

[1]: #welcome-to-mypy-documentation
[2]: https://peps.python.org/pep-0484/
[3]: #contents
[4]: getting_started.html
[5]: getting_started.html#installing-and-running-mypy
[6]: getting_started.html#dynamic-vs-static-typing
[7]: getting_started.html#strict-mode-and-configuration
[8]: getting_started.html#more-complex-types
[9]: getting_started.html#local-type-inference
[10]: getting_started.html#types-from-libraries
[11]: getting_started.html#next-steps
[12]: cheat_sheet_py3.html
[13]: cheat_sheet_py3.html#variables
[14]: cheat_sheet_py3.html#useful-built-in-types
[15]: cheat_sheet_py3.html#functions
[16]: cheat_sheet_py3.html#classes
[17]: cheat_sheet_py3.html#when-you-re-puzzled-or-when-things-are-complicated
[18]: cheat_sheet_py3.html#standard-duck-types
[19]: cheat_sheet_py3.html#forward-references
[20]: cheat_sheet_py3.html#decorators
[21]: cheat_sheet_py3.html#coroutines-and-asyncio
[22]: existing_code.html
[23]: existing_code.html#start-small
[24]: existing_code.html#run-mypy-consistently-and-prevent-regressions
[25]: existing_code.html#ignoring-errors-from-certain-modules
[26]: existing_code.html#fixing-errors-related-to-imports
[27]: existing_code.html#prioritise-annotating-widely-imported-modules
[28]: existing_code.html#write-annotations-as-you-go
[29]: existing_code.html#automate-annotation-of-legacy-code
[30]: existing_code.html#introduce-stricter-options
[31]: existing_code.html#speed-up-mypy-runs
[32]: builtin_types.html
[33]: builtin_types.html#simple-types
[34]: builtin_types.html#any-type
[35]: builtin_types.html#generic-types
[36]: type_inference_and_annotations.html
[37]: type_inference_and_annotations.html#type-inference
[38]: type_inference_and_annotations.html#explicit-types-for-variables
[39]: type_inference_and_annotations.html#explicit-types-for-collections
[40]: type_inference_and_annotations.html#compatibility-of-container-types
[41]: type_inference_and_annotations.html#context-in-type-inference
[42]: type_inference_and_annotations.html#silencing-type-errors
[43]: kinds_of_types.html
[44]: kinds_of_types.html#class-types
[45]: kinds_of_types.html#the-any-type
[46]: kinds_of_types.html#tuple-types
[47]: kinds_of_types.html#callable-types-and-lambdas
[48]: kinds_of_types.html#alternative-union-syntax
[49]: kinds_of_types.html#optional-types-and-the-none-type
[50]: kinds_of_types.html#type-aliases
[51]: kinds_of_types.html#named-tuples
[52]: kinds_of_types.html#the-type-of-class-objects
[53]: kinds_of_types.html#generators
[54]: class_basics.html
[55]: class_basics.html#instance-and-class-attributes
[56]: class_basics.html#annotating-init-methods
[57]: class_basics.html#class-attribute-annotations
[58]: class_basics.html#overriding-statically-typed-methods
[59]: class_basics.html#abstract-base-classes-and-multiple-inheritance
[60]: class_basics.html#slots
[61]: runtime_troubles.html
[62]: runtime_troubles.html#string-literal-types-and-type-comments
[63]: runtime_troubles.html#future-annotations-import-pep-563
[64]: runtime_troubles.html#typing-type-checking
[65]: runtime_troubles.html#class-name-forward-references
[66]: runtime_troubles.html#import-cycles
[67]: runtime_troubles.html#using-classes-that-are-generic-in-stubs-but-not-at-runtime
[68]: runtime_troubles.html#using-types-defined-in-stubs-but-not-at-runtime
[69]: runtime_troubles.html#using-generic-builtins
[70]: runtime_troubles.html#using-x-y-syntax-for-unions
[71]: runtime_troubles.html#using-new-additions-to-the-typing-module
[72]: protocols.html
[73]: protocols.html#predefined-protocols
[74]: protocols.html#simple-user-defined-protocols
[75]: protocols.html#defining-subprotocols-and-subclassing-protocols
[76]: protocols.html#invariance-of-protocol-attributes
[77]: protocols.html#recursive-protocols
[78]: protocols.html#using-isinstance-with-protocols
[79]: protocols.html#callback-protocols
[80]: protocols.html#binding-of-types-in-protocol-attributes
[81]: protocols.html#predefined-protocol-reference
[82]: dynamic_typing.html
[83]: dynamic_typing.html#operations-on-any-values
[84]: dynamic_typing.html#any-vs-object
[85]: type_narrowing.html
[86]: type_narrowing.html#type-narrowing-expressions
[87]: type_narrowing.html#casts
[88]: type_narrowing.html#user-defined-type-guards
[89]: type_narrowing.html#typeis
[90]: type_narrowing.html#limitations
[91]: duck_type_compatibility.html
[92]: stubs.html
[93]: stubs.html#creating-a-stub
[94]: stubs.html#stub-file-syntax
[95]: stubs.html#using-stub-file-syntax-at-runtime
[96]: generics.html
[97]: generics.html#defining-generic-classes
[98]: generics.html#defining-subclasses-of-generic-classes
[99]: generics.html#generic-functions
[100]: generics.html#type-variables-with-upper-bounds
[101]: generics.html#generic-methods-and-generic-self
[102]: generics.html#automatic-self-types-using-typing-self
[103]: generics.html#variance-of-generic-types
[104]: generics.html#type-variables-with-value-restriction
[105]: generics.html#declaring-decorators
[106]: generics.html#generic-protocols
[107]: generics.html#generic-type-aliases
[108]: generics.html#differences-between-the-new-and-old-syntax
[109]: generics.html#generic-class-internals
[110]: more_types.html
[111]: more_types.html#the-noreturn-type
[112]: more_types.html#newtypes
[113]: more_types.html#function-overloading
[114]: more_types.html#advanced-uses-of-self-types
[115]: more_types.html#typing-async-await
[116]: literal_types.html
[117]: literal_types.html#literal-types
[118]: literal_types.html#enums
[119]: typed_dict.html
[120]: typed_dict.html#totality
[121]: typed_dict.html#supported-operations
[122]: typed_dict.html#class-based-syntax
[123]: typed_dict.html#mixing-required-and-non-required-items
[124]: typed_dict.html#read-only-items
[125]: typed_dict.html#unions-of-typeddicts
[126]: typed_dict.html#inline-typeddict-types
[127]: final_attrs.html
[128]: final_attrs.html#final-names
[129]: final_attrs.html#final-methods
[130]: final_attrs.html#final-classes
[131]: metaclasses.html
[132]: metaclasses.html#defining-a-metaclass
[133]: metaclasses.html#metaclass-usage-example
[134]: metaclasses.html#gotchas-and-limitations-of-metaclass-support
[135]: running_mypy.html
[136]: running_mypy.html#specifying-code-to-be-checked
[137]: running_mypy.html#reading-a-list-of-files-from-a-file
[138]: running_mypy.html#mapping-file-paths-to-modules
[139]: running_mypy.html#how-mypy-handles-imports
[140]: running_mypy.html#missing-imports
[141]: running_mypy.html#how-imports-are-found
[142]: running_mypy.html#following-imports
[143]: command_line.html
[144]: command_line.html#specifying-what-to-type-check
[145]: command_line.html#optional-arguments
[146]: command_line.html#config-file
[147]: command_line.html#import-discovery
[148]: command_line.html#platform-configuration
[149]: command_line.html#disallow-dynamic-typing
[150]: command_line.html#untyped-definitions-and-calls
[151]: command_line.html#none-and-optional-handling
[152]: command_line.html#configuring-warnings
[153]: command_line.html#miscellaneous-strictness-flags
[154]: command_line.html#configuring-error-messages
[155]: command_line.html#incremental-mode
[156]: command_line.html#advanced-options
[157]: command_line.html#report-generation
[158]: command_line.html#enabling-incomplete-experimental-features
[159]: command_line.html#miscellaneous
[160]: config_file.html
[161]: config_file.html#config-file-format
[162]: config_file.html#per-module-and-global-options
[163]: config_file.html#inverting-option-values
[164]: config_file.html#example-mypy-ini
[165]: config_file.html#import-discovery
[166]: config_file.html#platform-configuration
[167]: config_file.html#disallow-dynamic-typing
[168]: config_file.html#untyped-definitions-and-calls
[169]: config_file.html#none-and-optional-handling
[170]: config_file.html#configuring-warnings
[171]: config_file.html#suppressing-errors
[172]: config_file.html#miscellaneous-strictness-flags
[173]: config_file.html#configuring-error-messages
[174]: config_file.html#incremental-mode
[175]: config_file.html#advanced-options
[176]: config_file.html#report-generation
[177]: config_file.html#miscellaneous
[178]: config_file.html#using-a-pyproject-toml-file
[179]: config_file.html#example-pyproject-toml
[180]: inline_config.html
[181]: inline_config.html#configuration-comment-format
[182]: mypy_daemon.html
[183]: mypy_daemon.html#basic-usage
[184]: mypy_daemon.html#daemon-client-commands
[185]: mypy_daemon.html#additional-daemon-flags
[186]: mypy_daemon.html#static-inference-of-annotations
[187]: mypy_daemon.html#statically-inspect-expressions
[188]: installed_packages.html
[189]: installed_packages.html#using-installed-packages-with-mypy-pep-561
[190]: installed_packages.html#creating-pep-561-compatible-packages
[191]: extending_mypy.html
[192]: extending_mypy.html#integrating-mypy-into-another-python-application
[193]: extending_mypy.html#extending-mypy-using-plugins
[194]: extending_mypy.html#configuring-mypy-to-use-plugins
[195]: extending_mypy.html#high-level-overview
[196]: extending_mypy.html#current-list-of-plugin-hooks
[197]: extending_mypy.html#useful-tools
[198]: stubgen.html
[199]: stubgen.html#specifying-what-to-stub
[200]: stubgen.html#specifying-how-to-generate-stubs
[201]: stubgen.html#additional-flags
[202]: stubtest.html
[203]: stubtest.html#what-stubtest-does-and-does-not-do
[204]: stubtest.html#example
[205]: stubtest.html#usage
[206]: stubtest.html#allowlist
[207]: stubtest.html#cli
[208]: common_issues.html
[209]: common_issues.html#no-errors-reported-for-obviously-wrong-code
[210]: common_issues.html#spurious-errors-and-locally-silencing-the-checker
[211]: common_issues.html#ignoring-a-whole-file
[212]: common_issues.html#issues-with-code-at-runtime
[213]: common_issues.html#mypy-runs-are-slow
[214]: common_issues.html#types-of-empty-collections
[215]: common_issues.html#redefinitions-with-incompatible-types
[216]: common_issues.html#invariance-vs-covariance
[217]: common_issues.html#declaring-a-supertype-as-variable-type
[218]: common_issues.html#complex-type-tests
[219]: common_issues.html#python-version-and-system-platform-checks
[220]: common_issues.html#displaying-the-type-of-an-expression
[221]: common_issues.html#silencing-linters
[222]: common_issues.html#covariant-subtyping-of-mutable-protocol-members-is-rejected
[223]: common_issues.html#dealing-with-conflicting-names
[224]: common_issues.html#using-a-development-mypy-build
[225]: common_issues.html#variables-vs-type-aliases
[226]: common_issues.html#incompatible-overrides
[227]: common_issues.html#unreachable-code
[228]: common_issues.html#narrowing-and-inner-functions
[229]: common_issues.html#incorrect-use-of-self
[230]: supported_python_features.html
[231]: supported_python_features.html#runtime-definition-of-methods-and-functions
[232]: error_codes.html
[233]: error_codes.html#silencing-errors-based-on-error-codes
[234]: error_codes.html#enabling-disabling-specific-error-codes-globally
[235]: error_codes.html#per-module-enabling-disabling-error-codes
[236]: error_codes.html#subcodes-of-error-codes
[237]: error_codes.html#requiring-error-codes
[238]: error_code_list.html
[239]: error_code_list.html#check-that-attribute-exists-attr-defined
[240]: error_code_list.html#check-that-attribute-exists-in-each-union-item-union-attr
[241]: error_code_list.html#check-that-name-is-defined-name-defined
[242]: error_code_list.html#check-that-a-variable-is-not-used-before-it-s-defined-used-before-def
[243]: error_code_list.html#check-arguments-in-calls-call-arg
[244]: error_code_list.html#check-argument-types-arg-type
[245]: error_code_list.html#check-calls-to-overloaded-functions-call-overload
[246]: error_code_list.html#check-validity-of-types-valid-type
[247]: error_code_list.html#check-the-validity-of-a-class-s-metaclass-metaclass
[248]: error_code_list.html#require-annotation-if-variable-type-is-unclear-var-annotated
[249]: error_code_list.html#check-validity-of-overrides-override
[250]: error_code_list.html#check-that-function-returns-a-value-return
[251]: error_code_list.html#check-that-functions-don-t-have-empty-bodies-outside-stubs-empty-body
[252]: error_code_list.html#check-that-return-value-is-compatible-return-value
[253]: error_code_list.html#check-types-in-assignment-statement-assignment
[254]: error_code_list.html#check-that-assignment-target-is-not-a-method-method-assign
[255]: error_code_list.html#check-type-variable-values-type-var
[256]: error_code_list.html#check-uses-of-various-operators-operator
[257]: error_code_list.html#check-indexing-operations-index
[258]: error_code_list.html#check-list-items-list-item
[259]: error_code_list.html#check-dict-items-dict-item
[260]: error_code_list.html#check-typeddict-items-typeddict-item
[261]: error_code_list.html#check-typeddict-keys-typeddict-unknown-key
[262]: error_code_list.html#check-that-type-of-target-is-known-has-type
[263]: error_code_list.html#check-for-an-issue-with-imports-import
[264]: error_code_list.html#check-that-import-target-can-be-found-import-not-found
[265]: error_code_list.html#check-that-import-target-can-be-found-import-untyped
[266]: error_code_list.html#check-that-each-name-is-defined-once-no-redef
[267]: error_code_list.html#check-that-called-function-returns-a-value-func-returns-value
[268]: error_code_list.html#check-instantiation-of-abstract-classes-abstract
[269]: error_code_list.html#safe-handling-of-abstract-type-object-types-type-abstract
[270]: error_code_list.html#check-that-call-to-an-abstract-method-via-super-is-valid-safe-super
[271]: error_code_list.html#check-the-target-of-newtype-valid-newtype
[272]: error_code_list.html#check-the-return-type-of-exit-exit-return
[273]: error_code_list.html#check-that-naming-is-consistent-name-match
[274]: error_code_list.html#check-that-literal-is-used-where-expected-literal-required
[275]: error_code_list.html#check-that-overloaded-functions-have-an-implementation-no-overload-impl
[276]: error_code_list.html#check-that-coroutine-return-value-is-used-unused-coroutine
[277]: error_code_list.html#warn-about-top-level-await-expressions-top-level-await
[278]: error_code_list.html#warn-about-await-expressions-used-outside-of-coroutines-await-not-async
[279]: error_code_list.html#check-types-in-assert-type-assert-type
[280]: error_code_list.html#check-that-function-isn-t-used-in-boolean-context-truthy-function
[281]: error_code_list.html#check-that-string-formatting-interpolation-is-type-safe-str-format
[282]: error_code_list.html#check-for-implicit-bytes-coercions-str-bytes-safe
[283]: error_code_list.html#check-that-overloaded-functions-don-t-overlap-overload-overlap
[284]: error_code_list.html#check-for-overload-signatures-that-cannot-match-overload-cannot-match
[285]: error_code_list.html#notify-about-an-annotation-in-an-unchecked-function-annotation-unchecked
[286]: error_code_list.html#decorator-preceding-property-not-supported-prop-decorator
[287]: error_code_list.html#report-syntax-errors-syntax
[288]: error_code_list.html#readonly-key-of-a-typeddict-is-mutated-typeddict-readonly-mutated
[289]: error_code_list.html#check-that-typeis-narrows-types-narrowed-type-not-subtype
[290]: error_code_list.html#string-appears-in-a-context-which-expects-a-typeform-maybe-unrecognized-
str-typeform
[291]: error_code_list.html#miscellaneous-checks-misc
[292]: error_code_list2.html
[293]: error_code_list2.html#check-that-type-arguments-exist-type-arg
[294]: error_code_list2.html#check-that-every-function-has-an-annotation-no-untyped-def
[295]: error_code_list2.html#check-that-cast-is-not-redundant-redundant-cast
[296]: error_code_list2.html#check-that-methods-do-not-have-redundant-self-annotations-redundant-sel
f
[297]: error_code_list2.html#check-that-comparisons-are-overlapping-comparison-overlap
[298]: error_code_list2.html#check-that-no-untyped-functions-are-called-no-untyped-call
[299]: error_code_list2.html#check-that-function-does-not-return-any-value-no-any-return
[300]: error_code_list2.html#check-that-types-have-no-any-components-due-to-missing-imports-no-any-u
nimported
[301]: error_code_list2.html#check-that-statement-or-expression-is-unreachable-unreachable
[302]: error_code_list2.html#check-that-imported-or-used-feature-is-deprecated-deprecated
[303]: error_code_list2.html#check-that-expression-is-redundant-redundant-expr
[304]: error_code_list2.html#warn-about-variables-that-are-defined-only-in-some-execution-paths-poss
ibly-undefined
[305]: error_code_list2.html#check-that-expression-is-not-implicitly-true-in-boolean-context-truthy-
bool
[306]: error_code_list2.html#check-that-iterable-is-not-implicitly-true-in-boolean-context-truthy-it
erable
[307]: error_code_list2.html#check-that-type-ignore-include-an-error-code-ignore-without-code
[308]: error_code_list2.html#check-that-awaitable-return-value-is-used-unused-awaitable
[309]: error_code_list2.html#check-that-type-ignore-comment-is-used-unused-ignore
[310]: error_code_list2.html#check-that-override-is-used-when-overriding-a-base-class-method-explici
t-override
[311]: error_code_list2.html#check-that-overrides-of-mutable-attributes-are-safe-mutable-override
[312]: error_code_list2.html#check-that-reveal-type-is-imported-from-typing-or-typing-extensions-uni
mported-reveal
[313]: error_code_list2.html#check-that-explicit-any-type-annotations-are-not-allowed-explicit-any
[314]: error_code_list2.html#check-that-match-statements-match-exhaustively-exhaustive-match
[315]: error_code_list2.html#error-if-an-untyped-decorator-makes-a-typed-function-effectively-untype
d-untyped-decorator
[316]: additional_features.html
[317]: additional_features.html#dataclasses
[318]: additional_features.html#data-class-transforms
[319]: additional_features.html#the-attrs-package
[320]: additional_features.html#using-a-remote-cache-to-speed-up-mypy-runs
[321]: additional_features.html#extended-callable-types
[322]: faq.html
[323]: faq.html#why-have-both-dynamic-and-static-typing
[324]: faq.html#would-my-project-benefit-from-static-typing
[325]: faq.html#can-i-use-mypy-to-type-check-my-existing-python-code
[326]: faq.html#will-static-typing-make-my-programs-run-faster
[327]: faq.html#is-mypy-free
[328]: faq.html#can-i-use-duck-typing-with-mypy
[329]: faq.html#i-like-python-and-i-have-no-need-for-static-typing
[330]: faq.html#how-are-mypy-programs-different-from-normal-python
[331]: faq.html#how-is-mypy-different-from-cython
[332]: faq.html#does-it-run-on-pypy
[333]: faq.html#mypy-is-a-cool-project-can-i-help
[334]: changelog.html
[335]: changelog.html#next-release
[336]: changelog.html#mypy-1-19
[337]: changelog.html#mypy-1-18
[338]: changelog.html#mypy-1-17
[339]: changelog.html#mypy-1-16
[340]: changelog.html#mypy-1-15
[341]: changelog.html#mypy-1-14
[342]: changelog.html#mypy-1-13
[343]: changelog.html#mypy-1-12
[344]: changelog.html#mypy-1-11
[345]: changelog.html#mypy-1-10
[346]: changelog.html#mypy-1-9
[347]: changelog.html#mypy-1-8
[348]: changelog.html#mypy-1-7
[349]: changelog.html#mypy-1-6
[350]: changelog.html#mypy-1-5
[351]: changelog.html#mypy-1-4
[352]: changelog.html#mypy-1-3
[353]: changelog.html#mypy-1-2
[354]: changelog.html#mypy-1-1-1
[355]: changelog.html#mypy-1-0
[356]: changelog.html#previous-releases
[357]: #indices-and-tables
[358]: genindex.html
[359]: search.html
