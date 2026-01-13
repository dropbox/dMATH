* Typeguard
* [ View page source][1]

# Typeguard[][2]

[[Build Status] ][3] [[Code Coverage] ][4] [[Documentation] ][5]

This library provides run-time type checking for functions defined with [PEP 484][6] argument (and
return) type annotations, and any arbitrary objects. It can be used together with static type
checkers as an additional layer of type safety, to catch type violations that could only be detected
at run time.

Two principal ways to do type checking are provided:

1. The `check_type` function:
   
   * like `isinstance()`, but supports arbitrary type annotations (within limits)
   * can be used as a `cast()` replacement, but with actual checking of the value
2. Code instrumentation:
   
   * entire modules, or individual functions (via `@typechecked`) are recompiled, with type checking
     code injected into them
   * automatically checks function arguments, return values and assignments to annotated local
     variables
   * for generator functions (regular and async), checks yield and send values
   * requires the original source code of the instrumented module(s) to be accessible

Two options are provided for code instrumentation:

1. the `@typechecked` function:
   
   * can be applied to functions individually
2. the import hook (`typeguard.install_import_hook()`):
   
   * automatically instruments targeted modules on import
   * no manual code changes required in the target modules
   * requires the import hook to be installed before the targeted modules are imported
   * may clash with other import hooks

## Quick links[][7]

* [User guide][8]
* [Features][9]
* [Extending Typeguard][10]
* [Contributing to Typeguard][11]
* [API reference][12]
* [Version history][13]
[Next ][14]

© Copyright 2015, Alex Grönholm.

Built with [Sphinx][15] using a [theme][16] provided by [Read the Docs][17].

[1]: _sources/index.rst.txt
[2]: #typeguard
[3]: https://github.com/agronholm/typeguard/actions/workflows/test.yml
[4]: https://coveralls.io/github/agronholm/typeguard?branch=master
[5]: https://typeguard.readthedocs.io/en/latest/?badge=latest
[6]: https://www.python.org/dev/peps/pep-0484/
[7]: #quick-links
[8]: userguide.html
[9]: features.html
[10]: extending.html
[11]: contributing.html
[12]: api.html
[13]: versionhistory.html
[14]: userguide.html
[15]: https://www.sphinx-doc.org/
[16]: https://github.com/readthedocs/sphinx_rtd_theme
[17]: https://readthedocs.org
