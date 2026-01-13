* User guide
* [ View page source][1]

# User guide[][2]

## Checking types directly[][3]

The most straightfoward way to do type checking with Typeguard is with [`check_type()`][4]. It can
be used as as a beefed-up version of [`isinstance()`][5] that also supports checking against
annotations in the [`typing`][6] module:

from typeguard import check_type

# Raises TypeCheckError if there's a problem
check_type([1234], List[int])

It’s also useful for safely casting the types of objects dynamically constructed from external
sources:

import json
from typing import List, TypedDict

from typeguard import check_type

# Example contents of "people.json":
# [
#   {"name": "John Smith", "phone": "111-123123", "address": "123 Main Street"},
#   {"name": "Jane Smith", "phone": "111-456456", "address": "123 Main Street"}
# ]

class Person(TypedDict):
    name: str
    phone: str
    address: str

 with open("people.json") as f:
    people = check_type(json.load(f), List[Person])

With this code, static type checkers will recognize the type of `people` to be `List[Person]`.

## Using the decorator[][7]

The [`@typechecked`][8] decorator is the simplest way to add type checking on a case-by-case basis.
It can be used on functions directly, or on entire classes, in which case all the contained methods
are instrumented:

from typeguard import typechecked

@typechecked
def some_function(a: int, b: float, c: str, *args: str) -> bool:
    ...
    return retval

@typechecked
class SomeClass:
    # All type annotated methods (including static and class methods and properties)
    # are type checked.
    # Does not apply to inner classes!
    def method(x: int) -> int:
        ...

The decorator instruments functions by fetching the source code, parsing it to an abstract syntax
tree using [`ast.parse()`][9], modifying it to add type checking, and finally compiling the modified
AST into byte code. This code is then used to make a new function object that is used to replace the
original one.

To explicitly set type checking options on a per-function basis, you can pass them as keyword
arguments to [`@typechecked`][10]:

from typeguard import CollectionCheckStrategy, typechecked

@typechecked(collection_check_strategy=CollectionCheckStrategy.ALL_ITEMS)
def some_function(a: int, b: float, c: str, *args: str) -> bool:
    ...
    return retval

This also allows you to override the global options for specific functions when using the import
hook.

Note

You should always place this decorator closest to the original function, as it will not work when
there is another decorator wrapping the function. For the same reason, when you use it on a class
that has wrapping decorators on its methods, such methods will not be instrumented. In contrast, the
import hook has no such restrictions.

## Using the import hook[][11]

The import hook, when active, automatically instruments all type annotated functions to type check
arguments, return values and values yielded by or sent to generator functions. This allows for a
non-invasive method of run time type checking. This method does not modify the source code on disk,
but instead modifies its AST (Abstract Syntax Tree) when the module is loaded.

Using the import hook is as straightforward as installing it before you import any modules you wish
to be type checked. Give it the name of your top level package (or a list of package names):

from typeguard import install_import_hook

install_import_hook('myapp')
from myapp import some_module  # import only AFTER installing the hook, or it won't take effect

If you wish, you can uninstall the import hook:

manager = install_import_hook('myapp')
from myapp import some_module
manager.uninstall()

or using the context manager approach:

with install_import_hook('myapp'):
    from myapp import some_module

You can also customize the logic used to select which modules to instrument:

from typeguard import TypeguardFinder, install_import_hook

class CustomFinder(TypeguardFinder):
    def should_instrument(self, module_name: str):
        # disregard the module names list and instrument all loaded modules
        return True

install_import_hook('', cls=CustomFinder)

## Notes on forward reference handling[][12]

The internal type checking functions, injected to instrumented code by either [`@typechecked`][13]
or the import hook, use the “naked” versions of any annotations, undoing any quotations in them (and
the effects of `from __future__ import annotations`). As such, in instrumented code, the
[`forward_ref_policy`][14] only applies when using type variables containing forward references, or
type aliases likewise containing forward references.

To facilitate the use of types only available to static type checkers, Typeguard recognizes
module-level imports guarded by `if typing.TYPE_CHECKING:` or `if TYPE_CHECKING:` (add the
appropriate [`typing`][15] imports). Imports made within such blocks on the module level will be
replaced in calls to internal type checking functions with [`Any`][16].

## Using the pytest plugin[][17]

Typeguard comes with a plugin for pytest (v7.0 or newer) that installs the import hook (explained in
the previous section). To use it, run `pytest` with the appropriate `--typeguard-packages` option.
For example, if you wanted to instrument the `foo.bar` and `xyz` packages for type checking, you can
do the following:

pytest --typeguard-packages=foo.bar,xyz

It is also possible to set option for the pytest plugin using pytest’s own configuration. For
example, here’s how you might specify several options in `pyproject.toml`:

[tool.pytest.ini_options]
typeguard-packages = """
foo.bar
xyz"""
typeguard-debug-instrumentation = true
typeguard-typecheck-fail-callback = "mypackage:failcallback"
typeguard-forward-ref-policy = "ERROR"
typeguard-collection-check-strategy = "ALL_ITEMS"

See the next section for details on how the individual options work.

Note

There is currently no support for specifying a customized module finder.

## Setting configuration options[][18]

There are several configuration options that can be set that influence how type checking is done.
The [`typeguard.config`][19] (which is of type [`TypeCheckConfiguration`][20]) controls the options
applied to code instrumented via either [`@typechecked`][21] or the import hook. The
[`check_type()`][22], function, however, uses the built-in defaults and is not affected by the
global configuration, so you must pass any configuration overrides explicitly with each call.

You can also override specific configuration options in instrumented functions (or entire classes)
by passing keyword arguments to [`@typechecked`][23]. You can do this even if you’re using the
import hook, as the import hook will remove the decorator to ensure that no double instrumentation
takes place. If you’re using the import hook to type check your code only during tests and don’t
want to include `typeguard` as a run-time dependency, you can use a dummy replacement for the
decorator.

For example, the following snippet will only import the decorator during a [pytest][24] run:

import sys

if "pytest" in sys.modules:
    from typeguard import typechecked
else:
    from typing import TypeVar
    _T = TypeVar("_T")

    def typechecked(target: _T, **kwargs) -> _T:
        return target if target else typechecked

## Suppressing type checks[][25]

### Temporarily disabling type checks[][26]

If you need to temporarily suppress type checking, you can use the [`suppress_type_checks()`][27]
function, either as a context manager or a decorator, to skip the checks:

from typeguard import check_type, suppress_type_checks

with suppress_type_checks():
    check_type(1, str)  # would fail without the suppression

@suppress_type_checks
def my_suppressed_function(x: int) -> None:
    ...

Suppression state is tracked globally. Suppression ends only when all the context managers have
exited and all calls to decorated functions have returned.

### Permanently suppressing type checks for selected functions[][28]

To exclude specific functions from run time type checking, you can use one of the following
decorators:

> * [`@typeguard_ignore`][29]: prevents the decorated function from being instrumentated by the
>   import hook
> * [`@no_type_check`][30]: as above, but disables static type checking too

For example, calling the function defined below will not result in a type check error when the
containing module is instrumented by the import hook:

from typeguard import typeguard_ignore

@typeguard_ignore
def f(x: int) -> int:
    return str(x)

Warning

The [`@no_type_check_decorator`][31] decorator is not currently recognized by Typeguard.

## Suppressing the `@typechecked` decorator in production[][32]

If you’re using the [`@typechecked`][33] decorator to gradually introduce run-time type checks to
your code base, you can disable the checks in production by running Python in optimized mode (as
opposed to debug mode which is the default mode). You can do this by either starting Python with the
`-O` or `-OO` option, or by setting the [PYTHONOPTIMIZE][34] environment variable. This will cause
[`@typechecked`][35] to become a no-op when the import hook is not being used to instrument the
code.

## Debugging instrumented code[][36]

If you find that your code behaves in an unexpected fashion with the Typeguard instrumentation in
place, you should set the `typeguard.config.debug_instrumentation` flag to `True`. This will print
all the instrumented code after the modifications, which you can check to find the reason for the
unexpected behavior.

If you’re using the pytest plugin, you can also pass the `--typeguard-debug-instrumentation` and
`-s` flags together for the same effect.

[ Previous][37] [Next ][38]

© Copyright 2015, Alex Grönholm.

Built with [Sphinx][39] using a [theme][40] provided by [Read the Docs][41].

[1]: _sources/userguide.rst.txt
[2]: #user-guide
[3]: #checking-types-directly
[4]: api.html#typeguard.check_type
[5]: https://docs.python.org/3/library/functions.html#isinstance
[6]: https://docs.python.org/3/library/typing.html#module-typing
[7]: #using-the-decorator
[8]: api.html#typeguard.typechecked
[9]: https://docs.python.org/3/library/ast.html#ast.parse
[10]: api.html#typeguard.typechecked
[11]: #using-the-import-hook
[12]: #notes-on-forward-reference-handling
[13]: api.html#typeguard.typechecked
[14]: api.html#typeguard.TypeCheckConfiguration.forward_ref_policy
[15]: https://docs.python.org/3/library/typing.html#module-typing
[16]: https://docs.python.org/3/library/typing.html#typing.Any
[17]: #using-the-pytest-plugin
[18]: #setting-configuration-options
[19]: api.html#typeguard.config
[20]: api.html#typeguard.TypeCheckConfiguration
[21]: api.html#typeguard.typechecked
[22]: api.html#typeguard.check_type
[23]: api.html#typeguard.typechecked
[24]: https://docs.pytest.org/
[25]: #suppressing-type-checks
[26]: #temporarily-disabling-type-checks
[27]: api.html#typeguard.suppress_type_checks
[28]: #permanently-suppressing-type-checks-for-selected-functions
[29]: api.html#typeguard.typeguard_ignore
[30]: https://docs.python.org/3/library/typing.html#typing.no_type_check
[31]: https://docs.python.org/3/library/typing.html#typing.no_type_check_decorator
[32]: #suppressing-the-typechecked-decorator-in-production
[33]: api.html#typeguard.typechecked
[34]: https://docs.python.org/3/using/cmdline.html#envvar-PYTHONOPTIMIZE
[35]: api.html#typeguard.typechecked
[36]: #debugging-instrumented-code
[37]: index.html
[38]: features.html
[39]: https://www.sphinx-doc.org/
[40]: https://github.com/readthedocs/sphinx_rtd_theme
[41]: https://readthedocs.org
