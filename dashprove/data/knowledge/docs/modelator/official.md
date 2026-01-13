# Modelator

[[License]][1] [[Release]][2] [[Build Status]][3]

Modelator is a tool that enables automatic generation of tests from models. Modelator takes [TLA+
models][4] as its input and generates tests that can be executed against an implementation of the
model.

Under the hood, Modelator uses [Apalache][5], our in-house model checker, to generate tests from
models.

Modelator is used by [Atomkraft][6], a testing framework for the [Cosmos blockchains network][7].

# Installing Modelator

First, make sure your system has

* `Python 3.8` or newer
* `Java 17` or newer

To install Modelator, simply run `pip install modelator`. That's it! Please verify that the tool is
working by writing `modelator` on the command line. You should see something like this:

[[Modelator CLI]][8]

For detailed installation instructions and clarifications of dependencies, check
[INSTALLATION.md][9]

# Using Modelator

For a full example of running Modelator together with the system implementation and the
corresponding TLA+ model, see the [Towers of Hanoi example][10].

To see all commands of the Modelator CLI, run `modelator --help`.

The command `apalache` provides the info about the current version of Apalache installed and enables
you to download different versions. (In order to check the usage of this command, run `modelator
apalache --help`. Do the same for details of all other commands.)

Commands `load`, `info`, and `reset` are used to manipulate the default value for a TLA+ model. (A
default version is not strictly necessary since a model can always be given as an argument.)

The most useful commands are `simulate`, `sample`, and `check`. Command `simulate` will generate a
number of (randomly chosen) behaviors described by the model. If you would like to obtain a
particular model behavior, use command `sample`: it will generate behaviors described by a TLA+
predicate. Finally, to check if a TLA+ predicate is an invariant of the model, use `check`. To see
all the options of these commands, run `modelator simulate --help`, `modelator sample --help`, or
`modelator check --help`.

## Contributing

If you encounter a bug or have a an idea for a new feature of Modelator, please submit [an
issue][11].

If you wish to contribute code to Modelator, set up the repository as follows:

`git clone git@github.com/informalsystems/modelator
cd modelator
poetry install
poetry shell
`

## License

Copyright © 2021-2022 Informal Systems Inc. and modelator authors.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use the files in this
repository except in compliance with the License. You may obtain a copy of the License at

`https://www.apache.org/licenses/LICENSE-2.0
`

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.

[1]: /informalsystems/modelator/blob/dev/LICENSE
[2]: https://pypi.org/project/modelator
[3]: https://github.com/informalsystems/modelator/actions/workflows/python.yml
[4]: https://mbt.informal.systems/docs/tla_basics_tutorials/tutorial.html
[5]: https://apalache.informal.systems
[6]: https://github.com/informalsystems/atomkraft
[7]: https://cosmos.network
[8]: /informalsystems/modelator/blob/dev/docs/images/modelator_cli.png
[9]: /informalsystems/modelator/blob/dev/INSTALLATION.md
[10]: /informalsystems/modelator/blob/dev/examples/hanoi
[11]: https://github.com/informalsystems/modelator/issues
