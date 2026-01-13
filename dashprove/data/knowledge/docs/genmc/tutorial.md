# GenMC: Generic Model Checking for C Programs

GenMC is a stateless model checker for C programs that works on the level of LLVM Intermediate
Representation.

This repository mirrors an internal repository and is only updated periodically. For changes between
different versions please refer to the CHANGELOG.

Author: Michalis Kokologiannakis.

* [Getting GenMC][1]
* [Usage][2]
* [Troubleshooting][3]
* [License][4]
* [Contact][5]

## Getting GenMC

### Using Docker

To pull a container containing GenMC from [Docker Hub][6] please issue the following command:

`       docker pull genmc/genmc
`

### Building from source

#### Dependencies

You will need a C++20-compliant compiler (recommended: g++ >= 14) and an LLVM installation. The LLVM
versions currently supported are: 15.0.0, 16.0.0, 17.0.0, 18.0.0, 19.0.0, 20.0.0.

GNU/Linux

In order to use the tool on a Debian-based installation, you need the following packages:

`       cmake clang  llvm  llvm-dev  libffi-dev
        zlib1g-dev libedit-dev
`

`clang` and `llvm` are only necessary when building the `genmc` executable. To run the default
tests, `bc` is also required.

Max OS X

Using `brew`, the following packages are necessary:

`       cmake llvm libffi
`

#### Installing

For a default build issue the command below. The usual `CMAKE_PREFIX_PATH` can be used to specify
the paths of the libraries required.

`       cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -B RelWithDebInfo -S .
        cmake --build RelWithDebInfo
`

This will leave the `genmc` executable in the `RelWithDebInfo` directory. You can either run it from
there (as in the examples below), or issue `cmake --install .`.

To run a subset of all the tests that come with the system to see if the system was built correctly
or not:

`       cd RelWithDebInfo && ctest -R fast-driver
`

## Usage

* To see a list of available options run:
  
  `  ./RelWithDebInfo/genmc --help
  `
* To run a particular testcase run:
  
  `  ./RelWithDebInfo/genmc [options] <file>
  `
* For more detailed usage examples please refer to the [manual][7].

## Troubleshooting

* Undefined references to symbols that involve types `std::__cxx11` during linking:
  
  This probably indicates that you are using an old version of LLVM with a new version of libstdc++.
  Configuring with the following flags should fix the problem:
  
  `       CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" ./configure --with-llvm=LLVM_PATH
  `
* Linking problems under Arch Linux:
  
  Arch Linux does not provide the `libclang*.a` files required for linking against `clang`
  libraries. In order to for linking to succeed, please change the last line in `src/Makefile.am` to
  the following:
  
  `       genmc_LDADD = libgenmc.a -lclang-cpp
  `

## License

GenMC (with the exception of some files, see [Exceptions][8]) is distributed under either the Apache
License, Version 2.0 (see [LICENSE-APACHE][9]) or the MIT license (see [LICENSE-MIT][10]).

### Exceptions

Part of the code in the files listed below are originating from the [LLVM Compiler Framework][11],
version 3.5. These parts are licensed under the University of Illinois/NCSA Open Source License as
well as under Apache/MIT. Please see the LLVMLICENSE file for details on the University of
Illinois/NCSA Open Source License.

`       src/Interpreter.h
        src/Interpreter.cpp
        src/Execution.cpp
        src/ExternalFunctions.cpp
`

Also note that the `include` and `tests` directories contain files that are copied from other
codebases. Please inspect the source code for more information on the license of the respective
files.

## Contact

For feedback, questions, and bug reports please send an e-mail to `michalis AT mpi-sws DOT org`.

[1]: #getting-genmc
[2]: #usage
[3]: #troubleshooting
[4]: #license
[5]: #contact
[6]: https://hub.docker.com
[7]: /MPI-SWS/genmc/blob/master/doc/manual.md
[8]: #exceptions
[9]: /MPI-SWS/genmc/blob/master/LICENSE-APACHE
[10]: /MPI-SWS/genmc/blob/master/LICENSE-MIT
[11]: https://llvm.org
