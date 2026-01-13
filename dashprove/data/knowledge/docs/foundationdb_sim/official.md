[[FoundationDB logo]][1]

[[Build Status]][2]

FoundationDB is a distributed database designed to handle large volumes of structured data across
clusters of commodity servers. It organizes data as an ordered key-value store and employs ACID
transactions for all operations. It is especially well-suited for read/write workloads, but also has
excellent performance for write-intensive workloads. Users interact with the database using API
language binding.

To learn more about FoundationDB, visit [foundationdb.org][3]

## Documentation

Documentation can be found online at [https://apple.github.io/foundationdb/][4]. The documentation
covers details of API usage, background information on design philosophy, and extensive usage
examples. Docs are built from the [source in this repo][5].

## Forums

[The FoundationDB Forums][6] are the home for most of the discussion and communication about the
FoundationDB project. We welcome your participation! We want FoundationDB to be a great project to
be a part of, and as part of that, we have established a [Code of Conduct][7] to define what
constitutes permissible modes of interaction.

## Contributing

Contributing to FoundationDB can be in contributions to the codebase, sharing your experience and
insights in the community on the Forums, or contributing to projects that make use of FoundationDB.
Please see the [contributing guide][8] for more specifics.

## Getting Started

### Latest Stable Releases

The latest stable releases are (were) versions that are recommended for production use, which have
been extensively validated via simulation and real cluster tests and used in our production
environment.

──────┬─────────────────────────┬────────────
Branch│Latest Production Release│Notes       
──────┼─────────────────────────┼────────────
7.3   │[7.3.69][9]              │Supported   
──────┼─────────────────────────┼────────────
7.2   │                         │Experimental
──────┼─────────────────────────┼────────────
7.1   │[7.1.57][10]             │Bug fixes   
──────┼─────────────────────────┼────────────
7.0   │                         │Experimental
──────┼─────────────────────────┼────────────
6.3   │[6.3.25][11]             │Unsupported 
──────┴─────────────────────────┴────────────

* `Supported` branches are those we actively maintain and will publish new patch releases.
* `Bug fixes` are branches where we still accept bug fixes, but may not publish newer patch
  releases. The community can build the latest release binaries if needed and is encouraged to
  upgrade to the `Supported` branches.
* `Experimental` branches are those used for internal feature testing. They are not recommended for
  production use.
* `Unsupported` branches are those that will no longer receive any updates.

If you are running on old production releases, we recommend always upgrading to the next major
release's latest version, and then continuing to the next major version, e.g., 6.2.X -> 6.3.25 ->
7.1.57 -> 7.3.69. These upgrade paths have been well tested in production (skipping a major release,
not marked as `Experimental`, for an upgrade is only tested in simulation).

### Binary Downloads

Developers interested in using FoundationDB can get started by downloading and installing a binary
package. Please see the [downloads page][12] for a list of available packages.

### Compiling from source

Developers on an OS for which there is no binary package, or who would like to start hacking on the
code, can get started by compiling from source.

NOTE: FoundationDB has a lot of dependencies. The Docker container listed below tracks them and is
what we use internally and is the recommended method of building FDB.

#### Build Using the Official Docker Image

The official Docker image for building is [`foundationdb/build`][13], which includes all necessary
dependencies. The Docker image definitions used by FoundationDB team members can be found in the
[dedicated repository][14].

To build FoundationDB with the clang toolchain,

mkdir /some/build_output_dir
cd /some/build_output_dir
CC=clang CXX=clang++ LD=lld cmake -D USE_LD=LLD -D USE_LIBCXX=1 -G Ninja /some/fdb/source_dir
ninja

To use GCC, a non-default version is necessary. The following modifies environment variables ($PATH,
$LD_LIBRARY_PATH, etc) to pick up the right GCC version:

source /opt/rh/gcc-toolset-13/enable
gcc --version  # should say 13
mkdir /some/build_output_dir
cd /some/build_output_dir
cmake -G Ninja /some/fdb/source_dir
ninja

Slightly more elaborate compile commands can be found in the shell aliases defined in
`/root/.bashrc` in the container image.

#### Build Locally

To build outside of the official Docker image, you'll need at least these dependencies:

1. [CMake][15] version 3.24.2 or higher
2. [Mono][16]
3. [ninja][17]

This list is likely to be incomplete. Refer to the rockylinux9 Dockerfile in the `fdb-build-support`
repo linked above for reference material on specific packages and versions that are likely to be
required.

If compiling for local development, please set `-DUSE_WERROR=ON` in CMake. Our CI compiles with
`-Werror` on, so this way you'll find out about compiler warnings that break the build earlier.

Once you have your dependencies, you can run `cmake` and then build:

1. Check out this repository.
2. Create a build directory (you can place it anywhere you like).
3. `cd <FDB_BUILD_DIR>`
4. `cmake -G Ninja <FDB_SOURCE_DIR>`
5. `ninja`

Building FoundationDB requires at least 8GB of memory. More memory is needed when building in
parallel. If the computer freezes or crashes, consider disabling parallelized build using `ninja
-j1`.

#### FreeBSD

1. Check out this repo on your server.
2. Install compile-time dependencies from ports.
3. (Optional) Use tmpfs & ccache for significantly faster repeat builds
4. (Optional) Install a [JDK][18] for Java Bindings. FoundationDB currently builds with Java 8.
5. Navigate to the directory where you checked out the FoundationDB repository.
6. Build from source.
   
   sudo pkg install -r FreeBSD \
       shells/bash devel/cmake devel/ninja devel/ccache  \
       lang/mono lang/python3 \
       devel/boost-libs devel/libeio \
       security/openssl
   mkdir .build && cd .build
   cmake -G Ninja \
       -DUSE_CCACHE=on \
       -DUSE_DTRACE=off \
       ..
   ninja -j 10
   # run fast tests
   ctest -L fast
   # run all tests
   ctest --output-on-failure -v

### macOS

The build under macOS will work the same way as on Linux. [Homebrew][19] can be used to install the
`boost` library and the `ninja` build tool.

cmake -G Ninja <FDB_SOURCE_DIR>
ninja

To generate an installable package,

<FDB_SOURCE_DIR>/packaging/osx/buildpkg.sh . <FDB_SOURCE_DIR>

### Windows

Under Windows, only Visual Studio with ClangCl is supported

1.  Install Visual Studio 2019 (IDE or Build Tools), and enable LLVM support
2.  Install [CMake 3.24.2][20] or higher
3.  Download [Boost 1.86.0][21]
4.  Unpack boost to C:\boost, or use `-DBOOST_ROOT=<PATH_TO_BOOST>` with `cmake` if unpacked
    elsewhere
5.  Install [Python][22] if it is not already installed by Visual Studio
6.  (Optional) Install [OpenJDK 11][23] to build Java bindings
7.  (Optional) Install [OpenSSL 3.x][24] to build with TLS support
8.  (Optional) Install [WIX Toolset][25] to build the Windows installer
9.  `mkdir build && cd build`
10. `cmake -G "Visual Studio 16 2019" -A x64 -T ClangCl <FDB_SOURCE_DIR>`
11. `msbuild /p:Configuration=Release foundationdb.sln`
12. To increase build performance, use `/p:UseMultiToolTask=true` and
    `/p:CL_MPCount=<NUMBER_OF_PARALLEL_JOBS>`

### Language Bindings

The language bindings that CMake supports will have a corresponding `README.md` file in the
`bindings/lang` directory corresponding to each language.

Generally, CMake will build all language bindings for which it can find all necessary dependencies.
After each successful CMake run, CMake will tell you which language bindings it is going to build.

### Generating `compile_commands.json`

CMake can build a compilation database for you. However, the default generated one is not too useful
as it operates on the generated files. When running `ninja`, the build system creates another
`compile_commands.json` file in the source directory. This can then be used for tools such as
[CCLS][26] and [CQuery][27], among others. This way, you can get code completion and code navigation
in flow. It is not yet perfect (it will show a few errors), but we are continually working to
improve the development experience.

CMake will not produce a `compile_commands.json` by default; you must pass
`-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`. This also enables the target `processed_compile_commands`,
which rewrites `compile_commands.json` to describe the actor compiler source file, not the
post-processed output files, and places the output file in the source directory. This file should
then be picked up automatically by any tooling.

Note that if the building is done inside the `foundationdb/build` Docker image, the resulting paths
will still be incorrect and require manual fixing. One will wish to re-run `cmake` with
`-DCMAKE_EXPORT_COMPILE_COMMANDS=OFF` to prevent it from reverting the manual changes.

### Using IDEs

CMake provides built-in support for several popular IDEs. However, most FoundationDB files are
written in the `flow` language, which is an extension of the C++ programming language, for coroutine
support (Note that when FoundationDB was being developed, C++20 was not available). The `flow`
language will be transpiled into C++ code using `actorcompiler`, while preventing most IDEs from
recognizing `flow`-specific syntax.

It is possible to generate project files for editing `flow` with a supported IDE. There is a CMake
option called `OPEN_FOR_IDE`, which creates a project that can be opened in an IDE for editing. This
project cannot be built, but you will be able to edit the files and utilize most of the editing and
navigation features that your IDE supports.

For example, if you want to use Xcode to make changes to FoundationDB, you can create an Xcode
project with the following command:

cmake -G Xcode -DOPEN_FOR_IDE=ON <FDB_SOURCE_DIRECTORY>

A second build directory with the `OPEN_FOR_IDE` flag off can be created for building and debugging
purposes.

[1]: /apple/foundationdb/blob/main/documentation/FDB_logo.png?raw=true
[2]: https://camo.githubusercontent.com/c9f50a0fbbf4bfb7614ff73fd363a3863ff14c64ad7aa0c4b08fe7bfe90a
1285/68747470733a2f2f636f64656275696c642e75732d776573742d322e616d617a6f6e6177732e636f6d2f62616467657
33f757569643d65794a6c626d4e79655842305a57524559585268496a6f69566a567a623152514e555a54614778474e6d396
9556e6b344f555a316430394764544d7a5a6e564f5431597a615555315255317852326f3254454e5257465a6a62335a72544
84a45636e67725a56646e4e45343062584a4a564445724f475677656e64494c336c465746593359336f78516d646a5053497
3496d6c32554746795957316c644756795533426c59794936496c4a556257686e61556c4a5658524f52554e4a546a51694c4
34a745958526c636d6c6862464e6c64464e6c636d6c68624349364d5830253344266272616e63683d6d61696e
[3]: https://www.foundationdb.org/
[4]: https://apple.github.io/foundationdb/
[5]: /apple/foundationdb/blob/main/documentation/sphinx/source
[6]: https://forums.foundationdb.org/
[7]: /apple/foundationdb/blob/main/CODE_OF_CONDUCT.md
[8]: /apple/foundationdb/blob/main/CONTRIBUTING.md
[9]: https://github.com/apple/foundationdb/releases/tag/7.3.69
[10]: https://github.com/apple/foundationdb/releases/tag/7.1.57
[11]: https://github.com/apple/foundationdb/releases/tag/6.3.25
[12]: https://github.com/apple/foundationdb/releases
[13]: https://hub.docker.com/r/foundationdb/build
[14]: https://github.com/FoundationDB/fdb-build-support
[15]: https://cmake.org/
[16]: https://www.mono-project.com/download/stable/
[17]: https://ninja-build.org/
[18]: https://www.freshports.org/java/openjdk8/
[19]: https://brew.sh/
[20]: https://cmake.org/download/
[21]: https://archives.boost.io/release/1.86.0/source/boost_1_86_0.tar.bz2
[22]: https://www.python.org/downloads/
[23]: https://developers.redhat.com/products/openjdk/download
[24]: https://slproweb.com/products/Win32OpenSSL.html
[25]: https://wixtoolset.org/
[26]: https://github.com/MaskRay/ccls
[27]: https://github.com/cquery-project/cquery
