# CMake - Cross-Platform Build System Generator

CMake is an open-source, cross-platform build system generator. It generates native build files (Makefiles, Ninja, Visual Studio projects, etc.) from platform-independent configuration files.

## Basic Usage

### Project Structure

```
project/
├── CMakeLists.txt
├── src/
│   └── main.cpp
└── include/
    └── mylib.h
```

### Minimal CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.16)
project(MyProject VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(myapp src/main.cpp)
```

### Building

```bash
# Configure
cmake -B build

# Build
cmake --build build

# Build with specific config
cmake --build build --config Release

# Parallel build
cmake --build build -j 8
```

## Project Configuration

### Project Declaration

```cmake
cmake_minimum_required(VERSION 3.16)
project(MyProject
    VERSION 1.0.0
    DESCRIPTION "My awesome project"
    LANGUAGES CXX
)
```

### Targets

```cmake
# Executable
add_executable(myapp src/main.cpp src/utils.cpp)

# Static library
add_library(mylib STATIC src/lib.cpp)

# Shared library
add_library(mylib SHARED src/lib.cpp)

# Header-only library
add_library(mylib INTERFACE)
```

### Include Directories

```cmake
target_include_directories(myapp
    PUBLIC include/
    PRIVATE src/
)
```

### Link Libraries

```cmake
target_link_libraries(myapp
    PUBLIC mylib
    PRIVATE pthread
)
```

## Variables and Options

### Setting Variables

```cmake
set(MY_VAR "value")
set(MY_LIST item1 item2 item3)
set(MY_VAR "value" CACHE STRING "Description")

# Environment variable
set(ENV{MY_ENV_VAR} "value")
```

### Options

```cmake
option(BUILD_TESTS "Build tests" ON)
option(USE_FEATURE_X "Enable feature X" OFF)

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()
```

### CMake Variables

```cmake
# Paths
CMAKE_SOURCE_DIR        # Top-level source directory
CMAKE_BINARY_DIR        # Top-level build directory
CMAKE_CURRENT_SOURCE_DIR # Current CMakeLists.txt directory
PROJECT_SOURCE_DIR      # Project source directory

# Compiler
CMAKE_CXX_COMPILER      # C++ compiler path
CMAKE_C_COMPILER        # C compiler path
CMAKE_CXX_STANDARD      # C++ standard version
CMAKE_BUILD_TYPE        # Debug, Release, RelWithDebInfo, MinSizeRel
```

## Finding Packages

### find_package

```cmake
# Find package
find_package(Boost 1.70 REQUIRED COMPONENTS filesystem)
find_package(OpenSSL REQUIRED)
find_package(Threads REQUIRED)

# Use found package
target_link_libraries(myapp
    Boost::filesystem
    OpenSSL::SSL
    Threads::Threads
)
```

### pkg-config

```cmake
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBFOO REQUIRED libfoo>=1.0)

target_include_directories(myapp PRIVATE ${LIBFOO_INCLUDE_DIRS})
target_link_libraries(myapp ${LIBFOO_LIBRARIES})
```

## FetchContent

```cmake
include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1
)

FetchContent_MakeAvailable(googletest)

target_link_libraries(mytest GTest::gtest_main)
```

## Generator Expressions

```cmake
# Conditional based on config
target_compile_definitions(myapp PRIVATE
    $<$<CONFIG:Debug>:DEBUG_MODE>
)

# Conditional based on compiler
target_compile_options(myapp PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)
```

## Compiler Options

```cmake
# Compile flags
target_compile_options(myapp PRIVATE
    -Wall -Wextra -Wpedantic
)

# Compile definitions
target_compile_definitions(myapp PRIVATE
    VERSION="1.0.0"
    DEBUG_LEVEL=2
)

# Compile features
target_compile_features(myapp PRIVATE cxx_std_17)
```

## Installation

```cmake
include(GNUInstallDirs)

install(TARGETS myapp mylib
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(FILES LICENSE README.md
    DESTINATION ${CMAKE_INSTALL_DOCDIR}
)
```

## Testing

```cmake
enable_testing()

add_executable(mytest tests/test.cpp)
target_link_libraries(mytest GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(mytest)

# Or with CTest
add_test(NAME MyTest COMMAND mytest)
```

## Presets (CMake 3.19+)

Create `CMakePresets.json`:

```json
{
    "version": 3,
    "configurePresets": [
        {
            "name": "debug",
            "generator": "Ninja",
            "binaryDir": "${sourceDir}/build/debug",
            "cacheVariables": {
                "CMAKE_BUILD_TYPE": "Debug"
            }
        },
        {
            "name": "release",
            "generator": "Ninja",
            "binaryDir": "${sourceDir}/build/release",
            "cacheVariables": {
                "CMAKE_BUILD_TYPE": "Release"
            }
        }
    ],
    "buildPresets": [
        {"name": "debug", "configurePreset": "debug"},
        {"name": "release", "configurePreset": "release"}
    ]
}
```

Usage:
```bash
cmake --preset debug
cmake --build --preset debug
```

## Common Patterns

### Modern CMake Target-Based

```cmake
# Library
add_library(mylib src/lib.cpp)
target_include_directories(mylib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Executable using library
add_executable(myapp src/main.cpp)
target_link_libraries(myapp PRIVATE mylib)
```

### Sanitizers

```cmake
option(ENABLE_SANITIZERS "Enable sanitizers" OFF)

if(ENABLE_SANITIZERS)
    add_compile_options(-fsanitize=address,undefined)
    add_link_options(-fsanitize=address,undefined)
endif()
```

## Command Line Options

```bash
# Specify generator
cmake -G "Ninja" -B build
cmake -G "Unix Makefiles" -B build

# Set variable
cmake -DCMAKE_BUILD_TYPE=Release -B build
cmake -DBUILD_TESTS=ON -B build

# Specify toolchain
cmake -DCMAKE_TOOLCHAIN_FILE=toolchain.cmake -B build

# Install prefix
cmake -DCMAKE_INSTALL_PREFIX=/usr/local -B build

# Build and install
cmake --build build --target install
```

## Documentation

- Official: https://cmake.org/documentation/
- Tutorial: https://cmake.org/cmake/help/latest/guide/tutorial/index.html
- Modern CMake: https://cliutils.gitlab.io/modern-cmake/
