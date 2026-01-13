# Clang - LLVM C/C++/Objective-C Compiler

Clang is a compiler front end for C, C++, Objective-C, and Objective-C++ built on the LLVM compiler infrastructure.

## Basic Usage

### Compilation

```bash
# Compile C source
clang source.c -o program

# Compile C++ source
clang++ source.cpp -o program

# Compile to object file
clang -c source.c -o source.o

# Compile to LLVM IR
clang -S -emit-llvm source.c -o source.ll

# Compile to assembly
clang -S source.c -o source.s
```

## Warning Flags

### Essential Warnings

```bash
-Wall           # Common warnings
-Wextra         # Extra warnings
-Werror         # Treat warnings as errors
-pedantic       # Strict ISO compliance
-Weverything    # ALL warnings (very strict)
```

### Specific Warnings

```bash
-Wshadow               # Variable shadowing
-Wconversion           # Type conversions
-Wfloat-equal          # Floating point equality
-Wundef                # Undefined macro usage
-Wcast-align           # Pointer cast alignment
-Wstrict-overflow=5    # Strict overflow checks
-Wwrite-strings        # String literal modification
-Waggregate-return     # Aggregate returns
-Wcast-qual            # Cast removes qualifiers
-Wswitch-default       # Missing default in switch
-Wswitch-enum          # Missing enum cases
-Wunreachable-code     # Unreachable code
-Wformat=2             # Format string checks
```

### Disable Specific Warnings

```bash
clang -Wno-unused-parameter source.c
clang -Wno-sign-conversion source.c
```

## Optimization

```bash
# Optimization levels
clang -O0 source.c    # No optimization (default)
clang -O1 source.c    # Basic optimization
clang -O2 source.c    # Standard optimization
clang -O3 source.c    # Aggressive optimization
clang -Os source.c    # Size optimization
clang -Oz source.c    # Aggressive size optimization
clang -Ofast source.c # Non-standard fast

# Link-time optimization
clang -flto source.c -o program
clang -flto=thin source.c -o program  # ThinLTO
```

## Sanitizers

### Address Sanitizer (ASan)

```bash
# Detect memory errors
clang -fsanitize=address source.c -o program

# With debug info
clang -fsanitize=address -g -fno-omit-frame-pointer source.c -o program
```

### Undefined Behavior Sanitizer (UBSan)

```bash
# Detect undefined behavior
clang -fsanitize=undefined source.c -o program

# Specific checks
clang -fsanitize=integer source.c
clang -fsanitize=bounds source.c
clang -fsanitize=null source.c
```

### Memory Sanitizer (MSan)

```bash
# Detect uninitialized memory reads
clang -fsanitize=memory source.c -o program
```

### Thread Sanitizer (TSan)

```bash
# Detect data races
clang -fsanitize=thread source.c -o program
```

### DataFlow Sanitizer

```bash
# Data flow tracking
clang -fsanitize=dataflow source.c -o program
```

### Multiple Sanitizers

```bash
clang -fsanitize=address,undefined source.c -o program
```

## Static Analysis

### Clang Static Analyzer

```bash
# Basic analysis
scan-build clang source.c

# Full analysis
scan-build -v -V clang source.c

# Generate HTML report
scan-build -o reports/ clang source.c
```

### Clang-Tidy

```bash
# Run clang-tidy
clang-tidy source.c -- -std=c11

# With specific checks
clang-tidy -checks='*' source.c

# Fix issues automatically
clang-tidy -fix source.c
```

## Security Hardening

```bash
# Control Flow Integrity
clang -fsanitize=cfi -flto source.c -o program

# Safe Stack
clang -fsanitize=safe-stack source.c -o program

# Stack protector
clang -fstack-protector-strong source.c -o program

# PIE
clang -fPIE -pie source.c -o program

# Fortify source
clang -D_FORTIFY_SOURCE=2 source.c -o program
```

## Standards

```bash
# C standards
clang -std=c89 source.c
clang -std=c99 source.c
clang -std=c11 source.c
clang -std=c17 source.c
clang -std=c2x source.c

# C++ standards
clang++ -std=c++11 source.cpp
clang++ -std=c++14 source.cpp
clang++ -std=c++17 source.cpp
clang++ -std=c++20 source.cpp
clang++ -std=c++23 source.cpp

# GNU extensions
clang -std=gnu17 source.c
```

## Debugging

```bash
# Generate debug info
clang -g source.c -o program
clang -g3 source.c -o program     # Max debug info

# DWARF versions
clang -gdwarf-4 source.c
clang -gdwarf-5 source.c

# Debug optimized code
clang -O2 -g -fno-omit-frame-pointer source.c -o program
```

## Cross Compilation

```bash
# Specify target
clang --target=arm-linux-gnueabihf source.c -o program
clang --target=aarch64-linux-gnu source.c -o program
clang --target=wasm32-unknown-wasi source.c -o program.wasm

# Sysroot
clang --sysroot=/path/to/sysroot source.c
```

## Code Coverage

```bash
# Generate coverage data
clang -fprofile-instr-generate -fcoverage-mapping source.c -o program

# Run program (generates default.profraw)
./program

# Process coverage data
llvm-profdata merge -sparse default.profraw -o default.profdata

# Generate report
llvm-cov show ./program -instr-profile=default.profdata
```

## Modules (C++20)

```bash
# Compile module interface
clang++ -std=c++20 -fmodules -c module.cppm -o module.pcm

# Use module
clang++ -std=c++20 -fmodules -fprebuilt-module-path=. main.cpp -o program
```

## Diagnostics

```bash
# Colored output
clang -fcolor-diagnostics source.c

# Caret diagnostics
clang -fcaret-diagnostics source.c

# Show include tree
clang -H source.c

# Print macro definitions
clang -dM -E source.c
```

## Comparison with GCC

| Feature | Clang | GCC |
|---------|-------|-----|
| Error messages | More detailed | Basic |
| Compile speed | Faster | Slower |
| Sanitizers | More options | Good |
| Static analysis | Excellent | Basic |
| C++ standard support | Leading edge | Good |

## Documentation

- Official: https://clang.llvm.org/docs/
- User Manual: https://clang.llvm.org/docs/UsersManual.html
- Clang-Tidy: https://clang.llvm.org/extra/clang-tidy/
- Static Analyzer: https://clang-analyzer.llvm.org/
