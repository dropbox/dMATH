# GCC - GNU Compiler Collection

GCC (GNU Compiler Collection) is a compiler system supporting various programming languages including C, C++, Objective-C, Fortran, Ada, Go, and D.

## Basic Usage

### Compilation

```bash
# Compile C source
gcc source.c -o program

# Compile C++ source
g++ source.cpp -o program

# Compile to object file
gcc -c source.c -o source.o

# Link object files
gcc file1.o file2.o -o program
```

### Common Options

```bash
# With warnings
gcc -Wall -Wextra -Werror source.c -o program

# With optimization
gcc -O2 source.c -o program        # Standard optimization
gcc -O3 source.c -o program        # Aggressive optimization
gcc -Os source.c -o program        # Optimize for size
gcc -Ofast source.c -o program     # Fast, may break standards

# Debug symbols
gcc -g source.c -o program         # GDB-compatible debug info
gcc -g3 source.c -o program        # Maximum debug info

# Preprocessor
gcc -E source.c                    # Output preprocessed source
gcc -D MACRO=value source.c        # Define macro
gcc -I /path/to/headers source.c   # Include path
```

## Warning Flags

### Essential Warnings

```bash
-Wall          # All common warnings
-Wextra        # Extra warnings
-Werror        # Treat warnings as errors
-pedantic      # ISO C compliance warnings
```

### Security Warnings

```bash
-Wformat=2               # Format string vulnerabilities
-Wformat-security        # Security-related format issues
-Wstack-protector        # Stack protection warnings
-Wstrict-overflow        # Strict overflow checking
```

### Additional Warnings

```bash
-Wshadow                 # Variable shadowing
-Wconversion             # Type conversion issues
-Wsign-conversion        # Signed/unsigned conversion
-Wcast-align             # Pointer alignment issues
-Wwrite-strings          # String literal modification
-Wlogical-op             # Suspicious logical operations
-Wdouble-promotion       # Float to double promotion
-Wformat-truncation      # Format string truncation
```

## Optimization Levels

| Level | Description |
|-------|-------------|
| `-O0` | No optimization (default, fastest compile) |
| `-O1` | Basic optimization |
| `-O2` | Standard optimization (recommended) |
| `-O3` | Aggressive optimization |
| `-Os` | Optimize for size |
| `-Ofast` | Fast, non-standard compliant |
| `-Og` | Optimize for debugging |

## Security Hardening

```bash
# Stack protection
gcc -fstack-protector-strong source.c -o program

# Position Independent Executable
gcc -fPIE -pie source.c -o program

# Full RELRO
gcc -Wl,-z,relro,-z,now source.c -o program

# ASLR support
gcc -fPIC source.c -shared -o library.so

# All hardening
gcc -D_FORTIFY_SOURCE=2 -fstack-protector-strong \
    -fPIE -pie -Wl,-z,relro,-z,now source.c -o program
```

## Sanitizers

```bash
# Address Sanitizer (memory errors)
gcc -fsanitize=address source.c -o program

# Undefined Behavior Sanitizer
gcc -fsanitize=undefined source.c -o program

# Thread Sanitizer (data races)
gcc -fsanitize=thread source.c -o program

# Leak Sanitizer
gcc -fsanitize=leak source.c -o program

# Multiple sanitizers
gcc -fsanitize=address,undefined source.c -o program
```

## Static Analysis

```bash
# Static analyzer (GCC 10+)
gcc -fanalyzer source.c -o program

# Generate dependency info
gcc -M source.c           # Full dependencies
gcc -MM source.c          # Without system headers
```

## Link-Time Optimization

```bash
# Enable LTO
gcc -flto source.c -o program

# With parallel LTO
gcc -flto=auto source.c -o program
```

## Cross Compilation

```bash
# Target ARM
arm-linux-gnueabihf-gcc source.c -o program

# Specify target
gcc -target arm-linux-gnueabihf source.c -o program
```

## Debugging

```bash
# Generate debug info
gcc -g source.c -o program

# Debug levels
gcc -g1 source.c          # Minimal debug info
gcc -g2 source.c          # Default debug info
gcc -g3 source.c          # Maximum debug info

# DWARF version
gcc -gdwarf-4 source.c    # DWARF 4 format
```

## Standards Compliance

```bash
# C Standards
gcc -std=c89 source.c     # ANSI C / C89
gcc -std=c99 source.c     # C99
gcc -std=c11 source.c     # C11
gcc -std=c17 source.c     # C17 (default in GCC 8+)
gcc -std=c2x source.c     # C23 (draft)

# C++ Standards
g++ -std=c++11 source.cpp # C++11
g++ -std=c++14 source.cpp # C++14
g++ -std=c++17 source.cpp # C++17
g++ -std=c++20 source.cpp # C++20
g++ -std=c++23 source.cpp # C++23

# GNU extensions
gcc -std=gnu17 source.c   # C17 with GNU extensions
```

## Library Linking

```bash
# Link with library
gcc source.c -lm -o program          # Math library
gcc source.c -lpthread -o program    # POSIX threads

# Library search path
gcc -L/path/to/libs source.c -lmylib

# Static linking
gcc -static source.c -o program

# Shared library
gcc -shared -fPIC source.c -o libmylib.so
```

## Machine Options

```bash
# Architecture
gcc -march=native source.c           # Native CPU
gcc -march=x86-64-v3 source.c        # x86-64 level 3

# CPU tuning
gcc -mtune=skylake source.c

# Instruction sets
gcc -mavx2 source.c                  # Enable AVX2
gcc -msse4.2 source.c                # Enable SSE4.2
```

## Verbose Output

```bash
# Show commands
gcc -v source.c -o program

# Show search paths
gcc -print-search-dirs

# Show predefined macros
gcc -dM -E - < /dev/null
```

## Documentation

- Official: https://gcc.gnu.org/onlinedocs/
- Manual: https://gcc.gnu.org/onlinedocs/gcc/
- Wiki: https://gcc.gnu.org/wiki
