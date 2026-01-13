# Make - Build Automation Tool

GNU Make is a build automation tool that automatically builds executable programs and libraries from source code by reading Makefiles that specify how to derive the target program.

## Basic Usage

### Running Make

```bash
# Build default target
make

# Build specific target
make target_name

# Build multiple targets
make target1 target2

# Specify makefile
make -f Makefile.custom

# Dry run (show commands without executing)
make -n

# Debug mode
make -d

# Parallel jobs
make -j4
make -j$(nproc)

# Keep going on errors
make -k

# Silent mode
make -s
```

## Makefile Basics

### Simple Makefile

```makefile
# Variables
CC = gcc
CFLAGS = -Wall -Wextra -O2

# Default target
all: program

# Compile object file
main.o: main.c header.h
	$(CC) $(CFLAGS) -c main.c -o main.o

# Link executable
program: main.o utils.o
	$(CC) main.o utils.o -o program

# Clean target
clean:
	rm -f *.o program

# Phony targets (not files)
.PHONY: all clean
```

### Variables

```makefile
# Simple assignment (expanded when used)
CC = gcc
CFLAGS = -Wall

# Immediate assignment (expanded when defined)
FILES := $(wildcard *.c)

# Conditional assignment (only if not set)
CC ?= gcc

# Append
CFLAGS += -O2

# Automatic variables
# $@ - target name
# $< - first prerequisite
# $^ - all prerequisites
# $* - stem (% match in pattern rules)
# $? - prerequisites newer than target

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@
```

### Pattern Rules

```makefile
# Pattern rule for object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Pattern rule for executables
%: %.o
	$(CC) $< -o $@
```

### Functions

```makefile
# String functions
SOURCES = main.c utils.c
OBJECTS = $(SOURCES:.c=.o)  # main.o utils.o
OBJECTS = $(patsubst %.c,%.o,$(SOURCES))  # Same as above

# File functions
SOURCES = $(wildcard src/*.c)
DIRS = $(dir $(SOURCES))        # Directory part
NAMES = $(notdir $(SOURCES))    # File name only
BASE = $(basename $(SOURCES))   # Without extension

# Shell function
DATE = $(shell date +%Y%m%d)
FILES = $(shell find . -name "*.c")

# Conditional
DEBUG ?= 0
ifeq ($(DEBUG),1)
    CFLAGS += -g -DDEBUG
else
    CFLAGS += -O2 -DNDEBUG
endif
```

## Common Patterns

### C/C++ Project

```makefile
CC = gcc
CXX = g++
CFLAGS = -Wall -Wextra -std=c11
CXXFLAGS = -Wall -Wextra -std=c++17
LDFLAGS = -lm

SRCDIR = src
OBJDIR = obj
BINDIR = bin

SOURCES = $(wildcard $(SRCDIR)/*.c)
OBJECTS = $(SOURCES:$(SRCDIR)/%.c=$(OBJDIR)/%.o)
TARGET = $(BINDIR)/program

.PHONY: all clean dirs

all: dirs $(TARGET)

dirs:
	@mkdir -p $(OBJDIR) $(BINDIR)

$(TARGET): $(OBJECTS)
	$(CC) $^ $(LDFLAGS) -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJDIR) $(BINDIR)

# Dependencies
-include $(OBJECTS:.o=.d)

$(OBJDIR)/%.d: $(SRCDIR)/%.c
	@$(CC) $(CFLAGS) -MM -MT '$(OBJDIR)/$*.o $@' $< > $@
```

### Recursive Make

```makefile
SUBDIRS = lib app tests

.PHONY: all clean $(SUBDIRS)

all: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@

clean:
	for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir clean; \
	done
```

### Non-Recursive (Preferred)

```makefile
# Define targets in each directory
include lib/Makefile.mk
include app/Makefile.mk
include tests/Makefile.mk

all: $(ALL_TARGETS)
```

## Special Targets

```makefile
.PHONY: all clean test     # Not file targets
.PRECIOUS: %.o             # Don't delete intermediate files
.SECONDARY:                # Keep all intermediate files
.DELETE_ON_ERROR:          # Delete target on error
.SILENT:                   # Don't print commands
.SUFFIXES: .c .o           # Define suffixes
.DEFAULT:                  # Default rule for unmatched targets
.NOTPARALLEL:              # Disable parallel execution
```

## Debugging

```bash
# Print variable value
make print-VARIABLE

# With this in Makefile:
print-%:
	@echo $* = $($*)

# Show implicit rules
make -p

# Show why target is being rebuilt
make --debug=b

# Full debug
make -d
```

## Automatic Dependencies

```makefile
DEPFLAGS = -MMD -MP

%.o: %.c
	$(CC) $(CFLAGS) $(DEPFLAGS) -c $< -o $@

-include $(OBJECTS:.o=.d)
```

## Help Target

```makefile
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all     - Build everything"
	@echo "  clean   - Remove build artifacts"
	@echo "  test    - Run tests"
	@echo "  install - Install program"
```

## Cross-Platform

```makefile
ifeq ($(OS),Windows_NT)
    RM = del /Q
    EXE = .exe
else
    RM = rm -f
    EXE =
endif

clean:
	$(RM) *$(OBJ) program$(EXE)
```

## Best Practices

1. **Use .PHONY** for non-file targets
2. **Generate dependencies** automatically
3. **Use pattern rules** instead of explicit rules
4. **Use variables** for compiler and flags
5. **Quote shell variables** properly
6. **Use @** prefix for silent commands
7. **Use -** prefix to ignore errors
8. **Organize with subdirectories** (non-recursive preferred)

## Documentation

- Official: https://www.gnu.org/software/make/manual/
- Quick Reference: https://www.gnu.org/software/make/manual/html_node/Quick-Reference.html
