# WORKER HINT: Kani Library Location

The Kani library (libkani.rlib) is installed in one of these locations:

```
~/.kani/kani-0.66.0-fast/lib/libkani.rlib  (preferred)
~/.kani/kani-0.66.0/lib/libkani.rlib       (fallback)
```

The test driver auto-detects the library, preferring `kani-0.66.0-fast` if available.

To manually compile with Kani transforms:

```bash
rustc test.rs -L ~/.kani/kani-0.66.0-fast/lib --extern kani=~/.kani/kani-0.66.0-fast/lib/libkani.rlib
```

This lets code use `kani::any()`, `kani::assume()`, etc. Set `KANI_LIB_DIR` to override detection.
