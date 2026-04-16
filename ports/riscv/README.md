# AudioMark - RISC-V Port

This directory contains the RISC-V port of the EEMBC AudioMark™ benchmark.

## Build

From the root of the repository:

```bash
mkdir build
cd build
cmake .. -DPORT_DIR=ports/riscv -DCMAKE_TOOLCHAIN_FILE=/path-to-toolchain-file
make audiomark
```

Bare metal builds are also supported for RISC-V. You must define the following preprocessor macros:
- Required: `CLOCK_FREQ_HZ` - your platform's clock frequency (in Hz)
- Optional: `RISCV_READ_MCYCLE` to use `mcycle/mcycleh` instead of `cycle/cycleh`

These must be passed as preprocessor defines (e.g., via `CMAKE_C_FLAGS` or in the toolchain file)

```bash
cmake .. -DPORT_DIR=ports/riscv -DCMAKE_TOOLCHAIN_FILE=/path-to-toolchain-file -DCMAKE_C_FLAGS="-DCLOCK_FREQ_HZ=50000000"
make audiomark
```

## Running Tests

To build and run all tests:

```bash
make test
```

## Optimization

Different implementations can be selected via `PORT_DIR`:

Scalar (portable C): 
```bash
-DPORT_DIR=ports/riscv
```

Optimized implementation for the RISC-V "V" Vector Extension
```bash
-DPORT_DIR=ports/riscv/v
```

Optimized implementation for the RISC-V "P" Packed SIMD Extension
```bash
-DPORT_DIR=ports/riscv/p
```

Note: Optimizations for the V and P extensions are currently work in progress. Some parts of these implementations may still fall back to scalar routines. For more details on each implementation, refer to the README in the respective port directory. 