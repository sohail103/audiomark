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