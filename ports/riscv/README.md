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

Optimized implementation for the RISC-V Vector (V) Extension
```bash
-DPORT_DIR=ports/riscv/v
```

## AudioMark - RISC-V Vector Port

The RISC-V port of AudioMark also contains a sub-port optimized for the RISC-V vector extension. See this [README](v/README.md) for details.

## Code provenance and licensing

The RISC-V port includes selected kernels adapted from existing implementations in [muriscv-nn](https://github.com/tum-ei-eda/muriscv-nn/), [CMSIS-NN](https://github.com/ARM-software/CMSIS-NN) and [CMSIS-DSP](https://github.com/ARM-software/CMSIS-DSP). These implementations have been integrated into AudioMark and modified for its interfaces and RISC-V specific optimizations. The port does not incorporate these libraries as a whole.

Upstream copyright notices, applicable license identifiers, and attribution notices are retained in the corresponding source files. Refer to the individual source headers for licensing information.

File origins:

| Directory | File | Source |
|---|---|---|
| `ports/riscv/src/` | `nn_classify.c` | Adapted from `ports/arm/th_api.c` |
| | `nn_init.c` | Adapted from `ports/arm/th_api.c` |
| | All other files directly in this directory | Original implementation |
| `ports/riscv/src/nn/` | `avgpool_25x5x64_s8.c` | Original implementation |
| | `buffer_utils.h` | Original implementation |
| | `convolve_s8.c` | [muriscv-nn](https://github.com/tum-ei-eda/muriscv-nn/) |
| | `depthwise_conv_s8.c` | Original implementation |
| | `fully_connected_s8.c` | [muriscv-nn](https://github.com/tum-ei-eda/muriscv-nn/) |
| | `functions.h` | [muriscv-nn](https://github.com/tum-ei-eda/muriscv-nn/) |
| | `mat_mult_kernel_s8_s16.c` | [muriscv-nn](https://github.com/tum-ei-eda/muriscv-nn/) |
| | `math_types.h` | [muriscv-nn](https://github.com/tum-ei-eda/muriscv-nn/) |
| | `q7_to_q15_with_offset.c` | [muriscv-nn](https://github.com/tum-ei-eda/muriscv-nn/) |
| | `softmax_luts.c` | Original implementation |
| | `softmax_row12_s8.c` | Original implementation |
| | `support_functions.h` | [muriscv-nn](https://github.com/tum-ei-eda/muriscv-nn/) |
| | `types.h` | [muriscv-nn](https://github.com/tum-ei-eda/muriscv-nn/) |
| | `vec_mat_mult_t_s8.c` | [muriscv-nn](https://github.com/tum-ei-eda/muriscv-nn/) |
| `ports/riscv/v/src/` | All direct files (excluding subdirectories) | Original implementation |
| `ports/riscv/v/src/nn/` | `avgpool_25x5x64_s8.c` | Original implementation |
| | `conv0_s8.c` | [muriscv-nn](https://github.com/tum-ei-eda/muriscv-nn/) |
| | `conv1x1_s8.c` | [muriscv-nn](https://github.com/tum-ei-eda/muriscv-nn/) |
| | `depthwise_conv_s8.c` | Original implementation |
| | `mat_mult_core_1x1_s8.c` | [muriscv-nn](https://github.com/tum-ei-eda/muriscv-nn/) |
| | `mat_mult_kernel_s8_s8.c` | [muriscv-nn](https://github.com/tum-ei-eda/muriscv-nn/) |
| | `mat_mult_kernel_s8_s16.c` | Original implementation |
| | `q7_to_q15_with_offset.c` | Original implementation |
| | `rvv_support_functions.h` | [muriscv-nn](https://github.com/tum-ei-eda/muriscv-nn/) |
| `ports/riscv/` | `th_api.c` | Adapted from `ports/arm/th_api.c` |
| | `th_types.h` | Adapted from `ports/arm/th_types.h` |
