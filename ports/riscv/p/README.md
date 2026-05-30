# RISC-V P-Extension (Packed SIMD) Support

## Overview

This directory contains an implementation of selected DSP and NN kernels targeting the **RISC-V P-extension (Packed SIMD)**.

The current implementation is based on the **v0.21 draft specification** of the P-extension and provides a vectorized backend using packed SIMD intrinsics.

* P-extension specification: 
  [https://github.com/riscv/riscv-p-spec](https://github.com/riscv/riscv-p-spec)

---

## Toolchain & Dependencies

This implementation has been tested with the following toolchain and environment:

* LLVM Trunk:
  [https://github.com/llvm/llvm-project.git](https://github.com/llvm/llvm-project.git)

* **Clang Commit Hash** [MemCpyOpt] Keep volatile memset before memcpy (#200100). commit 42010078e75053f09e46a8d58033fa768c57961f

* Reference intrinsics definitions:
  [https://github.com/topperc/p-ext-intrinsics/blob/main/source/riscv_p_asm.h](https://github.com/topperc/p-ext-intrinsics/blob/main/source/riscv_p_asm.h)

> Note: The upstream intrinsics header support is still evolving. In this setup, the required intrinsics definitions were aligned with the reference implementation above to ensure compatibility with the v0.21 specification.

---

## Emulator / Testing Environment

Functional verification was performed using QEMU branch `dev-p-020`:

* Repository: [https://github.com/mollybuild/qemu](https://github.com/mollybuild/qemu)

---

## Intrinsic Support Notes

This implementation relies on the RISC-V P-extension Intrinsics Specification v0.21. While the reference header below was used as a starting point, several intrinsics required by this backend are not yet available in the current upstream header implementation and must be provided according to the behavior defined in the specification.

Reference intrinsics definitions:

https://github.com/topperc/p-ext-intrinsics/blob/main/source/riscv_p_asm.h

Both RV32 and RV64 builds require architecture-specific intrinsic implementations that conform to the v0.21 specification. A number of intrinsics do not map directly to a single instruction on all targets and are specified as emulation sequences in the spec sheet. The instruction mappings and expansion sequences documented in the specification should be followed to ensure portability and consistent code generation across architectures.

The following intrinsics are currently required by this implementation and should be implemented according to the v0.21 intrinsic specification:

### RV64

* `__riscv_pnclipr_i16x4`
* `__riscv_pnclipr_i8x4`
* `__riscv_pwadda_i16x4`

Some of these operations are specified as RV32-oriented packed operations and may require expansion into multiple RV64 instructions. The emulation sequences defined by the specification should be preserved.

### RV32

* `__riscv_pas_x_i32x2`
* `__riscv_psa_x_i32x2`
* `__riscv_paas_x_i32x2`
* `__riscv_pasa_x_i32x2`
* `__riscv_pmulqr_i32x2`

These intrinsics are used by the FFT and DSP kernels and are expected to behave exactly as defined by the v0.21 intrinsic specification.

Future updates to the upstream intrinsic headers may provide native definitions for some of these operations. Until then, local implementations are required to maintain compatibility with the kernels contained in this backend.
