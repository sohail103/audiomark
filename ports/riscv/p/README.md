# RISC-V P-Extension (Packed SIMD) Support

## Overview

This directory contains an implementation of selected DSP kernels targeting the **RISC-V P-extension (Packed SIMD)**.

The current implementation is based on the **v0.19 draft specification** of the P-extension and provides a vectorized backend using packed SIMD intrinsics.

* P-extension specification: [https://github.com/riscv/riscv-p-spec](https://github.com/riscv/riscv-p-spec)

---

## Toolchain & Dependencies

This implementation has been tested with the following toolchain and environment:


* LLVM fork with P-extension support:
  [https://github.com/sihuan/llvm-project/tree/p-int-headeronly](https://github.com/sihuan/llvm-project/tree/p-int-headeronly)

* **Clang Commit Hash** [Clang][RISCV] Mask shift amounts in P extension intrinsics to avoid UB. commit d905aeb42106d64d1c4302be1dbd7da2dd3d57a7

* Reference intrinsics definitions:
  [https://github.com/topperc/p-ext-intrinsics/blob/main/source/riscv_p_asm.h](https://github.com/topperc/p-ext-intrinsics/blob/main/source/riscv_p_asm.h)

> Note: The upstream intrinsics header support is still evolving. In this setup, the required intrinsics definitions were aligned with the reference implementation above to ensure compatibility with the v0.19 specification.

---

## Emulator / Testing Environment

Functional verification was performed using QEMU:

* Base repository: [https://github.com/JRobinNTA/qemu](https://github.com/JRobinNTA/qemu)
* Adapted from: [https://github.com/mollybuild/qemu](https://github.com/mollybuild/qemu)
* Updated to support **P-extension v0.19**

---

## Supported Architectures

* ✅ **RV64**: Supported and tested
* ⚠️ **RV32**: Not yet supported

RV32 support is expected as the P-extension intrinsics and toolchain support continue to mature.
