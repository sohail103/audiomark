# RISC-V P-Extension (Packed SIMD) Support

## Overview

This directory contains an implementation of selected DSP kernels targeting the **RISC-V P-extension (Packed SIMD)**.

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

## Supported Architectures

* ✅ **RV64**: Supported and tested
* ⚠️ **RV32**: Not yet supported

RV32 support is expected as the P-extension intrinsics and toolchain support continue to mature.
