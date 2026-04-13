/**
 * Copyright 2026 Sohail Raj Satapathy
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * This code is written against the RISC-V P-extension v0.21 (21000) spec.
 * Current Clang toolchains only expose __riscv_p = 19000 (v0.19), and
 * overriding it to 21000 breaks compilation. So, we retain the
 * compiler-provided macro for compatibility, while using updated v0.21
 * intrinsic headers.
 */

#if defined(__riscv) && defined(__riscv_p) && (__riscv_p >= 19000)
#include <riscv_p_asm.h>
#else
#error "RISCV P EXTENSION v0.21 NOT SUPPORTED"
#endif
