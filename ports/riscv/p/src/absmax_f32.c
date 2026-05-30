/**
 * Copyright 2026 Robin John
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

#include "ee_api.h"
#include "th_types.h"
#include "rvp_support_guard.h"
#include <stdint.h>

void th_absmax_f32(const ee_f32_t *p_in, uint32_t len,
                   ee_f32_t *p_max, uint32_t *p_index)
{
    const uint32x2_t V_smask = __riscv_pmv_s_u32x2(0x7FFFFFFF);
    const uint32_t  *p32     = (const uint32_t *)p_in;
    uint32_t        n       = len >> 1;

    /* Find the absolute max */
    uint32x2_t V_runmax = __riscv_pmv_s_u32x2(0);
    for (uint32_t i = 0; i < n; i++)
    {
        uint32x2_t V_abs = __riscv_pand_u32x2(
                              __riscv_pload_u32x2(p32 + 2*i), V_smask);
        V_runmax = __riscv_pmaxu_u32x2(V_runmax, V_abs);
    }

    /* Horizontal reduce: pick larger of 2 lanes */
    uint32_t v0 = (uint32_t)__riscv_pget_u32x2_u32(V_runmax, 0);
    uint32_t v1 = (uint32_t)__riscv_pget_u32x2_u32(V_runmax, 1);
    uint32_t max_bits = v0 > v1 ? v0 : v1;

    /* Odd tail */
    if (len & 1u) {
        uint32_t t = *(const uint32_t *)(p_in + len - 1) & 0x7FFFFFFFu;
        if (t > max_bits) max_bits = t;
    }

    /* Index search */
    uint32x2_t V_target = __riscv_pmv_s_u32x2(max_bits);
    uint32_t  max_idx  = len;

    for (uint32_t i = 0; i < n; i++)
    {
        uint32x2_t V_abs = __riscv_pand_u32x2(
                              __riscv_pload_u32x2(p32 + 2*i), V_smask);
        uint32x2_t V_eq  = __riscv_pmseq_u32x2(V_abs, V_target);
        if (__riscv_preinterpret_u32x2_u64(V_eq)) {
            /* lower 32 bits nonzero → lane 0 matched first */
            max_idx = 2*i + (__riscv_pget_u32x2_u32(V_eq, 0) ? 0u : 1u);
            break;
        }
    }

    /* Odd tail */
    if ((max_idx == len) && (len & 1u)) {
        uint32_t t = *(const uint32_t *)(p_in + len - 1) & 0x7FFFFFFFu;
        if (t == max_bits) max_idx = len - 1;
    }

    *p_max   = *(const ee_f32_t *)&max_bits;
    *p_index = max_idx;
}
