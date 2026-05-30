/**
 * Copyright 2026 Robin John
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

#include "functions.h"
#include "rvp_support_guard.h"
#include <riscv_p_asm.h>

extern const int32_t EXP_LUT[256];
extern const int32_t RECIP_LUT[256];

void
nn_softmax_row12_s8(const int8_t *restrict input, int8_t *restrict output)
{
    const int8_t *in_ptr =
    __builtin_assume_aligned(input, 4);

    int8_t *out_ptr =
        __builtin_assume_aligned(output, 4);
    /* Find the max */
    int8x4_t V_in1   = __riscv_pload_i8x4(in_ptr);
    int8x4_t V_in2   = __riscv_pload_i8x4(in_ptr + 4);
    int8x4_t V_in3   = __riscv_pload_i8x4(in_ptr + 8);
    int8x4_t V_mbuf  = __riscv_pmax_i8x4(V_in1, V_in2);
    int8x4_t V_mfin  = __riscv_pmax_i8x4(V_mbuf, V_in3);
    int8_t   b0      = __riscv_pget_i8x4_i8(V_mfin, 0);
    int8_t   b1      = __riscv_pget_i8x4_i8(V_mfin, 1);
    int8_t   b2      = __riscv_pget_i8x4_i8(V_mfin, 2);
    int8_t   b3      = __riscv_pget_i8x4_i8(V_mfin, 3);
    int8_t   max0    = (b0 > b1 ? b0 : b1);
    int8_t   max1    = (b2 > b3 ? b2 : b3);
    int8_t   max_val = max0 > max1 ? max0 : max1;

    int32_t  exp_res[12];
    uint64_t u_sum = 0;

    /* Exp lookup and accumulate */
    for (int i = 0; i < 12; i++)
    {
        exp_res[i] = EXP_LUT[(uint8_t)(max_val - in_ptr[i])];
        u_sum += (uint64_t)exp_res[i];
    }

    int      c      = __builtin_clzll(u_sum);
    uint64_t norm64 = u_sum << c;

    uint32_t recip_index = (uint32_t)(norm64 >> 55) & 0xFF;
    uint32_t fraction    = (uint32_t)(norm64 >> 48) & 0x7F;

    int32_t y0               = RECIP_LUT[recip_index];
    int32_t y1               = RECIP_LUT[recip_index + (recip_index < 255)];
    int32_t normalized_recip = y0 + ((int32_t)(fraction * (y1 - y0)) >> 7);

    int       final_shift = 86 - c;
    int32x2_t vRecip      = __riscv_pmv_s_i32x2(normalized_recip);

    /* Requantize */
    for (int i = 0; i < 12; i += 4)
    {

        int32x2_t vp0 = __riscv_pload_i32x2(exp_res + i);
        int32x2_t vp1 = __riscv_pload_i32x2(exp_res + i + 2);

        vp0 = __riscv_pmulhr_i32x2(vp0, vRecip);
        vp1 = __riscv_pmulhr_i32x2(vp1, vRecip);

        int16x2_t nVp0 = __riscv_pnclipr_s_i16x2(vp0, final_shift - 32);
        int16x2_t nVp1 = __riscv_pnclipr_s_i16x2(vp1, final_shift - 32);
        int16x4_t vp   = __riscv_pmv_s_i16x4(0);

        vp = __riscv_pset_i16x2_i16x4(vp, nVp0, 0);
        vp = __riscv_pset_i16x2_i16x4(vp, nVp1, 1);

        vp = __riscv_psub_i16x4(vp, __riscv_pmv_s_i16x4(128));

        int8x4_t nvp = __riscv_pnclipr_s_i8x4(vp, 0);

        __riscv_pstore_i8x4(out_ptr + i, nvp);
    }
}
