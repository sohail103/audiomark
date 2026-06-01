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

#include "math_types.h"
#include "functions.h"
#include "rvp_support_guard.h"
#include <stdint.h>

/*
 * Specialized for KWS model:
 * input: 25x5x64
 * output: 1x1x64
 */
#define INPUT_Y  25
#define INPUT_X  5
#define CHANNELS 64
#define COUNT    (INPUT_Y * INPUT_X)
#define HALF     (COUNT / 2)
/* 1 * 2^21 /125 = 16777 */
#define MAGIC_COUNT ((1 * 2097152) / COUNT)

int32_t
nn_avgpool_25x5x64_s8(const q7_t *input_data, q7_t *output_data)
{
    int16x4_t V_rec = __riscv_pmv_s_i16x4(MAGIC_COUNT);

    /* Process 4 groups (16 channels) at a time. */
    for (uint8_t c_block = 0; c_block < CHANNELS / 16; c_block++)
    {
        int16x4_t s0 = { 0 };
        int16x4_t s1 = { 0 };
        int16x4_t s2 = { 0 };
        int16x4_t s3 = { 0 };

        /* Start pointer at the current 16-channel block offset */
        const int8_t *in_ptr
            = __builtin_assume_aligned(input_data + (c_block * 16), 4);

        /* Walk vertically through the 124 spatial pixels */
        for (uint8_t spatial = HALF; spatial > 0; spatial--)
        {
            int8x4_t v00 = __riscv_pload_i8x4(in_ptr + 0);
            int8x4_t v01 = __riscv_pload_i8x4(in_ptr + CHANNELS + 0);
            s0           = __riscv_pwadda_i16x4(s0, v00, v01);

            int8x4_t v10 = __riscv_pload_i8x4(in_ptr + 4);
            int8x4_t v11 = __riscv_pload_i8x4(in_ptr + CHANNELS + 4);
            s1           = __riscv_pwadda_i16x4(s1, v10, v11);

            int8x4_t v20 = __riscv_pload_i8x4(in_ptr + 8);
            int8x4_t v21 = __riscv_pload_i8x4(in_ptr + CHANNELS + 8);
            s2           = __riscv_pwadda_i16x4(s2, v20, v21);

            int8x4_t v30 = __riscv_pload_i8x4(in_ptr + 12);
            int8x4_t v31 = __riscv_pload_i8x4(in_ptr + CHANNELS + 12);
            s3           = __riscv_pwadda_i16x4(s3, v30, v31);

            in_ptr += (CHANNELS * 2);
        }

        /* Handle the tail */
        const int8_t *tail_ptr = __builtin_assume_aligned(
            input_data + (c_block * 16) + ((COUNT - 1) * CHANNELS), 4);

        int8x4_t  vt0  = __riscv_pload_i8x4(tail_ptr + 0);
        int16x4_t vwt0 = __riscv_pwcvt_i16x4(vt0);
        s0             = __riscv_padd_i16x4(s0, vwt0);

        int8x4_t  vt1  = __riscv_pload_i8x4(tail_ptr + 4);
        int16x4_t vwt1 = __riscv_pwcvt_i16x4(vt1);
        s1             = __riscv_padd_i16x4(s1, vwt1);

        int8x4_t  vt2  = __riscv_pload_i8x4(tail_ptr + 8);
        int16x4_t vwt2 = __riscv_pwcvt_i16x4(vt2);
        s2             = __riscv_padd_i16x4(s2, vwt2);

        int8x4_t  vt3  = __riscv_pload_i8x4(tail_ptr + 12);
        int16x4_t vwt3 = __riscv_pwcvt_i16x4(vt3);
        s3             = __riscv_padd_i16x4(s3, vwt3);

        /* Requantize and store the 16 channels */
        int8_t *out_ptr
            = __builtin_assume_aligned(output_data + (c_block * 16), 4);

        vwt0 = __riscv_pmulq_i16x4(s0, V_rec);
        vt0  = __riscv_pnclipr_s_i8x4(vwt0, 6);
        __riscv_pstore_i8x4(out_ptr + 0, vt0);

        vwt1 = __riscv_pmulq_i16x4(s1, V_rec);
        vt1  = __riscv_pnclipr_s_i8x4(vwt1, 6);
        __riscv_pstore_i8x4(out_ptr + 4, vt1);

        vwt2 = __riscv_pmulq_i16x4(s2, V_rec);
        vt2  = __riscv_pnclipr_s_i8x4(vwt2, 6);
        __riscv_pstore_i8x4(out_ptr + 8, vt2);

        vwt3 = __riscv_pmulq_i16x4(s3, V_rec);
        vt3  = __riscv_pnclipr_s_i8x4(vwt3, 6);
        __riscv_pstore_i8x4(out_ptr + 12, vt3);
    }

    return 0;
}
