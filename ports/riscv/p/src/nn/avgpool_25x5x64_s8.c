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
int32_t
nn_avgpool_25x5x64_s8(const q7_t *input_data, q7_t *output_data)
{
    /* 1/125 = 0.008.  0.008 * 2^21 = 16777 */
    int16x4_t V_rec = __riscv_pmv_s_i16x4(16777);

    /* Process 4 groups (16 channels) at a time. */
    for (uint8_t c_block = 0; c_block < 4; c_block++)
    {
        int16x4_t s0 = __riscv_pmv_s_i16x4(0);
        int16x4_t s1 = __riscv_pmv_s_i16x4(0);
        int16x4_t s2 = __riscv_pmv_s_i16x4(0);
        int16x4_t s3 = __riscv_pmv_s_i16x4(0);

        /* Start pointer at the current 16-channel block offset */
        const int8_t *in_ptr = input_data + (c_block * 16);

        /* Walk vertically through the 125 spatial pixels */
        for (uint8_t spatial = 0; spatial < 125; spatial++)
        {
            s0 = __riscv_padd_i16x4(
                s0, __riscv_pwcvt_i16x4(__riscv_pload_i8x4(in_ptr + 0)));

            s1 = __riscv_padd_i16x4(
                s1, __riscv_pwcvt_i16x4(__riscv_pload_i8x4(in_ptr + 4)));

            s2 = __riscv_padd_i16x4(
                s2, __riscv_pwcvt_i16x4(__riscv_pload_i8x4(in_ptr + 8)));

            s3 = __riscv_padd_i16x4(
                s3, __riscv_pwcvt_i16x4(__riscv_pload_i8x4(in_ptr + 12)));

            in_ptr += 64;
        }

        /* Requantize and store the 16 channels */
        int8_t *out_ptr = output_data + (c_block * 16);

        __riscv_pstore_i8x4(
            out_ptr + 0,
            __riscv_pnclipr_s_i8x4(__riscv_pmulq_i16x4(s0, V_rec), 6));

        __riscv_pstore_i8x4(
            out_ptr + 4,
            __riscv_pnclipr_s_i8x4(__riscv_pmulq_i16x4(s1, V_rec), 6));

        __riscv_pstore_i8x4(
            out_ptr + 8,
            __riscv_pnclipr_s_i8x4(__riscv_pmulq_i16x4(s2, V_rec), 6));

        __riscv_pstore_i8x4(
            out_ptr + 12,
            __riscv_pnclipr_s_i8x4(__riscv_pmulq_i16x4(s3, V_rec), 6));
    }

    return 0;
}
