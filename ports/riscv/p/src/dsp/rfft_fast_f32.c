/*
 * Copyright 2026 Robin John
 * Copyright (C) 2010-2021 ARM Limited or its affiliates. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <dsp_types.h>
#include "dsp.h"
#include "convert.h"
#include "rvp_support_guard.h"

void
riscv_rfft_fast_f32(const riscv_rfft_fast_instance_q31 *p_instance,
            float                        *p_in,
            float                        *p_out,
            uint8_t                       ifftFlag)
{
    uint32_t L2 = p_instance->fftLenRFFT >> 1U; /* half-length for CFFT */

    static q31_t q31_in[1026];
    static q31_t q31_out[1026];
    float        dynamic_scale, final_mult;

    if (ifftFlag == 1U)
    {
        dynamic_scale = riscv_float_to_q31_normalize(
            p_in, q31_in, p_instance->fftLenRFFT + 2);

        riscv_rfft_fast_q31(p_instance, q31_in, q31_out, ifftFlag);
        /* inverse: CFFT scales by 1/L2, compensate with *2 for the merge
         * halving */
        final_mult = (1.0f / 2147483648.0f) * (1.0f / dynamic_scale);

        riscv_q31_to_float_unnormalize(
            q31_out, p_out, p_instance->fftLenRFFT, final_mult);
    }
    else
    {
        dynamic_scale = riscv_float_to_q31_normalize(
            p_in, q31_in, p_instance->fftLenRFFT);

        riscv_rfft_fast_q31(p_instance, q31_in, q31_out, ifftFlag);
        /* forward CFFT scales by 1/L2, stage halves once more, total = 1/(2*L2)
         */
        final_mult
            = (1.0f / 2147483648.0f) * (1.0f / dynamic_scale) * (float)L2;

        riscv_q31_to_float_unnormalize(
            q31_out, p_out, p_instance->fftLenRFFT + 2, final_mult);
    }
}
