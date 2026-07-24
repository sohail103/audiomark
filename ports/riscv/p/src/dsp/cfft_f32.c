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
#include "ee_api.h"

void
riscv_cfft_f32(const riscv_cfft_instance_q31 *__EE_RESTRICT p_instance,
               float                         *__EE_RESTRICT p_buf,
               uint8_t                                      ifftFlag,
               uint8_t                                      bitReverseFlagR)
{
    uint32_t fftLen = p_instance->fftLen;

    /* Alloc for max cfft size, Needs fftLen * 2 elements to hold Real + Imag!
     */
    static q31_t q31_buf[1024 * 2];

    float scale = riscv_float_to_q31_normalize(p_buf, q31_buf, fftLen * 2);

    riscv_cfft_q31(p_instance, q31_buf, ifftFlag, bitReverseFlagR);

    float algorithmic_mult = (ifftFlag == 0U) ? (float)fftLen : 1.0f;
    float final_mult
        = (1.0f / 2147483648.0f) * (1.0f / scale) * algorithmic_mult;
    riscv_q31_to_float_unnormalize(q31_buf, p_buf, fftLen * 2, final_mult);
}
