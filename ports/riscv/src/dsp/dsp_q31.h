/*
 * Copyright 2026 Robin John
 * Copyright (c) 2010-2021 Arm Limited or its affiliates. All rights reserved.
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

#ifndef _RISCV_DSP_Q31_H

#define _RISCV_DSP_Q31_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdint.h>
#include <dsp_types.h> /* each port brings their own types.h */

    extern const q31_t twiddleCoef_rfft_q31_512[512];
    extern const q31_t twiddleCoef_rfft_q31_1024[1024];

    extern const q31_t twiddleCoef_q31_128[192];
    extern const q31_t twiddleCoef_q31_256[384];
    extern const q31_t twiddleCoef_q31_512[768];

#define RISCVBITREVINDEXTABLE_FIXED_128_TABLE_LENGTH ((uint16_t)112)
    extern const uint16_t riscvBitRevIndexTable_q31_128
        [RISCVBITREVINDEXTABLE_FIXED_128_TABLE_LENGTH];

#define RISCVBITREVINDEXTABLE_FIXED_256_TABLE_LENGTH ((uint16_t)240)
    extern const uint16_t riscvBitRevIndexTable_q31_256
        [RISCVBITREVINDEXTABLE_FIXED_256_TABLE_LENGTH];

#define RISCVBITREVINDEXTABLE_FIXED_512_TABLE_LENGTH ((uint16_t)480)
    extern const uint16_t riscvBitRevIndexTable_q31_512
        [RISCVBITREVINDEXTABLE_FIXED_512_TABLE_LENGTH];

    extern const uint32_t rearranged_twiddle_tab_stride1_arr_64_q31[3];
    extern const uint32_t rearranged_twiddle_tab_stride2_arr_64_q31[3];
    extern const uint32_t rearranged_twiddle_tab_stride3_arr_64_q31[3];
    extern const q31_t    rearranged_twiddle_stride1_64_q31[40];
    extern const q31_t    rearranged_twiddle_stride2_64_q31[40];
    extern const q31_t    rearranged_twiddle_stride3_64_q31[40];

    extern const uint32_t rearranged_twiddle_tab_stride1_arr_256_q31[4];
    extern const uint32_t rearranged_twiddle_tab_stride2_arr_256_q31[4];
    extern const uint32_t rearranged_twiddle_tab_stride3_arr_256_q31[4];
    extern const q31_t    rearranged_twiddle_stride1_256_q31[168];
    extern const q31_t    rearranged_twiddle_stride2_256_q31[168];
    extern const q31_t    rearranged_twiddle_stride3_256_q31[168];

#ifdef __cplusplus
}
#endif

#endif /*ifndef _RISCV_DSP_Q31_H*/
