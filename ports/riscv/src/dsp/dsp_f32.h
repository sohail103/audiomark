/*
 * Copyright 2026 Harshit Kumar Shivhare
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

#ifndef _RISCV_DSP_F32_H

#define _RISCV_DSP_F32_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdint.h>
#include <dsp_types.h> /* each port brings their own types.h */

    extern const float32_t twiddleCoef_rfft_f32_512[512];
    extern const float32_t twiddleCoef_rfft_f32_1024[1024];

    extern const float32_t twiddleCoef_f32_128[256];
    extern const float32_t twiddleCoef_f32_256[512];
    extern const float32_t twiddleCoef_f32_512[1024];

#define RISCVBITREVINDEXTABLE_FLOAT_128_TABLE_LENGTH ((uint16_t)208)
    extern const uint16_t riscvBitRevIndexTable_f32_128
        [RISCVBITREVINDEXTABLE_FLOAT_128_TABLE_LENGTH];

#define RISCVBITREVINDEXTABLE_FLOAT_256_TABLE_LENGTH ((uint16_t)440)
    extern const uint16_t riscvBitRevIndexTable_f32_256
        [RISCVBITREVINDEXTABLE_FLOAT_256_TABLE_LENGTH];

#define RISCVBITREVINDEXTABLE_FLOAT_512_TABLE_LENGTH ((uint16_t)448)
    extern const uint16_t riscvBitRevIndexTable_f32_512
        [RISCVBITREVINDEXTABLE_FLOAT_512_TABLE_LENGTH];

#ifdef __cplusplus
}
#endif

#endif /*ifndef _RISCV_DSP_F32_H*/
