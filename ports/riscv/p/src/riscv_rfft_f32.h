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



#ifndef P_RISCV_RFFT_H
#define P_RISCV_RFFT_H


#include "riscv_cfft_f32.h"

extern const q31_t rfftFastTwiddleQ31_1024[1024];
extern const q31_t rfftFastTwiddleQ31_128[128];

void riscv_merge_rfft_q31(
    const q31_t  *pTwiddleRFFT,
    q31_t        *p,        /* RIFFT packed input */
    q31_t        *pOut,
    uint32_t      fftLen)   /* half-length = N/2 */
;

void riscv_stage_rfft_q31(
    const q31_t  *pTwiddleRFFT,
    q31_t        *p,        /* CFFT output in-place */
    q31_t        *pOut,
    uint32_t      fftLen)   /* half-length = N/2 */
;

#endif /* P_RISCV_RFFT_H */
