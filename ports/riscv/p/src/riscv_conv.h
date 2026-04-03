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


#ifndef P_RISCV_CONV_H
#define P_RISCV_CONV_H

#include <th_types.h>


void riscv_float_to_q31(const float *pSrc, q31_t *pDst, uint32_t blockSize);

float riscv_float_to_q31_normalize(const float *pSrc, q31_t *pDst, uint32_t blockSize);

void riscv_q31_to_float(const q31_t *pSrc, float *pDst, uint32_t blockSize);

void riscv_q31_to_float_unnormalize(const q31_t *pSrc, float *pDst, uint32_t blockSize, float scale_factor );

#endif /* P_RISCV_CONV_H */
