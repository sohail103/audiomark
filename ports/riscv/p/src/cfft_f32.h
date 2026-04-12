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

#ifndef P_RISCV_CFFT_H
#define P_RISCV_CFFT_H

#include "conv.h"
#include "th_types.h"
#include "rvp_support_guard.h"

#define RISCVBITREVINDEXTABLE_FIXED_64_TABLE_LENGTH  ((uint16_t)56)
#define RISCVBITREVINDEXTABLE_FIXED_128_TABLE_LENGTH ((uint16_t)112)
#define RISCVBITREVINDEXTABLE_FIXED_512_TABLE_LENGTH ((uint16_t)480)

/* vecC = { vecA[low]*vecB[low] - vecA[high]*vecB[high],
           vecA[high]*vecB[low] + vecA[low]*vecB[high]} */
#define RISCV_PEXT_CMPLX_MULT_FX_AxB(_vecA, _vecB, _vecC)                   \
    do                                                                      \
    {                                                                       \
        q31x2_t _prod1 = __riscv_pmulqr_i32x2(_vecA, _vecB);                \
        q31x2_t _swapB = __riscv_ppairoe_i32x2(_vecB, _vecB);               \
        q31x2_t _prod2 = __riscv_pmulqr_i32x2(_vecA, _swapB);               \
        _vecC = __riscv_psa_x_i32x2(__riscv_ppaireo_i32x2(_prod1, _prod2),  \
                                    __riscv_ppairoe_i32x2(_prod1, _prod2)); \
    } while (0)

/* vecC = { vecA[low]*vecB[low] + vecA[high]*vecB[high],
           vecA[high]*vecB[low] - vecA[low]*vecB[high] } */
#define RISCV_PEXT_CMPLX_MULT_FX_AxConjB(_vecA, _vecB, _vecC)               \
    do                                                                      \
    {                                                                       \
        q31x2_t _prod1 = __riscv_pmulqr_i32x2(_vecA, _vecB);                \
        q31x2_t _swapB = __riscv_ppairoe_i32x2(_vecB, _vecB);               \
        q31x2_t _prod2 = __riscv_pmulqr_i32x2(_vecA, _swapB);               \
        _vecC = __riscv_pas_x_i32x2(__riscv_ppaireo_i32x2(_prod1, _prod2),  \
                                    __riscv_ppairoe_i32x2(_prod1, _prod2)); \
    } while (0)

/* vecC = { vecA[low] - vecB[high],
           vecA[high] + vecB[low] } */
#define RISCV_PEXT_CMPLX_ADD_FX_A_ixB(_vecA, _vecB, _vecC) \
    do                                                     \
    {                                                      \
        _vecC = __riscv_paas_x_i32x2(_vecA, _vecB);        \
    } while (0)

/* vecC = { vecA[low] + vecB[high],
           vecA[high] - vecB[low] } */
#define RISCV_PEXT_CMPLX_SUB_FX_A_ixB(_vecA, _vecB, _vecC) \
    do                                                     \
    {                                                      \
        _vecC = __riscv_pasa_x_i32x2(_vecA, _vecB);        \
    } while (0)

extern const uint16_t
    riscvBitRevIndexTable_fixed_64[RISCVBITREVINDEXTABLE_FIXED_64_TABLE_LENGTH];

extern const q31_t twiddleCoef_64_q31[96];

extern const uint32_t rearranged_twiddle_tab_stride1_arr_32_q31[3];

extern const uint32_t rearranged_twiddle_tab_stride2_arr_32_q31[3];

extern const uint32_t rearranged_twiddle_tab_stride3_arr_32_q31[3];

extern const q31_t rearranged_twiddle_stride1_32_q31[40];

extern const q31_t rearranged_twiddle_stride2_32_q31[40];

extern const q31_t rearranged_twiddle_stride3_32_q31[40];

extern const uint16_t riscvBitRevIndexTable_fixed_128
    [RISCVBITREVINDEXTABLE_FIXED_128_TABLE_LENGTH];

extern const q31_t twiddleCoef_128_q31[192];

extern const uint32_t rearranged_twiddle_tab_stride1_arr_64_q31[3];

extern const uint32_t rearranged_twiddle_tab_stride2_arr_64_q31[3];

extern const uint32_t rearranged_twiddle_tab_stride3_arr_64_q31[3];

extern const q31_t rearranged_twiddle_stride1_64_q31[40];

extern const q31_t rearranged_twiddle_stride2_64_q31[40];

extern const q31_t rearranged_twiddle_stride3_64_q31[40];

extern const uint16_t riscvBitRevIndexTable_fixed_512
    [RISCVBITREVINDEXTABLE_FIXED_512_TABLE_LENGTH];

extern const q31_t twiddleCoef_512_q31[768];

extern const uint32_t rearranged_twiddle_tab_stride1_arr_256_q31[4];

extern const uint32_t rearranged_twiddle_tab_stride2_arr_256_q31[4];

extern const uint32_t rearranged_twiddle_tab_stride3_arr_256_q31[4];

extern const q31_t rearranged_twiddle_stride1_256_q31[168];

extern const q31_t rearranged_twiddle_stride2_256_q31[168];

extern const q31_t rearranged_twiddle_stride3_256_q31[168];

void riscv_cfft_radix4by2_inverse_q31(const riscv_cfft_instance_q31 *p_instance,
                                      q31_t                         *pSrc,
                                      uint32_t                       fftLen);

void riscv_cfft_radix4by2_q31(const riscv_cfft_instance_q31 *p_instance,
                              q31_t                         *pSrc,
                              uint32_t                       fftLen);

void riscv_radix4_butterfly_inverse_q31(const riscv_cfft_instance_q31 *S,
                                        q31_t                         *pSrc,
                                        uint32_t                       fftLen);

void riscv_radix4_butterfly_q31(const riscv_cfft_instance_q31 *S,
                                q31_t                         *pSrc,
                                uint32_t                       fftLen);

void riscv_bitreversal_32_inpl(uint32_t       *p_Src,
                               uint16_t        bitRevLength,
                               const uint16_t *pBitRevTable);

#endif /* P_RISCV_CFFT_H */
