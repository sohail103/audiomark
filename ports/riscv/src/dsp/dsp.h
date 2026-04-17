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

#ifndef _RISCV_DSP_DECL_H

#define _RISCV_DSP_DECL_H

#ifdef __cplusplus
extern "C"
{
#endif
#include <stdint.h>

    typedef enum
    {
        RISCV_MATH_SUCCESS = 0,
        RISCV_MATH_ARGUMENT_ERROR = -1,
    } riscv_status;

    typedef float float32_t;

    typedef struct
    {
        uint16_t         fftLen;   /**< length of the FFT. */
        const float32_t *pTwiddle; /**< points to the Twiddle factor table. */
        const uint16_t  *pBitRevTable; /**< points to the bit reversal table. */
        uint16_t         bitRevLength; /**< bit reversal table length. */
    } riscv_cfft_instance_f32;

    typedef struct
    {
        riscv_cfft_instance_f32 Sint;       /**< Internal CFFT structure. */
        uint16_t                fftLenRFFT; /**< length of the real sequence */
        const float32_t *pTwiddleRFFT;      /**< Twiddle factors real stage  */
    } riscv_rfft_fast_instance_f32;

    void riscv_cfft_f32(const riscv_cfft_instance_f32 *S,
                        float32_t                     *p1,
                        uint8_t                        ifftFlag,
                        uint8_t                        bitReverseFlag);

    riscv_status riscv_cfft_init_f32(riscv_cfft_instance_f32 *S,
                                     uint16_t                 fftLen);

    riscv_status riscv_rfft_fast_init_f32(riscv_rfft_fast_instance_f32 *S,
                                          uint16_t                      fftLen);

    void riscv_rfft_fast_f32(const riscv_rfft_fast_instance_f32 *S,
                             float32_t                          *p,
                             float32_t                          *pOut,

                             uint8_t ifftFlag);

    extern const riscv_cfft_instance_f32 riscv_cfft_sR_f32_len128;
    extern const riscv_cfft_instance_f32 riscv_cfft_sR_f32_len256;
    extern const riscv_cfft_instance_f32 riscv_cfft_sR_f32_len512;

    extern const float32_t twiddleCoef_rfft_512[512];
    extern const float32_t twiddleCoef_rfft_1024[1024];

    extern const float32_t twiddleCoef_128[256];
    extern const float32_t twiddleCoef_256[512];
    extern const float32_t twiddleCoef_512[1024];

#define RISCVBITREVINDEXTABLE_128_TABLE_LENGTH ((uint16_t)208)
    extern const uint16_t
        riscvBitRevIndexTable128[RISCVBITREVINDEXTABLE_128_TABLE_LENGTH];

#define RISCVBITREVINDEXTABLE_256_TABLE_LENGTH ((uint16_t)440)
    extern const uint16_t
        riscvBitRevIndexTable256[RISCVBITREVINDEXTABLE_256_TABLE_LENGTH];

#define RISCVBITREVINDEXTABLE_512_TABLE_LENGTH ((uint16_t)448)
    extern const uint16_t
        riscvBitRevIndexTable512[RISCVBITREVINDEXTABLE_512_TABLE_LENGTH];

#define FFTINIT(EXT, SIZE)                                          \
    S->bitRevLength = riscv_cfft_sR_##EXT##_len##SIZE.bitRevLength; \
    S->pBitRevTable = riscv_cfft_sR_##EXT##_len##SIZE.pBitRevTable; \
    S->pTwiddle     = riscv_cfft_sR_##EXT##_len##SIZE.pTwiddle;

#ifdef __cplusplus
}
#endif

#endif /*ifndef _RISCV_DSP_DECL_H */
