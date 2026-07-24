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
#include "dsp_q31.h"
#include "rvp_support_guard.h"
#include "ee_api.h"

riscv_status
riscv_rfft_fast_init_q31(riscv_rfft_fast_instance_q31 *__EE_RESTRICT S, uint16_t fftLenReal)
{
    riscv_status status = RISCV_MATH_SUCCESS;

    /*  Initialize the Real FFT length */
    S->fftLenRFFT = (uint16_t)fftLenReal;

    /*  Initialization of coef modifier depending on the FFT length */
    switch (fftLenReal)
    {
        case 1024U:
            S->pTwiddleRFFT = (q31_t *)twiddleCoef_rfft_q31_1024;
            status          = riscv_cfft_init_q31(&(S->Sint), 512);
            break;
        case 512U:
            S->pTwiddleRFFT = (q31_t *)twiddleCoef_rfft_q31_512;
            status          = riscv_cfft_init_q31(&(S->Sint), 256);
            break;
        default:
            status = RISCV_MATH_ERROR;
            break;
    }

    return status;
}

void
riscv_merge_rfft_q31(const q31_t *__EE_RESTRICT pTwiddleRFFT,
                     q31_t       *__EE_RESTRICT p, /* RIFFT packed input */
                     q31_t       *__EE_RESTRICT pOut,
                     uint32_t                   fftLen) /* half-length = N/2 */
{
    /* k=0 */
    q31x2_t cmplxA = __riscv_pload_i32x2(p);
    q31x2_t crossA = __riscv_paas_x_i32x2(cmplxA, cmplxA);
    q31x2_t swapA  = __riscv_ppairoe_i32x2(crossA, crossA);
    __riscv_pstore_i32x2(pOut, swapA);

    q31_t       *pA    = p + 2;
    q31_t       *pB    = p + 2 * (fftLen - 1);
    const q31_t *pCoef = pTwiddleRFFT + 2;
    q31_t       *pO    = pOut + 2;

    uint32_t k = fftLen - 1;
    while (k > 0)
    {

        /* conjugate twiddle for inverse */
        q31x2_t cmplxA = __riscv_pload_i32x2(pA);
        q31x2_t cmplxB = __riscv_pload_i32x2(pB);
        q31x2_t swapxB = __riscv_ppairoe_i32x2(cmplxB, cmplxB);

        /* twidx = (twR, twI) */
        q31x2_t twidx = __riscv_pload_i32x2(pCoef);

        /* t1x = (xAR - xBR), (xAI + xBI) */
        q31x2_t t1x = __riscv_psas_x_i32x2(cmplxA, swapxB);

        /* rs = (twx.R * t1a), (twx.I * t1b) */
        q31x2_t rs = __riscv_pmulqr_i32x2(twidx, t1x);

        /* tu = (twx.I * t1a), (twx.R * t1b) */
        q31x2_t ctwidx = __riscv_ppairoe_i32x2(twidx, twidx);
        q31x2_t tu     = __riscv_pmulqr_i32x2(ctwidx, t1x);

        /* px.R = (xAR + xBR - r - s )/2 */
        /* px.I = (xAI - xBI - (u - t) )/2 */
        q31x2_t xAB = __riscv_pasa_x_i32x2(cmplxA, swapxB);
        q31x2_t tr  = __riscv_ppaire_i32x2(tu, rs);
        q31x2_t su  = __riscv_ppairo_i32x2(rs, tu);
        q31x2_t xVW = __riscv_pasa_x_i32x2(su, tr);
        q31x2_t px  = __riscv_pssub_i32x2(xAB, xVW);
        __riscv_pstore_i32x2(pO, px);
        pA += 2;
        pB -= 2;
        pCoef += 2;
        pO += 2;
        k--;
    }
}

void
riscv_stage_rfft_q31(const q31_t *__EE_RESTRICT pTwiddleRFFT,
                     q31_t       *__EE_RESTRICT p, /* CFFT output in-place */
                     q31_t       *__EE_RESTRICT pOut,
                     uint32_t                   fftLen) /* half-length = N/2 */
{
    /* k=0 */
    q31x2_t cmplxA = __riscv_pload_i32x2(p);
    q31x2_t crossA = __riscv_psas_x_i32x2(cmplxA, cmplxA);
    q31x2_t swapA  = __riscv_ppairoe_i32x2(crossA, crossA);
    __riscv_pstore_i32x2(pOut, swapA);

    q31_t       *pA    = p + 2;
    q31_t       *pB    = p + 2 * (fftLen - 1);
    const q31_t *pCoef = pTwiddleRFFT + 2; /* skip k=0 */
    q31_t       *pO    = pOut + 2;

    uint32_t k = fftLen - 1;
    while (k > 0)
    {

        /* conjugate twiddle for inverse */
        q31x2_t cmplxA = __riscv_pload_i32x2(pA);
        q31x2_t cmplxB = __riscv_pload_i32x2(pB);
        q31x2_t swapxA = __riscv_ppairoe_i32x2(cmplxA, cmplxA);

        /* twidx = (twR, twI) */
        q31x2_t twidx = __riscv_pload_i32x2(pCoef);

        /* t1a = xBR - xAR,  t1b = xBI + xAI */
        q31x2_t t1x = __riscv_psas_x_i32x2(cmplxB, swapxA);

        /* p01 = (twidx.R * t1a), (twidx.I * t1b) */
        q31x2_t p03 = __riscv_pmulqr_i32x2(twidx, t1x);

        /* p23 = (twidx.I * t1a), (twidx.R * t1b) */
        q31x2_t ctwidx = __riscv_ppairoe_i32x2(twidx, twidx);
        q31x2_t p12    = __riscv_pmulqr_i32x2(ctwidx, t1x);

        /* px.R = (xAR + xBR + p0 + p3)/2 */
        /* px.I = (xAI - xBI + p1 - p2)/2 */
        q31x2_t xswapAB = __riscv_paas_x_i32x2(swapxA, cmplxB);
        q31x2_t xAB     = __riscv_ppairoe_i32x2(xswapAB, xswapAB);
        q31x2_t p01     = __riscv_ppaire_i32x2(p03, p12);
        q31x2_t p23     = __riscv_ppairo_i32x2(p12, p03);
        q31x2_t xVW     = __riscv_pasa_x_i32x2(p01, p23);
        q31x2_t px      = __riscv_psadd_i32x2(xAB, xVW);
        __riscv_pstore_i32x2(pO, px);
        pA += 2;
        pB -= 2;
        pCoef += 2;
        pO += 2;
        k--;
    }
}

void
riscv_rfft_fast_q31(const riscv_rfft_fast_instance_q31 *__EE_RESTRICT p_instance,
                    q31_t                              *__EE_RESTRICT q_in,
                    q31_t                              *__EE_RESTRICT q_out,
                    uint8_t                                           ifftFlag)
{
    const riscv_cfft_instance_q31 *S_CFFT = &(p_instance->Sint);
    uint32_t L2 = p_instance->fftLenRFFT >> 1U; /* half-length for CFFT */

    /* Calculation of Real FFT */
    if (ifftFlag == 1U)
    {
        /*  Real FFT compression */
        riscv_merge_rfft_q31(p_instance->pTwiddleRFFT, q_in, q_out, L2);

        /* Complex radix-4 IFFT process */
        riscv_cfft_q31(S_CFFT, q_out, ifftFlag, 1);
    }
    else
    {
        /* Calculation of RFFT of input */
        riscv_cfft_q31(S_CFFT, q_in, ifftFlag, 1);

        /*  Real FFT extraction */
        riscv_stage_rfft_q31(p_instance->pTwiddleRFFT, q_in, q_out, L2);
    }
}
