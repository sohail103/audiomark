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

#include "ee_api.h"
#include "cfft_f32.h"
#include "th_types.h"
#include "rvp_support_guard.h"

void
th_cfft_f32(riscv_cfft_instance_q31 *p_instance,
            float                   *p_buf,
            uint8_t                  ifftFlag,
            uint8_t                  bitReverseFlagR)
{
    uint32_t fftLen = p_instance->fftLen;

    /* Alloc for max cfft size, Needs fftLen * 2 elements to hold Real + Imag!
     */
    q31_t q31_buf[1024 * 2];

    float scale = riscv_float_to_q31_normalize(p_buf, q31_buf, fftLen * 2);
    /* fftLen is always 128 */
    if (ifftFlag == 1U)
    {

        riscv_cfft_radix4by2_inverse_q31(p_instance, q31_buf, fftLen);
    }
    else
    {

        riscv_cfft_radix4by2_q31(p_instance, q31_buf, fftLen);
    }

    if (bitReverseFlagR)
    {

        riscv_bitreversal_32_inpl((uint32_t *)q31_buf,
                                  p_instance->bitRevLength,
                                  p_instance->pBitRevTable);
    }
    float algorithmic_mult = (ifftFlag == 0U) ? (float)fftLen : 1.0f;
    float final_mult
        = (1.0f / 2147483648.0f) * (1.0f / scale) * algorithmic_mult;
    riscv_q31_to_float_unnormalize(q31_buf, p_buf, fftLen * 2, final_mult);
}

void
riscv_bitreversal_32_inpl(uint32_t       *pSrc,
                          const uint16_t  bitRevLen,
                          const uint16_t *pBitRevTab)
{
    q31_t   *src      = (q31_t *)pSrc;
    uint32_t blkCnt   = bitRevLen / 4;
    uint32_t leftover = (bitRevLen % 4) / 2;

    while (blkCnt > 0U)
    {
        uint32_t idx_a = pBitRevTab[0] / sizeof(q31_t);
        uint32_t idx_b = pBitRevTab[1] / sizeof(q31_t);
        uint32_t idx_c = pBitRevTab[2] / sizeof(q31_t);
        uint32_t idx_d = pBitRevTab[3] / sizeof(q31_t);
        pBitRevTab += 4;

        /* Load */
        q31x2_t vecA = __riscv_pload_i32x2(&src[idx_a]);
        q31x2_t vecB = __riscv_pload_i32x2(&src[idx_b]);
        q31x2_t vecC = __riscv_pload_i32x2(&src[idx_c]);
        q31x2_t vecD = __riscv_pload_i32x2(&src[idx_d]);

        /* Swap */
        __riscv_pstore_i32x2(&src[idx_a], vecB);
        __riscv_pstore_i32x2(&src[idx_b], vecA);
        __riscv_pstore_i32x2(&src[idx_c], vecD);
        __riscv_pstore_i32x2(&src[idx_d], vecC);

        blkCnt--;
    }

    /* Handle leftover swap if bitRevLen not divisible by 4 */
    if (leftover)
    {
        uint32_t idx_a = pBitRevTab[0] / sizeof(q31_t);
        uint32_t idx_b = pBitRevTab[1] / sizeof(q31_t);

        q31x2_t vecA = __riscv_pload_i32x2(&src[idx_a]);
        q31x2_t vecB = __riscv_pload_i32x2(&src[idx_b]);

        __riscv_pstore_i32x2(&src[idx_a], vecB);
        __riscv_pstore_i32x2(&src[idx_b], vecA);
    }
}

void
riscv_cfft_radix4by2_inverse_q31(const riscv_cfft_instance_q31 *p_instance,
                                 q31_t                         *pSrc,
                                 uint32_t                       fftLen)
{

    uint32_t     n2;
    q31_t       *pIn0;
    q31_t       *pIn1;
    const q31_t *pCoef = p_instance->pTwiddle;
    uint32_t     blkCnt;
    q31x2_t      vecIn0, vecIn1, vecSum, vecDiff;
    q31x2_t      vecTw, vecCmplxTmp;

    n2   = fftLen >> 1;
    pIn0 = pSrc;
    pIn1 = pSrc + fftLen;

    blkCnt = n2;

    while (blkCnt > 0U)
    {
        vecIn0 = __riscv_pload_i32x2(pIn0);
        vecIn1 = __riscv_pload_i32x2(pIn1);

        vecSum = __riscv_paadd_i32x2(vecIn0, vecIn1);
        __riscv_pstore_i32x2(pIn0, vecSum);
        pIn0 += 2;

        vecTw = __riscv_pload_i32x2(pCoef);
        pCoef += 2;
        vecDiff = __riscv_pasub_i32x2(vecIn0, vecIn1);

        RISCV_PEXT_CMPLX_MULT_FX_AxB(vecDiff, vecTw, vecCmplxTmp);

        __riscv_pstore_i32x2(pIn1, vecCmplxTmp);
        pIn1 += 2;

        blkCnt--;
    }

    riscv_radix4_butterfly_inverse_q31(p_instance, pSrc, n2);

    riscv_radix4_butterfly_inverse_q31(p_instance, pSrc + fftLen, n2);

    /* No tail handling required since fftLen is always a multiple of 2 */
}

void
riscv_cfft_radix4by2_q31(const riscv_cfft_instance_q31 *p_instance,
                         q31_t                         *pSrc,
                         uint32_t                       fftLen)
{
    uint32_t     n2;
    q31_t       *pIn0;
    q31_t       *pIn1;
    const q31_t *pCoef = p_instance->pTwiddle;
    uint32_t     blkCnt;
    q31x2_t      vecIn0, vecIn1, vecSum, vecDiff;
    q31x2_t      vecTw, vecCmplxTmp;

    n2   = fftLen >> 1;
    pIn0 = pSrc;
    pIn1 = pSrc + fftLen;

    blkCnt = n2;

    while (blkCnt > 0U)
    {
        vecIn0 = __riscv_pload_i32x2(pIn0);
        vecIn1 = __riscv_pload_i32x2(pIn1);

        vecSum = __riscv_paadd_i32x2(vecIn0, vecIn1);
        __riscv_pstore_i32x2(pIn0, vecSum);
        pIn0 += 2;

        vecTw = __riscv_pload_i32x2(pCoef);
        pCoef += 2;
        vecDiff = __riscv_pasub_i32x2(vecIn0, vecIn1);

        RISCV_PEXT_CMPLX_MULT_FX_AxConjB(vecDiff, vecTw, vecCmplxTmp);

        __riscv_pstore_i32x2(pIn1, vecCmplxTmp);
        pIn1 += 2;

        blkCnt--;
    }

    riscv_radix4_butterfly_q31(p_instance, pSrc, n2);

    riscv_radix4_butterfly_q31(p_instance, pSrc + fftLen, n2);
}

void
riscv_radix4_butterfly_inverse_q31(const riscv_cfft_instance_q31 *S,
                                   q31_t                         *pSrc,
                                   uint32_t                       fftLen)
{
    q31x2_t  vecTmp0, vecTmp1;
    q31x2_t  vecSum0, vecDiff0, vecSum1, vecDiff1;
    q31x2_t  vecA, vecB, vecC, vecD;
    uint32_t blkCnt;
    uint32_t n1, n2;
    uint32_t stage = 0;
    int32_t  iter  = 1;

    /*
     * Process first stages
     * Each stage in middle stages provides two down scaling of the input
     */
    n2 = fftLen;
    n1 = n2;
    n2 >>= 2u;

    for (int k = fftLen / 4u; k > 1; k >>= 2u)
    {
        q31_t const *p_rearranged_twiddle_tab_stride2
            = &S->rearranged_twiddle_stride2
                   [S->rearranged_twiddle_tab_stride2_arr[stage]];
        q31_t const *p_rearranged_twiddle_tab_stride3
            = &S->rearranged_twiddle_stride3
                   [S->rearranged_twiddle_tab_stride3_arr[stage]];
        q31_t const *p_rearranged_twiddle_tab_stride1
            = &S->rearranged_twiddle_stride1
                   [S->rearranged_twiddle_tab_stride1_arr[stage]];

        q31_t *pBase = pSrc;
        for (int i = 0; i < iter; i++)
        {
            q31_t       *inA = pBase;
            q31_t       *inB = inA + n2 * 2;
            q31_t       *inC = inB + n2 * 2;
            q31_t       *inD = inC + n2 * 2;
            q31_t const *pW1 = p_rearranged_twiddle_tab_stride1;
            q31_t const *pW2 = p_rearranged_twiddle_tab_stride2;
            q31_t const *pW3 = p_rearranged_twiddle_tab_stride3;
            q31x2_t      vecW;

            blkCnt = n2;
            /*
             * load 2 x q31 complex pair
             */
            vecA = __riscv_pload_i32x2(inA);
            vecC = __riscv_pload_i32x2(inC);
            while (blkCnt > 0U)
            {
                vecB = __riscv_pload_i32x2(inB);
                vecD = __riscv_pload_i32x2(inD);

                vecSum0  = __riscv_paadd_i32x2(vecA, vecC);
                vecDiff0 = __riscv_pasub_i32x2(vecA, vecC);

                vecSum1  = __riscv_paadd_i32x2(vecB, vecD);
                vecDiff1 = __riscv_pasub_i32x2(vecB, vecD);
                /*
                 * [ 1 1 1 1 ] * [ A B C D ]' .* 1
                 */
                vecTmp0 = __riscv_paadd_i32x2(vecSum0, vecSum1);
                __riscv_pstore_i32x2(inA, vecTmp0);
                inA += 2;
                /*
                 * [ 1 -1 1 -1 ] * [ A B C D ]'
                 */
                vecTmp0 = __riscv_pasub_i32x2(vecSum0, vecSum1);
                /*
                 * [ 1 -1 1 -1 ] * [ A B C D ]'.* W2
                 */
                vecW = __riscv_pload_i32x2(pW2);
                pW2 += 2;
                RISCV_PEXT_CMPLX_MULT_FX_AxConjB(vecTmp0, vecW, vecTmp1);

                __riscv_pstore_i32x2(inB, vecTmp1);
                inB += 2;
                /*
                 * [ 1 -i -1 +i ] * [ A B C D ]'
                 */
                RISCV_PEXT_CMPLX_ADD_FX_A_ixB(vecDiff0, vecDiff1, vecTmp0);
                /*
                 * [ 1 -i -1 +i ] * [ A B C D ]'.* W1
                 */
                vecW = __riscv_pload_i32x2(pW1);
                pW1 += 2;
                RISCV_PEXT_CMPLX_MULT_FX_AxConjB(vecTmp0, vecW, vecTmp1);
                __riscv_pstore_i32x2(inC, vecTmp1);
                inC += 2;
                /*
                 * [ 1 +i -1 -i ] * [ A B C D ]'
                 */
                RISCV_PEXT_CMPLX_SUB_FX_A_ixB(vecDiff0, vecDiff1, vecTmp0);
                /*
                 * [ 1 +i -1 -i ] * [ A B C D ]'.* W3
                 */
                vecW = __riscv_pload_i32x2(pW3);
                pW3 += 2;
                RISCV_PEXT_CMPLX_MULT_FX_AxConjB(vecTmp0, vecW, vecTmp1);
                __riscv_pstore_i32x2(inD, vecTmp1);
                inD += 2;

                vecA = __riscv_pload_i32x2(inA);
                vecC = __riscv_pload_i32x2(inC);

                blkCnt--;
            }
            pBase += 2 * n1;
        }
        n1 = n2;
        n2 >>= 2u;
        iter = iter << 2;
        stage++;
    }

    /*
     * End of 1st stages process
     * data is in 11.21(q21) format for the 1024 point as there are 3 middle
     * stages data is in 9.23(q23) format for the 256 point as there are 2
     * middle stages data is in 7.25(q25) format for the 64 point as there are 1
     * middle stage data is in 5.27(q27) format for the 16 point as there are no
     * middle stages
     */

    /*
     * start of Last stage process
     */
    q31_t *p = pSrc;

    blkCnt = (fftLen >> 2);
    while (blkCnt > 0U)
    {
        /* Load 4 consecutive complex samples */
        vecA = __riscv_pload_i32x2(p);     /* { Re0, Im0 } */
        vecB = __riscv_pload_i32x2(p + 2); /* { Re1, Im1 } */
        vecC = __riscv_pload_i32x2(p + 4); /* { Re2, Im2 } */
        vecD = __riscv_pload_i32x2(p + 6); /* { Re3, Im3 } */

        vecSum0  = __riscv_paadd_i32x2(vecA, vecC);
        vecDiff0 = __riscv_pasub_i32x2(vecA, vecC);
        vecSum1  = __riscv_paadd_i32x2(vecB, vecD);
        vecDiff1 = __riscv_pasub_i32x2(vecB, vecD);

        /* [ 1  1  1  1 ] */
        vecTmp0 = __riscv_paadd_i32x2(vecSum0, vecSum1);
        __riscv_pstore_i32x2(p, vecTmp0);

        /* [ 1 -1  1 -1 ] */
        vecTmp0 = __riscv_pasub_i32x2(vecSum0, vecSum1);
        __riscv_pstore_i32x2(p + 2, vecTmp0);

        /* [ 1 -i -1 +i ] */
        RISCV_PEXT_CMPLX_ADD_FX_A_ixB(vecDiff0, vecDiff1, vecTmp0);
        __riscv_pstore_i32x2(p + 4, vecTmp0);

        /* [ 1 +i -1 -i ] */
        RISCV_PEXT_CMPLX_SUB_FX_A_ixB(vecDiff0, vecDiff1, vecTmp0);
        __riscv_pstore_i32x2(p + 6, vecTmp0);

        p += 8;
        blkCnt--;
    }
    /*
     * output is in 11.21(q21) format for the 1024 point
     * output is in 9.23(q23) format for the 256 point
     * output is in 7.25(q25) format for the 64 point
     * output is in 5.27(q27) format for the 16 point
     */
}

void
riscv_radix4_butterfly_q31(const riscv_cfft_instance_q31 *S,
                           q31_t                         *pSrc,
                           uint32_t                       fftLen)
{
    q31x2_t  vecTmp0, vecTmp1;
    q31x2_t  vecSum0, vecDiff0, vecSum1, vecDiff1;
    q31x2_t  vecA, vecB, vecC, vecD;
    uint32_t blkCnt;
    uint32_t n1, n2;
    uint32_t stage = 0;
    int32_t  iter  = 1;

    /*
     * Process first stages
     * Each stage in middle stages provides two down scaling of the input
     */
    n2 = fftLen;
    n1 = n2;
    n2 >>= 2u;

    for (int k = fftLen / 4u; k > 1; k >>= 2u)
    {
        q31_t const *p_rearranged_twiddle_tab_stride2
            = &S->rearranged_twiddle_stride2
                   [S->rearranged_twiddle_tab_stride2_arr[stage]];
        q31_t const *p_rearranged_twiddle_tab_stride3
            = &S->rearranged_twiddle_stride3
                   [S->rearranged_twiddle_tab_stride3_arr[stage]];
        q31_t const *p_rearranged_twiddle_tab_stride1
            = &S->rearranged_twiddle_stride1
                   [S->rearranged_twiddle_tab_stride1_arr[stage]];

        q31_t *pBase = pSrc;
        for (int i = 0; i < iter; i++)
        {
            q31_t       *inA = pBase;
            q31_t       *inB = inA + n2 * 2;
            q31_t       *inC = inB + n2 * 2;
            q31_t       *inD = inC + n2 * 2;
            q31_t const *pW1 = p_rearranged_twiddle_tab_stride1;
            q31_t const *pW2 = p_rearranged_twiddle_tab_stride2;
            q31_t const *pW3 = p_rearranged_twiddle_tab_stride3;
            q31x2_t      vecW;

            blkCnt = n2;
            /*
             * load 2 x q31 complex pair
             */
            vecA = __riscv_pload_i32x2(inA);
            vecC = __riscv_pload_i32x2(inC);
            while (blkCnt > 0U)
            {
                vecB = __riscv_pload_i32x2(inB);
                vecD = __riscv_pload_i32x2(inD);

                vecSum0  = __riscv_paadd_i32x2(vecA, vecC);
                vecDiff0 = __riscv_pasub_i32x2(vecA, vecC);

                vecSum1  = __riscv_paadd_i32x2(vecB, vecD);
                vecDiff1 = __riscv_pasub_i32x2(vecB, vecD);
                /*
                 * [ 1 1 1 1 ] * [ A B C D ]' .* 1
                 */
                vecTmp0 = __riscv_paadd_i32x2(vecSum0, vecSum1);
                __riscv_pstore_i32x2(inA, vecTmp0);
                inA += 2;
                /*
                 * [ 1 -1 1 -1 ] * [ A B C D ]'
                 */
                vecTmp0 = __riscv_pasub_i32x2(vecSum0, vecSum1);
                /*
                 * [ 1 -1 1 -1 ] * [ A B C D ]'.* W2
                 */
                vecW = __riscv_pload_i32x2(pW2);
                pW2 += 2;
                RISCV_PEXT_CMPLX_MULT_FX_AxB(vecTmp0, vecW, vecTmp1);

                __riscv_pstore_i32x2(inB, vecTmp1);
                inB += 2;
                /*
                 * [ 1 -i -1 +i ] * [ A B C D ]'
                 */
                RISCV_PEXT_CMPLX_SUB_FX_A_ixB(vecDiff0, vecDiff1, vecTmp0);
                /*
                 * [ 1 -i -1 +i ] * [ A B C D ]'.* W1
                 */
                vecW = __riscv_pload_i32x2(pW1);
                pW1 += 2;
                RISCV_PEXT_CMPLX_MULT_FX_AxB(vecTmp0, vecW, vecTmp1);
                __riscv_pstore_i32x2(inC, vecTmp1);
                inC += 2;
                /*
                 * [ 1 +i -1 -i ] * [ A B C D ]'
                 */
                RISCV_PEXT_CMPLX_ADD_FX_A_ixB(vecDiff0, vecDiff1, vecTmp0);
                /*
                 * [ 1 +i -1 -i ] * [ A B C D ]'.* W3
                 */
                vecW = __riscv_pload_i32x2(pW3);
                pW3 += 2;
                RISCV_PEXT_CMPLX_MULT_FX_AxB(vecTmp0, vecW, vecTmp1);
                __riscv_pstore_i32x2(inD, vecTmp1);
                inD += 2;

                vecA = __riscv_pload_i32x2(inA);
                vecC = __riscv_pload_i32x2(inC);

                blkCnt--;
            }
            pBase += 2 * n1;
        }
        n1 = n2;
        n2 >>= 2u;
        iter = iter << 2;
        stage++;
    }

    /*
     * End of 1st stages process
     * data is in 11.21(q21) format for the 1024 point as there are 3 middle
     * stages data is in 9.23(q23) format for the 256 point as there are 2
     * middle stages data is in 7.25(q25) format for the 64 point as there are 1
     * middle stage data is in 5.27(q27) format for the 16 point as there are no
     * middle stages
     */

    /*
     * start of Last stage process
     */
    q31_t *p = pSrc;

    blkCnt = (fftLen >> 2);
    while (blkCnt > 0U)
    {
        /* Load 4 consecutive complex samples */
        vecA = __riscv_pload_i32x2(p);     /* { Re0, Im0 } */
        vecB = __riscv_pload_i32x2(p + 2); /* { Re1, Im1 } */
        vecC = __riscv_pload_i32x2(p + 4); /* { Re2, Im2 } */
        vecD = __riscv_pload_i32x2(p + 6); /* { Re3, Im3 } */

        vecSum0  = __riscv_paadd_i32x2(vecA, vecC);
        vecDiff0 = __riscv_pasub_i32x2(vecA, vecC);
        vecSum1  = __riscv_paadd_i32x2(vecB, vecD);
        vecDiff1 = __riscv_pasub_i32x2(vecB, vecD);

        /* [ 1  1  1  1 ] */
        vecTmp0 = __riscv_paadd_i32x2(vecSum0, vecSum1);
        __riscv_pstore_i32x2(p, vecTmp0);

        /* [ 1 -1  1 -1 ] */
        vecTmp0 = __riscv_pasub_i32x2(vecSum0, vecSum1);
        __riscv_pstore_i32x2(p + 2, vecTmp0);

        /* [ 1 -i -1 +i ] */
        RISCV_PEXT_CMPLX_SUB_FX_A_ixB(vecDiff0, vecDiff1, vecTmp0);
        __riscv_pstore_i32x2(p + 4, vecTmp0);

        /* [ 1 +i -1 -i ] */
        RISCV_PEXT_CMPLX_ADD_FX_A_ixB(vecDiff0, vecDiff1, vecTmp0);
        __riscv_pstore_i32x2(p + 6, vecTmp0);

        p += 8;
        blkCnt--;
    }
    /*
     * output is in 11.21(q21) format for the 1024 point
     * output is in 9.23(q23) format for the 256 point
     * output is in 7.25(q25) format for the 64 point
     * output is in 5.27(q27) format for the 16 point
     */
}
