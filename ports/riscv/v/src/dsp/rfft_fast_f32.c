/*
 * Copyright 2026 Harshit Kumar Shivhare
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

#include "dsp/dsp.h"
#include "dsp/dsp_f32.h"
#include "../rvv_support_guard.h"

static riscv_status
riscv_rfft_512_fast_init_f32(riscv_rfft_fast_instance_f32 *S)
{
    riscv_status status;

    if (!S)
    {
        return RISCV_MATH_ERROR;
    }

    status = riscv_cfft_init_f32(&(S->Sint), 256);
    if (status != RISCV_MATH_SUCCESS)
    {
        return (status);
    }

    S->fftLenRFFT   = 512U;
    S->pTwiddleRFFT = (float32_t *)twiddleCoef_rfft_f32_512;
    return RISCV_MATH_SUCCESS;
}

static riscv_status
riscv_rfft_1024_fast_init_f32(riscv_rfft_fast_instance_f32 *S)
{
    riscv_status status;

    if (!S)
    {
        return RISCV_MATH_ERROR;
    }

    status = riscv_cfft_init_f32(&(S->Sint), 512);
    if (status != RISCV_MATH_SUCCESS)
    {
        return (status);
    }

    S->fftLenRFFT   = 1024U;
    S->pTwiddleRFFT = (float32_t *)twiddleCoef_rfft_f32_1024;
    return RISCV_MATH_SUCCESS;
}

riscv_status
riscv_rfft_fast_init_f32(riscv_rfft_fast_instance_f32 *S, uint16_t fftLen)
{
    typedef riscv_status (*fft_init_ptr)(riscv_rfft_fast_instance_f32 *);
    fft_init_ptr fptr = 0x0;

    switch (fftLen)
    {
        case 1024U:
            fptr = riscv_rfft_1024_fast_init_f32;
            break;
        case 512U:
            fptr = riscv_rfft_512_fast_init_f32;
            break;
        default:
            break;
    }

    if (!fptr)
    {
        return RISCV_MATH_ERROR;
    }
    return fptr(S);
}

static void
stage_rfft_f32(const riscv_rfft_fast_instance_f32 *S,
               float32_t                          *p,
               float32_t                          *pOut)
{
    int32_t          k      = (S->Sint).fftLen - 1;
    const float32_t *pCoeff = S->pTwiddleRFFT;
    float32_t       *pA     = p;
    float32_t       *pB     = p;

    float32_t xBR = pB[0], xBI = pB[1];
    float32_t xAR = pA[0], xAI = pA[1];
    float32_t twR = *pCoeff++;
    float32_t twI = *pCoeff++;
    float32_t t1a = xBR + xAR;
    float32_t t1b = xBI + xAI;

    *pOut++ = 0.5f * (t1a + t1b);
    *pOut++ = 0.5f * (t1a - t1b);

    pB = p + 2 * k;
    pA += 2;

    while (k > 0)
    {
        size_t       vl = __riscv_vsetvl_e32m1(k);
        vfloat32m1_t vxAR, vxAI, vtwR, vtwI;

        ptrdiff_t bstride = 2 * sizeof(float32_t);
        vxAR              = __riscv_vlse32_v_f32m1(pA, bstride, vl);
        vxAI              = __riscv_vlse32_v_f32m1(pA + 1, bstride, vl);

        vtwR = __riscv_vlse32_v_f32m1(pCoeff, bstride, vl);
        vtwI = __riscv_vlse32_v_f32m1(pCoeff + 1, bstride, vl);

        vfloat32m1_t vxBR_raw
            = __riscv_vlse32_v_f32m1(pB, -2 * (ptrdiff_t)sizeof(float32_t), vl);
        vfloat32m1_t vxBI_raw = __riscv_vlse32_v_f32m1(
            pB + 1, -2 * (ptrdiff_t)sizeof(float32_t), vl);

        vfloat32m1_t vt1a = __riscv_vfsub_vv_f32m1(vxBR_raw, vxAR, vl);
        vfloat32m1_t vt1b = __riscv_vfadd_vv_f32m1(vxBI_raw, vxAI, vl);

        vfloat32m1_t voutR = __riscv_vfadd_vv_f32m1(vxAR, vxBR_raw, vl);
        voutR              = __riscv_vfmacc_vv_f32m1(voutR, vtwR, vt1a, vl);
        voutR              = __riscv_vfmacc_vv_f32m1(voutR, vtwI, vt1b, vl);
        voutR              = __riscv_vfmul_vf_f32m1(voutR, 0.5f, vl);

        vfloat32m1_t voutI = __riscv_vfsub_vv_f32m1(vxAI, vxBI_raw, vl);
        voutI              = __riscv_vfmacc_vv_f32m1(voutI, vtwI, vt1a, vl);
        voutI              = __riscv_vfnmsac_vv_f32m1(voutI, vtwR, vt1b, vl);
        voutI              = __riscv_vfmul_vf_f32m1(voutI, 0.5f, vl);

        __riscv_vsse32_v_f32m1(pOut, bstride, voutR, vl);
        __riscv_vsse32_v_f32m1(pOut + 1, bstride, voutI, vl);

        pA += 2 * vl;
        pB -= 2 * vl;
        pCoeff += 2 * vl;
        pOut += 2 * vl;
        k -= vl;
    }
}

static void
merge_rfft_f32(const riscv_rfft_fast_instance_f32 *S,
               float32_t                          *p,
               float32_t                          *pOut)
{
    int32_t          k      = (S->Sint).fftLen - 1;
    const float32_t *pCoeff = S->pTwiddleRFFT;
    float32_t       *pA     = p;
    float32_t       *pB     = p;

    float32_t xAR = pA[0], xAI = pA[1];
    pCoeff += 2;
    *pOut++ = 0.5f * (xAR + xAI);
    *pOut++ = 0.5f * (xAR - xAI);

    pB = p + 2 * k;
    pA += 2;

    while (k > 0)
    {
        size_t       vl = __riscv_vsetvl_e32m1(k);
        vfloat32m1_t vxAR, vxAI, vtwR, vtwI;
        ptrdiff_t    bstride = 2 * sizeof(float32_t);

        vxAR = __riscv_vlse32_v_f32m1(pA, bstride, vl);
        vxAI = __riscv_vlse32_v_f32m1(pA + 1, bstride, vl);
        vtwR = __riscv_vlse32_v_f32m1(pCoeff, bstride, vl);
        vtwI = __riscv_vlse32_v_f32m1(pCoeff + 1, bstride, vl);

        vfloat32m1_t vxBR
            = __riscv_vlse32_v_f32m1(pB, -2 * (ptrdiff_t)sizeof(float32_t), vl);
        vfloat32m1_t vxBI = __riscv_vlse32_v_f32m1(
            pB + 1, -2 * (ptrdiff_t)sizeof(float32_t), vl);

        vfloat32m1_t vt1a  = __riscv_vfsub_vv_f32m1(vxAR, vxBR, vl);
        vfloat32m1_t vt1b  = __riscv_vfadd_vv_f32m1(vxAI, vxBI, vl);
        vfloat32m1_t voutR = __riscv_vfadd_vv_f32m1(vxAR, vxBR, vl);
        voutR              = __riscv_vfnmsac_vv_f32m1(voutR, vtwR, vt1a, vl);
        voutR              = __riscv_vfnmsac_vv_f32m1(voutR, vtwI, vt1b, vl);
        voutR              = __riscv_vfmul_vf_f32m1(voutR, 0.5f, vl);

        vfloat32m1_t voutI = __riscv_vfsub_vv_f32m1(vxAI, vxBI, vl);
        voutI              = __riscv_vfmacc_vv_f32m1(voutI, vtwI, vt1a, vl);
        voutI              = __riscv_vfnmsac_vv_f32m1(voutI, vtwR, vt1b, vl);
        voutI              = __riscv_vfmul_vf_f32m1(voutI, 0.5f, vl);

        __riscv_vsse32_v_f32m1(pOut, bstride, voutR, vl);
        __riscv_vsse32_v_f32m1(pOut + 1, bstride, voutI, vl);

        pA += 2 * vl;
        pB -= 2 * vl;
        pCoeff += 2 * vl;
        pOut += 2 * vl;
        k -= vl;
    }
}

void
riscv_rfft_fast_f32(const riscv_rfft_fast_instance_f32 *S,
                    float32_t                          *p,
                    float32_t                          *pOut,
                    uint8_t                             ifftFlag)
{
    const riscv_cfft_instance_f32 *Sint = &(S->Sint);

    if (ifftFlag)
    {
        merge_rfft_f32(S, p, pOut);
        riscv_cfft_f32(Sint, pOut, ifftFlag, 1);
    }
    else
    {
        riscv_cfft_f32(Sint, p, ifftFlag, 1);
        stage_rfft_f32(S, p, pOut);
    }
}
