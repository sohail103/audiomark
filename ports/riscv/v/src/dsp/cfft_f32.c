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

#define FFTINIT(EXT, SIZE)                                          \
    S->bitRevLength = riscv_cfft_sR_##EXT##_len##SIZE.bitRevLength; \
    S->pBitRevTable = riscv_cfft_sR_##EXT##_len##SIZE.pBitRevTable; \
    S->pTwiddle     = riscv_cfft_sR_##EXT##_len##SIZE.pTwiddle;

#include <stdlib.h>

riscv_status
riscv_cfft_init_f32(riscv_cfft_instance_f32 *S, uint16_t fftLen)
{
    /*  Initialise the default arm status */
    riscv_status status = RISCV_MATH_SUCCESS;

    /*  Initialise the FFT length */
    S->fftLen = fftLen;

    /*  Initialise the Twiddle coefficient pointer */
    S->pTwiddle = NULL;

    /*  Initializations of Instance structure depending on the FFT length */
    switch (S->fftLen)
    {
        case 512U:
            S->bitRevLength = RISCVBITREVINDEXTABLE_FLOAT_512_TABLE_LENGTH;
            S->pBitRevTable = riscvBitRevIndexTable_f32_512;
            S->pTwiddle     = twiddleCoef_f32_512;
            break;

        case 256U:
            S->bitRevLength = RISCVBITREVINDEXTABLE_FLOAT_256_TABLE_LENGTH;
            S->pBitRevTable = riscvBitRevIndexTable_f32_256;
            S->pTwiddle     = twiddleCoef_f32_256;
            break;

        case 128U:
            S->bitRevLength = RISCVBITREVINDEXTABLE_FLOAT_128_TABLE_LENGTH;
            S->pBitRevTable = riscvBitRevIndexTable_f32_128;
            S->pTwiddle     = twiddleCoef_f32_128;
            break;

        default:
            /*  Reporting error if fftSize is not valid value */
            status = RISCV_MATH_ERROR;
            break;
    }

    return (status);
}

static void
riscv_radix8_butterfly_f32(float32_t       *pSrc,
                           uint16_t         fftLen,
                           const float32_t *pCoef,
                           uint16_t         twidCoefModifier)
{
    uint32_t        ia1, ia2, ia3, ia4, ia5, ia6, ia7;
    uint32_t        i1, i2, i3, i4, i5, i6, i7, i8;
    uint32_t        id;
    uint32_t        n1, n2, j;
    float32_t       co2, co3, co4, co5, co6, co7, co8;
    float32_t       si2, si3, si4, si5, si6, si7, si8;
    const float32_t C81 = 0.70710678118f;

    n2 = fftLen;

    do
    {
        n1 = n2;
        n2 = n2 >> 3;
        i1 = 0;

        while (i1 < fftLen)
        {
            uint32_t   rem    = (fftLen - i1 + n1 - 1) / n1;
            size_t     vl     = __riscv_vsetvl_e32m1(rem);
            ssize_t    stride = (ssize_t)n1 * 2 * sizeof(float32_t);
            float32_t *b      = pSrc + 2 * i1;

            vfloat32m1_t vr1 = __riscv_vlse32_v_f32m1(b, stride, vl);
            vfloat32m1_t vs1 = __riscv_vlse32_v_f32m1(b + 1, stride, vl);
            vfloat32m1_t vr2 = __riscv_vlse32_v_f32m1(b + 2 * n2, stride, vl);
            vfloat32m1_t vs2
                = __riscv_vlse32_v_f32m1(b + 2 * n2 + 1, stride, vl);
            vfloat32m1_t vr3 = __riscv_vlse32_v_f32m1(b + 4 * n2, stride, vl);
            vfloat32m1_t vs3
                = __riscv_vlse32_v_f32m1(b + 4 * n2 + 1, stride, vl);
            vfloat32m1_t vr4 = __riscv_vlse32_v_f32m1(b + 6 * n2, stride, vl);
            vfloat32m1_t vs4
                = __riscv_vlse32_v_f32m1(b + 6 * n2 + 1, stride, vl);
            vfloat32m1_t vr5 = __riscv_vlse32_v_f32m1(b + 8 * n2, stride, vl);
            vfloat32m1_t vs5
                = __riscv_vlse32_v_f32m1(b + 8 * n2 + 1, stride, vl);
            vfloat32m1_t vr6 = __riscv_vlse32_v_f32m1(b + 10 * n2, stride, vl);
            vfloat32m1_t vs6
                = __riscv_vlse32_v_f32m1(b + 10 * n2 + 1, stride, vl);
            vfloat32m1_t vr7 = __riscv_vlse32_v_f32m1(b + 12 * n2, stride, vl);
            vfloat32m1_t vs7
                = __riscv_vlse32_v_f32m1(b + 12 * n2 + 1, stride, vl);
            vfloat32m1_t vr8 = __riscv_vlse32_v_f32m1(b + 14 * n2, stride, vl);
            vfloat32m1_t vs8
                = __riscv_vlse32_v_f32m1(b + 14 * n2 + 1, stride, vl);

            vfloat32m1_t vR1  = __riscv_vfadd_vv_f32m1(vr1, vr5, vl);
            vfloat32m1_t vR5  = __riscv_vfsub_vv_f32m1(vr1, vr5, vl);
            vfloat32m1_t vR2  = __riscv_vfadd_vv_f32m1(vr2, vr6, vl);
            vfloat32m1_t vR6  = __riscv_vfsub_vv_f32m1(vr2, vr6, vl);
            vfloat32m1_t vR3  = __riscv_vfadd_vv_f32m1(vr3, vr7, vl);
            vfloat32m1_t vR7  = __riscv_vfsub_vv_f32m1(vr3, vr7, vl);
            vfloat32m1_t vR4  = __riscv_vfadd_vv_f32m1(vr4, vr8, vl);
            vfloat32m1_t vR8  = __riscv_vfsub_vv_f32m1(vr4, vr8, vl);
            vfloat32m1_t vt1  = __riscv_vfsub_vv_f32m1(vR1, vR3, vl);
            vR1               = __riscv_vfadd_vv_f32m1(vR1, vR3, vl);
            vfloat32m1_t vR3s = __riscv_vfsub_vv_f32m1(vR2, vR4, vl);
            vR2               = __riscv_vfadd_vv_f32m1(vR2, vR4, vl);

            vfloat32m1_t vS1  = __riscv_vfadd_vv_f32m1(vs1, vs5, vl);
            vfloat32m1_t vS5  = __riscv_vfsub_vv_f32m1(vs1, vs5, vl);
            vfloat32m1_t vS2  = __riscv_vfadd_vv_f32m1(vs2, vs6, vl);
            vfloat32m1_t vS6  = __riscv_vfsub_vv_f32m1(vs2, vs6, vl);
            vfloat32m1_t vS3  = __riscv_vfadd_vv_f32m1(vs3, vs7, vl);
            vfloat32m1_t vS7  = __riscv_vfsub_vv_f32m1(vs3, vs7, vl);
            vfloat32m1_t vS4  = __riscv_vfadd_vv_f32m1(vs4, vs8, vl);
            vfloat32m1_t vS8  = __riscv_vfsub_vv_f32m1(vs4, vs8, vl);
            vfloat32m1_t vt2  = __riscv_vfsub_vv_f32m1(vS1, vS3, vl);
            vS1               = __riscv_vfadd_vv_f32m1(vS1, vS3, vl);
            vfloat32m1_t vS3s = __riscv_vfsub_vv_f32m1(vS2, vS4, vl);
            vS2               = __riscv_vfadd_vv_f32m1(vS2, vS4, vl);

            vfloat32m1_t vi1r = __riscv_vfadd_vv_f32m1(vR1, vR2, vl);
            vfloat32m1_t vi5r = __riscv_vfsub_vv_f32m1(vR1, vR2, vl);
            vfloat32m1_t vi1i = __riscv_vfadd_vv_f32m1(vS1, vS2, vl);
            vfloat32m1_t vi5i = __riscv_vfsub_vv_f32m1(vS1, vS2, vl);
            vfloat32m1_t vi3r = __riscv_vfadd_vv_f32m1(vt1, vS3s, vl);
            vfloat32m1_t vi7r = __riscv_vfsub_vv_f32m1(vt1, vS3s, vl);
            vfloat32m1_t vi3i = __riscv_vfsub_vv_f32m1(vt2, vR3s, vl);
            vfloat32m1_t vi7i = __riscv_vfadd_vv_f32m1(vt2, vR3s, vl);

            vfloat32m1_t vA = __riscv_vfsub_vv_f32m1(vR6, vR8, vl);
            vR6             = __riscv_vfadd_vv_f32m1(vR6, vR8, vl);
            vfloat32m1_t vB = __riscv_vfsub_vv_f32m1(vS6, vS8, vl);
            vS6             = __riscv_vfadd_vv_f32m1(vS6, vS8, vl);
            vA              = __riscv_vfmul_vf_f32m1(vA, C81, vl);
            vR6             = __riscv_vfmul_vf_f32m1(vR6, C81, vl);
            vB              = __riscv_vfmul_vf_f32m1(vB, C81, vl);
            vS6             = __riscv_vfmul_vf_f32m1(vS6, C81, vl);

            vt1 = __riscv_vfsub_vv_f32m1(vR5, vA, vl);
            vR5 = __riscv_vfadd_vv_f32m1(vR5, vA, vl);
            vR8 = __riscv_vfsub_vv_f32m1(vR7, vR6, vl);
            vR7 = __riscv_vfadd_vv_f32m1(vR7, vR6, vl);
            vt2 = __riscv_vfsub_vv_f32m1(vS5, vB, vl);
            vS5 = __riscv_vfadd_vv_f32m1(vS5, vB, vl);
            vS8 = __riscv_vfsub_vv_f32m1(vS7, vS6, vl);
            vS7 = __riscv_vfadd_vv_f32m1(vS7, vS6, vl);

            vfloat32m1_t vi2r = __riscv_vfadd_vv_f32m1(vR5, vS7, vl);
            vfloat32m1_t vi8r = __riscv_vfsub_vv_f32m1(vR5, vS7, vl);
            vfloat32m1_t vi6r = __riscv_vfadd_vv_f32m1(vt1, vS8, vl);
            vfloat32m1_t vi4r = __riscv_vfsub_vv_f32m1(vt1, vS8, vl);
            vfloat32m1_t vi2i = __riscv_vfsub_vv_f32m1(vS5, vR7, vl);
            vfloat32m1_t vi8i = __riscv_vfadd_vv_f32m1(vS5, vR7, vl);
            vfloat32m1_t vi6i = __riscv_vfsub_vv_f32m1(vt2, vR8, vl);
            vfloat32m1_t vi4i = __riscv_vfadd_vv_f32m1(vt2, vR8, vl);

            __riscv_vsse32_v_f32m1(b, stride, vi1r, vl);
            __riscv_vsse32_v_f32m1(b + 1, stride, vi1i, vl);
            __riscv_vsse32_v_f32m1(b + 2 * n2, stride, vi2r, vl);
            __riscv_vsse32_v_f32m1(b + 2 * n2 + 1, stride, vi2i, vl);
            __riscv_vsse32_v_f32m1(b + 4 * n2, stride, vi3r, vl);
            __riscv_vsse32_v_f32m1(b + 4 * n2 + 1, stride, vi3i, vl);
            __riscv_vsse32_v_f32m1(b + 6 * n2, stride, vi4r, vl);
            __riscv_vsse32_v_f32m1(b + 6 * n2 + 1, stride, vi4i, vl);
            __riscv_vsse32_v_f32m1(b + 8 * n2, stride, vi5r, vl);
            __riscv_vsse32_v_f32m1(b + 8 * n2 + 1, stride, vi5i, vl);
            __riscv_vsse32_v_f32m1(b + 10 * n2, stride, vi6r, vl);
            __riscv_vsse32_v_f32m1(b + 10 * n2 + 1, stride, vi6i, vl);
            __riscv_vsse32_v_f32m1(b + 12 * n2, stride, vi7r, vl);
            __riscv_vsse32_v_f32m1(b + 12 * n2 + 1, stride, vi7i, vl);
            __riscv_vsse32_v_f32m1(b + 14 * n2, stride, vi8r, vl);
            __riscv_vsse32_v_f32m1(b + 14 * n2 + 1, stride, vi8i, vl);

            i1 += (uint32_t)vl * n1;
        }

        if (n2 < 8)
        {
            break;
        }

        ia1 = 0;
        j   = 1;
        do
        {
            id  = ia1 + twidCoefModifier;
            ia1 = id;
            ia2 = ia1 + id;
            ia3 = ia2 + id;
            ia4 = ia3 + id;
            ia5 = ia4 + id;
            ia6 = ia5 + id;
            ia7 = ia6 + id;

            co2 = pCoef[2 * ia1];
            si2 = pCoef[2 * ia1 + 1];
            co3 = pCoef[2 * ia2];
            si3 = pCoef[2 * ia2 + 1];
            co4 = pCoef[2 * ia3];
            si4 = pCoef[2 * ia3 + 1];
            co5 = pCoef[2 * ia4];
            si5 = pCoef[2 * ia4 + 1];
            co6 = pCoef[2 * ia5];
            si6 = pCoef[2 * ia5 + 1];
            co7 = pCoef[2 * ia6];
            si7 = pCoef[2 * ia6 + 1];
            co8 = pCoef[2 * ia7];
            si8 = pCoef[2 * ia7 + 1];

            i1 = j;
            while (i1 < fftLen)
            {
                uint32_t   rem    = (fftLen - i1 + n1 - 1) / n1;
                size_t     vl     = __riscv_vsetvl_e32m1(rem);
                ssize_t    stride = (ssize_t)n1 * 2 * sizeof(float32_t);
                float32_t *b      = pSrc + 2 * i1;

                vfloat32m1_t vr1 = __riscv_vlse32_v_f32m1(b, stride, vl);
                vfloat32m1_t vs1 = __riscv_vlse32_v_f32m1(b + 1, stride, vl);
                vfloat32m1_t vr2
                    = __riscv_vlse32_v_f32m1(b + 2 * n2, stride, vl);
                vfloat32m1_t vs2
                    = __riscv_vlse32_v_f32m1(b + 2 * n2 + 1, stride, vl);
                vfloat32m1_t vr3
                    = __riscv_vlse32_v_f32m1(b + 4 * n2, stride, vl);
                vfloat32m1_t vs3
                    = __riscv_vlse32_v_f32m1(b + 4 * n2 + 1, stride, vl);
                vfloat32m1_t vr4
                    = __riscv_vlse32_v_f32m1(b + 6 * n2, stride, vl);
                vfloat32m1_t vs4
                    = __riscv_vlse32_v_f32m1(b + 6 * n2 + 1, stride, vl);
                vfloat32m1_t vr5
                    = __riscv_vlse32_v_f32m1(b + 8 * n2, stride, vl);
                vfloat32m1_t vs5
                    = __riscv_vlse32_v_f32m1(b + 8 * n2 + 1, stride, vl);
                vfloat32m1_t vr6
                    = __riscv_vlse32_v_f32m1(b + 10 * n2, stride, vl);
                vfloat32m1_t vs6
                    = __riscv_vlse32_v_f32m1(b + 10 * n2 + 1, stride, vl);
                vfloat32m1_t vr7
                    = __riscv_vlse32_v_f32m1(b + 12 * n2, stride, vl);
                vfloat32m1_t vs7
                    = __riscv_vlse32_v_f32m1(b + 12 * n2 + 1, stride, vl);
                vfloat32m1_t vr8
                    = __riscv_vlse32_v_f32m1(b + 14 * n2, stride, vl);
                vfloat32m1_t vs8
                    = __riscv_vlse32_v_f32m1(b + 14 * n2 + 1, stride, vl);

                vfloat32m1_t vR1  = __riscv_vfadd_vv_f32m1(vr1, vr5, vl);
                vfloat32m1_t vR5  = __riscv_vfsub_vv_f32m1(vr1, vr5, vl);
                vfloat32m1_t vR2  = __riscv_vfadd_vv_f32m1(vr2, vr6, vl);
                vfloat32m1_t vR6  = __riscv_vfsub_vv_f32m1(vr2, vr6, vl);
                vfloat32m1_t vR3  = __riscv_vfadd_vv_f32m1(vr3, vr7, vl);
                vfloat32m1_t vR7  = __riscv_vfsub_vv_f32m1(vr3, vr7, vl);
                vfloat32m1_t vR4  = __riscv_vfadd_vv_f32m1(vr4, vr8, vl);
                vfloat32m1_t vR8  = __riscv_vfsub_vv_f32m1(vr4, vr8, vl);
                vfloat32m1_t vt1  = __riscv_vfsub_vv_f32m1(vR1, vR3, vl);
                vR1               = __riscv_vfadd_vv_f32m1(vR1, vR3, vl);
                vfloat32m1_t vR3s = __riscv_vfsub_vv_f32m1(vR2, vR4, vl);
                vR2               = __riscv_vfadd_vv_f32m1(vR2, vR4, vl);
                vfloat32m1_t vi1r = __riscv_vfadd_vv_f32m1(vR1, vR2, vl);
                vfloat32m1_t vr2x = __riscv_vfsub_vv_f32m1(vR1, vR2, vl);

                vfloat32m1_t vS1  = __riscv_vfadd_vv_f32m1(vs1, vs5, vl);
                vfloat32m1_t vS5  = __riscv_vfsub_vv_f32m1(vs1, vs5, vl);
                vfloat32m1_t vS2  = __riscv_vfadd_vv_f32m1(vs2, vs6, vl);
                vfloat32m1_t vS6  = __riscv_vfsub_vv_f32m1(vs2, vs6, vl);
                vfloat32m1_t vS3  = __riscv_vfadd_vv_f32m1(vs3, vs7, vl);
                vfloat32m1_t vS7  = __riscv_vfsub_vv_f32m1(vs3, vs7, vl);
                vfloat32m1_t vS4  = __riscv_vfadd_vv_f32m1(vs4, vs8, vl);
                vfloat32m1_t vS8  = __riscv_vfsub_vv_f32m1(vs4, vs8, vl);
                vfloat32m1_t vt2  = __riscv_vfsub_vv_f32m1(vS1, vS3, vl);
                vS1               = __riscv_vfadd_vv_f32m1(vS1, vS3, vl);
                vfloat32m1_t vS3s = __riscv_vfsub_vv_f32m1(vS2, vS4, vl);
                vS2               = __riscv_vfadd_vv_f32m1(vS2, vS4, vl);
                vfloat32m1_t vi1i = __riscv_vfadd_vv_f32m1(vS1, vS2, vl);
                vfloat32m1_t vs2x = __riscv_vfsub_vv_f32m1(vS1, vS2, vl);

                vfloat32m1_t vR1t = __riscv_vfadd_vv_f32m1(vt1, vS3s, vl);
                vt1               = __riscv_vfsub_vv_f32m1(vt1, vS3s, vl);
                vfloat32m1_t vt2t = __riscv_vfsub_vv_f32m1(vt2, vR3s, vl);
                vt2               = __riscv_vfadd_vv_f32m1(vt2, vR3s, vl);

                vfloat32m1_t vi5r = __riscv_vfmacc_vf_f32m1(
                    __riscv_vfmul_vf_f32m1(vr2x, co5, vl), si5, vs2x, vl);
                vfloat32m1_t vi5i = __riscv_vfnmsac_vf_f32m1(
                    __riscv_vfmul_vf_f32m1(vs2x, co5, vl), si5, vr2x, vl);

                vfloat32m1_t vi3r = __riscv_vfmacc_vf_f32m1(
                    __riscv_vfmul_vf_f32m1(vR1t, co3, vl), si3, vt2t, vl);
                vfloat32m1_t vi3i = __riscv_vfnmsac_vf_f32m1(
                    __riscv_vfmul_vf_f32m1(vt2t, co3, vl), si3, vR1t, vl);

                vfloat32m1_t vi7r = __riscv_vfmacc_vf_f32m1(
                    __riscv_vfmul_vf_f32m1(vt1, co7, vl), si7, vt2, vl);
                vfloat32m1_t vi7i = __riscv_vfnmsac_vf_f32m1(
                    __riscv_vfmul_vf_f32m1(vt2, co7, vl), si7, vt1, vl);

                vfloat32m1_t vA = __riscv_vfsub_vv_f32m1(vR6, vR8, vl);
                vR6             = __riscv_vfadd_vv_f32m1(vR6, vR8, vl);
                vfloat32m1_t vB = __riscv_vfsub_vv_f32m1(vS6, vS8, vl);
                vS6             = __riscv_vfadd_vv_f32m1(vS6, vS8, vl);
                vA              = __riscv_vfmul_vf_f32m1(vA, C81, vl);
                vR6             = __riscv_vfmul_vf_f32m1(vR6, C81, vl);
                vB              = __riscv_vfmul_vf_f32m1(vB, C81, vl);
                vS6             = __riscv_vfmul_vf_f32m1(vS6, C81, vl);

                vt1 = __riscv_vfsub_vv_f32m1(vR5, vA, vl);
                vR5 = __riscv_vfadd_vv_f32m1(vR5, vA, vl);
                vR8 = __riscv_vfsub_vv_f32m1(vR7, vR6, vl);
                vR7 = __riscv_vfadd_vv_f32m1(vR7, vR6, vl);
                vt2 = __riscv_vfsub_vv_f32m1(vS5, vB, vl);
                vS5 = __riscv_vfadd_vv_f32m1(vS5, vB, vl);
                vS8 = __riscv_vfsub_vv_f32m1(vS7, vS6, vl);
                vS7 = __riscv_vfadd_vv_f32m1(vS7, vS6, vl);

                vR1 = __riscv_vfadd_vv_f32m1(vR5, vS7, vl);
                vR5 = __riscv_vfsub_vv_f32m1(vR5, vS7, vl);
                vR6 = __riscv_vfadd_vv_f32m1(vt1, vS8, vl);
                vt1 = __riscv_vfsub_vv_f32m1(vt1, vS8, vl);
                vS1 = __riscv_vfsub_vv_f32m1(vS5, vR7, vl);
                vS5 = __riscv_vfadd_vv_f32m1(vS5, vR7, vl);
                vS6 = __riscv_vfsub_vv_f32m1(vt2, vR8, vl);
                vt2 = __riscv_vfadd_vv_f32m1(vt2, vR8, vl);

                vfloat32m1_t vi2r = __riscv_vfmacc_vf_f32m1(
                    __riscv_vfmul_vf_f32m1(vR1, co2, vl), si2, vS1, vl);
                vfloat32m1_t vi2i = __riscv_vfnmsac_vf_f32m1(
                    __riscv_vfmul_vf_f32m1(vS1, co2, vl), si2, vR1, vl);

                vfloat32m1_t vi8r = __riscv_vfmacc_vf_f32m1(
                    __riscv_vfmul_vf_f32m1(vR5, co8, vl), si8, vS5, vl);
                vfloat32m1_t vi8i = __riscv_vfnmsac_vf_f32m1(
                    __riscv_vfmul_vf_f32m1(vS5, co8, vl), si8, vR5, vl);

                vfloat32m1_t vi6r = __riscv_vfmacc_vf_f32m1(
                    __riscv_vfmul_vf_f32m1(vR6, co6, vl), si6, vS6, vl);
                vfloat32m1_t vi6i = __riscv_vfnmsac_vf_f32m1(
                    __riscv_vfmul_vf_f32m1(vS6, co6, vl), si6, vR6, vl);

                vfloat32m1_t vi4r = __riscv_vfmacc_vf_f32m1(
                    __riscv_vfmul_vf_f32m1(vt1, co4, vl), si4, vt2, vl);
                vfloat32m1_t vi4i = __riscv_vfnmsac_vf_f32m1(
                    __riscv_vfmul_vf_f32m1(vt2, co4, vl), si4, vt1, vl);

                __riscv_vsse32_v_f32m1(b, stride, vi1r, vl);
                __riscv_vsse32_v_f32m1(b + 1, stride, vi1i, vl);
                __riscv_vsse32_v_f32m1(b + 2 * n2, stride, vi2r, vl);
                __riscv_vsse32_v_f32m1(b + 2 * n2 + 1, stride, vi2i, vl);
                __riscv_vsse32_v_f32m1(b + 4 * n2, stride, vi3r, vl);
                __riscv_vsse32_v_f32m1(b + 4 * n2 + 1, stride, vi3i, vl);
                __riscv_vsse32_v_f32m1(b + 6 * n2, stride, vi4r, vl);
                __riscv_vsse32_v_f32m1(b + 6 * n2 + 1, stride, vi4i, vl);
                __riscv_vsse32_v_f32m1(b + 8 * n2, stride, vi5r, vl);
                __riscv_vsse32_v_f32m1(b + 8 * n2 + 1, stride, vi5i, vl);
                __riscv_vsse32_v_f32m1(b + 10 * n2, stride, vi6r, vl);
                __riscv_vsse32_v_f32m1(b + 10 * n2 + 1, stride, vi6i, vl);
                __riscv_vsse32_v_f32m1(b + 12 * n2, stride, vi7r, vl);
                __riscv_vsse32_v_f32m1(b + 12 * n2 + 1, stride, vi7i, vl);
                __riscv_vsse32_v_f32m1(b + 14 * n2, stride, vi8r, vl);
                __riscv_vsse32_v_f32m1(b + 14 * n2 + 1, stride, vi8i, vl);

                i1 += (uint32_t)vl * n1;
            }

            j++;
        } while (j < n2);

        twidCoefModifier <<= 3;
    } while (n2 > 7);
}

static void
riscv_bitreversal_32(uint32_t       *pSrc,
                     const uint16_t  bitRevLen,
                     const uint16_t *pBitRevTab)
{
    uint32_t a, b, i, tmp;

    for (i = 0; i < bitRevLen;)
    {
        a = pBitRevTab[i] >> 2;
        b = pBitRevTab[i + 1] >> 2;

        tmp     = pSrc[a];
        pSrc[a] = pSrc[b];
        pSrc[b] = tmp;

        tmp         = pSrc[a + 1];
        pSrc[a + 1] = pSrc[b + 1];
        pSrc[b + 1] = tmp;

        i += 2;
    }
}

static void
riscv_cfft_radix8by2_f32(riscv_cfft_instance_f32 *S, float32_t *p1)
{
    uint32_t         L = S->fftLen;
    float32_t       *pCol1, *pCol2, *pMid1, *pMid2;
    float32_t       *p2 = p1 + L;
    const float32_t *tw = (float32_t *)S->pTwiddle;

    pCol1 = p1;
    pCol2 = p2;
    L >>= 1;
    pMid1 = p1 + L;
    pMid2 = p2 + L;

    uint32_t l = L >> 1;
    while (l > 0)
    {
        size_t vl = __riscv_vsetvl_e32m2(l);

        vfloat32m2_t vt1r, vt1i, vt2r, vt2i, vt3r, vt3i, vt4r, vt4i;
        vfloat32m2_t vtwR, vtwI;

        ssize_t stride = 2 * sizeof(float32_t);
        vt1r           = __riscv_vlse32_v_f32m2(p1, stride, vl);
        vt1i           = __riscv_vlse32_v_f32m2(p1 + 1, stride, vl);
        vt2r           = __riscv_vlse32_v_f32m2(p2, stride, vl);
        vt2i           = __riscv_vlse32_v_f32m2(p2 + 1, stride, vl);
        vt3r           = __riscv_vlse32_v_f32m2(pMid1, stride, vl);
        vt3i           = __riscv_vlse32_v_f32m2(pMid1 + 1, stride, vl);
        vt4r           = __riscv_vlse32_v_f32m2(pMid2, stride, vl);
        vt4i           = __riscv_vlse32_v_f32m2(pMid2 + 1, stride, vl);
        vtwR           = __riscv_vlse32_v_f32m2(tw, stride, vl);
        vtwI           = __riscv_vlse32_v_f32m2(tw + 1, stride, vl);

        vfloat32m2_t vsum1r = __riscv_vfadd_vv_f32m2(vt1r, vt2r, vl);
        vfloat32m2_t vsum1i = __riscv_vfadd_vv_f32m2(vt1i, vt2i, vl);
        __riscv_vsse32_v_f32m2(p1, stride, vsum1r, vl);
        __riscv_vsse32_v_f32m2(p1 + 1, stride, vsum1i, vl);

        vfloat32m2_t vd2r = __riscv_vfsub_vv_f32m2(vt1r, vt2r, vl);
        vfloat32m2_t vd2i = __riscv_vfsub_vv_f32m2(vt1i, vt2i, vl);

        vfloat32m2_t vsum3r = __riscv_vfadd_vv_f32m2(vt3r, vt4r, vl);
        vfloat32m2_t vsum3i = __riscv_vfadd_vv_f32m2(vt3i, vt4i, vl);
        __riscv_vsse32_v_f32m2(pMid1, stride, vsum3r, vl);
        __riscv_vsse32_v_f32m2(pMid1 + 1, stride, vsum3i, vl);

        vfloat32m2_t vd4r = __riscv_vfsub_vv_f32m2(vt4r, vt3r, vl);
        vfloat32m2_t vd4i = __riscv_vfsub_vv_f32m2(vt4i, vt3i, vl);

        vfloat32m2_t vp2r = __riscv_vfmacc_vv_f32m2(
            __riscv_vfmul_vv_f32m2(vd2r, vtwR, vl), vd2i, vtwI, vl);
        vfloat32m2_t vp2i = __riscv_vfnmsac_vv_f32m2(
            __riscv_vfmul_vv_f32m2(vd2i, vtwR, vl), vd2r, vtwI, vl);
        __riscv_vsse32_v_f32m2(p2, stride, vp2r, vl);
        __riscv_vsse32_v_f32m2(p2 + 1, stride, vp2i, vl);

        vfloat32m2_t vm4r = __riscv_vfnmsac_vv_f32m2(
            __riscv_vfmul_vv_f32m2(vd4r, vtwI, vl), vd4i, vtwR, vl);
        vfloat32m2_t vm4i = __riscv_vfmacc_vv_f32m2(
            __riscv_vfmul_vv_f32m2(vd4i, vtwI, vl), vd4r, vtwR, vl);
        __riscv_vsse32_v_f32m2(pMid2, stride, vm4r, vl);
        __riscv_vsse32_v_f32m2(pMid2 + 1, stride, vm4i, vl);

        p1 += 2 * vl;
        p2 += 2 * vl;
        pMid1 += 2 * vl;
        pMid2 += 2 * vl;
        tw += 2 * vl;
        l -= vl;
    }

    riscv_radix8_butterfly_f32(pCol1, L, (float32_t *)S->pTwiddle, 2U);
    riscv_radix8_butterfly_f32(pCol2, L, (float32_t *)S->pTwiddle, 2U);
}

static void
riscv_cfft_radix8by4_f32(riscv_cfft_instance_f32 *S, float32_t *p1)
{
    uint32_t         L = S->fftLen >> 1;
    const float32_t *tw2, *tw3, *tw4;
    float32_t       *p2    = p1 + L;
    float32_t       *p3    = p2 + L;
    float32_t       *p4    = p3 + L;
    float32_t       *pCol1 = p1;
    float32_t       *pCol2 = p2;
    float32_t       *pCol3 = p3;
    float32_t       *pCol4 = p4;
    float32_t       *pEnd1 = p2 - 1;
    float32_t       *pEnd2 = p3 - 1;
    float32_t       *pEnd3 = p4 - 1;
    float32_t       *pEnd4 = pEnd3 + L;

    tw2 = tw3 = tw4 = (float32_t *)S->pTwiddle;

    L >>= 1;

    float32_t p1ap3_0, p1sp3_0, p1ap3_1, p1sp3_1;
    float32_t t2[4], t3[4], t4[4], twR, twI;
    float32_t m0, m1, m2, m3;

    p1ap3_0 = p1[0] + p3[0];
    p1sp3_0 = p1[0] - p3[0];
    p1ap3_1 = p1[1] + p3[1];
    p1sp3_1 = p1[1] - p3[1];
    t2[0]   = p1sp3_0 + p2[1] - p4[1];
    t2[1]   = p1sp3_1 - p2[0] + p4[0];
    t3[0]   = p1ap3_0 - p2[0] - p4[0];
    t3[1]   = p1ap3_1 - p2[1] - p4[1];
    t4[0]   = p1sp3_0 - p2[1] + p4[1];
    t4[1]   = p1sp3_1 + p2[0] - p4[0];
    *p1++   = p1ap3_0 + p2[0] + p4[0];
    *p1++   = p1ap3_1 + p2[1] + p4[1];
    *p2++   = t2[0];
    *p2++   = t2[1];
    *p3++   = t3[0];
    *p3++   = t3[1];
    *p4++   = t4[0];
    *p4++   = t4[1];
    tw2 += 2;
    tw3 += 4;
    tw4 += 6;

    uint32_t l = (L - 2) >> 1;
    while (l > 0)
    {
        size_t vl = __riscv_vsetvl_e32m1(l);

        vfloat32m1_t vp1r, vp1i, vp2r, vp2i, vp3r, vp3i, vp4r, vp4i;
        ssize_t      stride = 2 * sizeof(float32_t);
        vp1r                = __riscv_vlse32_v_f32m1(p1, stride, vl);
        vp1i                = __riscv_vlse32_v_f32m1(p1 + 1, stride, vl);
        vp2r                = __riscv_vlse32_v_f32m1(p2, stride, vl);
        vp2i                = __riscv_vlse32_v_f32m1(p2 + 1, stride, vl);
        vp3r                = __riscv_vlse32_v_f32m1(p3, stride, vl);
        vp3i                = __riscv_vlse32_v_f32m1(p3 + 1, stride, vl);
        vp4r                = __riscv_vlse32_v_f32m1(p4, stride, vl);
        vp4i                = __riscv_vlse32_v_f32m1(p4 + 1, stride, vl);

        vfloat32m1_t va0 = __riscv_vfadd_vv_f32m1(vp1r, vp3r, vl);
        vfloat32m1_t vs0 = __riscv_vfsub_vv_f32m1(vp1r, vp3r, vl);
        vfloat32m1_t va1 = __riscv_vfadd_vv_f32m1(vp1i, vp3i, vl);
        vfloat32m1_t vs1 = __riscv_vfsub_vv_f32m1(vp1i, vp3i, vl);

        vfloat32m1_t vout1r = __riscv_vfadd_vv_f32m1(
            va0, __riscv_vfadd_vv_f32m1(vp2r, vp4r, vl), vl);
        vfloat32m1_t vout1i = __riscv_vfadd_vv_f32m1(
            va1, __riscv_vfadd_vv_f32m1(vp2i, vp4i, vl), vl);
        __riscv_vsse32_v_f32m1(p1, stride, vout1r, vl);
        __riscv_vsse32_v_f32m1(p1 + 1, stride, vout1i, vl);

        vfloat32m1_t vt2r = __riscv_vfadd_vv_f32m1(
            __riscv_vfsub_vv_f32m1(vs0, vp4i, vl), vp2i, vl);
        vfloat32m1_t vt2i = __riscv_vfadd_vv_f32m1(
            __riscv_vfsub_vv_f32m1(vs1, vp2r, vl), vp4r, vl);
        vfloat32m1_t vt3r = __riscv_vfsub_vv_f32m1(
            __riscv_vfsub_vv_f32m1(va0, vp2r, vl), vp4r, vl);
        vfloat32m1_t vt3i = __riscv_vfsub_vv_f32m1(
            __riscv_vfsub_vv_f32m1(va1, vp2i, vl), vp4i, vl);
        vfloat32m1_t vt4r = __riscv_vfadd_vv_f32m1(
            __riscv_vfsub_vv_f32m1(vs0, vp2i, vl), vp4i, vl);
        vfloat32m1_t vt4i = __riscv_vfadd_vv_f32m1(
            __riscv_vfsub_vv_f32m1(vs1, vp4r, vl), vp2r, vl);

        vfloat32m1_t vtw2R, vtw2I, vtw3R, vtw3I, vtw4R, vtw4I;
        ssize_t      stride2 = 2 * sizeof(float32_t);
        vtw2R                = __riscv_vlse32_v_f32m1(tw2, stride2, vl);
        vtw2I                = __riscv_vlse32_v_f32m1(tw2 + 1, stride2, vl);

        ssize_t stride3 = 4 * sizeof(float32_t);
        vtw3R           = __riscv_vlse32_v_f32m1(tw3, stride3, vl);
        vtw3I           = __riscv_vlse32_v_f32m1(tw3 + 1, stride3, vl);

        ssize_t stride4 = 6 * sizeof(float32_t);
        vtw4R           = __riscv_vlse32_v_f32m1(tw4, stride4, vl);
        vtw4I           = __riscv_vlse32_v_f32m1(tw4 + 1, stride4, vl);

        vfloat32m1_t vout2r = __riscv_vfmacc_vv_f32m1(
            __riscv_vfmul_vv_f32m1(vt2r, vtw2R, vl), vt2i, vtw2I, vl);
        vfloat32m1_t vout2i = __riscv_vfnmsac_vv_f32m1(
            __riscv_vfmul_vv_f32m1(vt2i, vtw2R, vl), vt2r, vtw2I, vl);
        __riscv_vsse32_v_f32m1(p2, stride, vout2r, vl);
        __riscv_vsse32_v_f32m1(p2 + 1, stride, vout2i, vl);

        vfloat32m1_t vout3r = __riscv_vfmacc_vv_f32m1(
            __riscv_vfmul_vv_f32m1(vt3r, vtw3R, vl), vt3i, vtw3I, vl);
        vfloat32m1_t vout3i = __riscv_vfnmsac_vv_f32m1(
            __riscv_vfmul_vv_f32m1(vt3i, vtw3R, vl), vt3r, vtw3I, vl);
        __riscv_vsse32_v_f32m1(p3, stride, vout3r, vl);
        __riscv_vsse32_v_f32m1(p3 + 1, stride, vout3i, vl);

        vfloat32m1_t vout4r = __riscv_vfmacc_vv_f32m1(
            __riscv_vfmul_vv_f32m1(vt4r, vtw4R, vl), vt4i, vtw4I, vl);
        vfloat32m1_t vout4i = __riscv_vfnmsac_vv_f32m1(
            __riscv_vfmul_vv_f32m1(vt4i, vtw4R, vl), vt4r, vtw4I, vl);
        __riscv_vsse32_v_f32m1(p4, stride, vout4r, vl);
        __riscv_vsse32_v_f32m1(p4 + 1, stride, vout4i, vl);

        ssize_t bcs = -2 * (ssize_t)sizeof(float32_t);

        vfloat32m1_t ve1r = __riscv_vlse32_v_f32m1(pEnd1 - 1, bcs, vl);
        vfloat32m1_t ve1i = __riscv_vlse32_v_f32m1(pEnd1, bcs, vl);
        vfloat32m1_t ve2r = __riscv_vlse32_v_f32m1(pEnd2 - 1, bcs, vl);
        vfloat32m1_t ve2i = __riscv_vlse32_v_f32m1(pEnd2, bcs, vl);
        vfloat32m1_t ve3r = __riscv_vlse32_v_f32m1(pEnd3 - 1, bcs, vl);
        vfloat32m1_t ve3i = __riscv_vlse32_v_f32m1(pEnd3, bcs, vl);
        vfloat32m1_t ve4r = __riscv_vlse32_v_f32m1(pEnd4 - 1, bcs, vl);
        vfloat32m1_t ve4i = __riscv_vlse32_v_f32m1(pEnd4, bcs, vl);

        vfloat32m1_t vea_r = __riscv_vfadd_vv_f32m1(ve1r, ve3r, vl);
        vfloat32m1_t ves_r = __riscv_vfsub_vv_f32m1(ve1r, ve3r, vl);
        vfloat32m1_t vea_i = __riscv_vfadd_vv_f32m1(ve1i, ve3i, vl);
        vfloat32m1_t ves_i = __riscv_vfsub_vv_f32m1(ve1i, ve3i, vl);

        __riscv_vsse32_v_f32m1(
            pEnd1 - 1,
            bcs,
            __riscv_vfadd_vv_f32m1(
                vea_r, __riscv_vfadd_vv_f32m1(ve2r, ve4r, vl), vl),
            vl);
        __riscv_vsse32_v_f32m1(
            pEnd1,
            bcs,
            __riscv_vfadd_vv_f32m1(
                vea_i, __riscv_vfadd_vv_f32m1(ve2i, ve4i, vl), vl),
            vl);

        vfloat32m1_t vet2_2 = __riscv_vfadd_vv_f32m1(
            __riscv_vfsub_vv_f32m1(ve2i, ve4i, vl), ves_r, vl);
        vfloat32m1_t vet2_3 = __riscv_vfadd_vv_f32m1(
            __riscv_vfsub_vv_f32m1(ves_i, ve2r, vl), ve4r, vl);
        vfloat32m1_t vet3_2 = __riscv_vfsub_vv_f32m1(
            __riscv_vfsub_vv_f32m1(vea_r, ve2r, vl), ve4r, vl);
        vfloat32m1_t vet3_3 = __riscv_vfsub_vv_f32m1(
            __riscv_vfsub_vv_f32m1(vea_i, ve2i, vl), ve4i, vl);
        vfloat32m1_t vet4_2 = __riscv_vfsub_vv_f32m1(
            __riscv_vfsub_vv_f32m1(ve2i, ve4i, vl), ves_r, vl);
        vfloat32m1_t vet4_3 = __riscv_vfsub_vv_f32m1(
            __riscv_vfsub_vv_f32m1(ve4r, ve2r, vl), ves_i, vl);

        __riscv_vsse32_v_f32m1(
            pEnd2 - 1,
            bcs,
            __riscv_vfmacc_vv_f32m1(
                __riscv_vfmul_vv_f32m1(vet2_2, vtw2I, vl), vet2_3, vtw2R, vl),
            vl);
        __riscv_vsse32_v_f32m1(
            pEnd2,
            bcs,
            __riscv_vfsub_vv_f32m1(__riscv_vfmul_vv_f32m1(vet2_3, vtw2I, vl),
                                   __riscv_vfmul_vv_f32m1(vet2_2, vtw2R, vl),
                                   vl),
            vl);

        __riscv_vsse32_v_f32m1(
            pEnd3 - 1,
            bcs,
            __riscv_vfsub_vv_f32m1(__riscv_vfmul_vv_f32m1(vet3_3, vtw3I, vl),
                                   __riscv_vfmul_vv_f32m1(vet3_2, vtw3R, vl),
                                   vl),
            vl);
        __riscv_vsse32_v_f32m1(
            pEnd3,
            bcs,
            __riscv_vfneg_v_f32m1(__riscv_vfmacc_vv_f32m1(
                                      __riscv_vfmul_vv_f32m1(vet3_3, vtw3R, vl),
                                      vet3_2,
                                      vtw3I,
                                      vl),
                                  vl),
            vl);

        __riscv_vsse32_v_f32m1(
            pEnd4 - 1,
            bcs,
            __riscv_vfmacc_vv_f32m1(
                __riscv_vfmul_vv_f32m1(vet4_2, vtw4I, vl), vet4_3, vtw4R, vl),
            vl);
        __riscv_vsse32_v_f32m1(
            pEnd4,
            bcs,
            __riscv_vfsub_vv_f32m1(__riscv_vfmul_vv_f32m1(vet4_3, vtw4I, vl),
                                   __riscv_vfmul_vv_f32m1(vet4_2, vtw4R, vl),
                                   vl),
            vl);

        p1 += 2 * vl;
        p2 += 2 * vl;
        p3 += 2 * vl;
        p4 += 2 * vl;
        pEnd1 -= 2 * vl;
        pEnd2 -= 2 * vl;
        pEnd3 -= 2 * vl;
        pEnd4 -= 2 * vl;
        tw2 += 2 * vl;
        tw3 += 4 * vl;
        tw4 += 6 * vl;
        l -= vl;
    }

    p1ap3_0 = p1[0] + p3[0];
    p1sp3_0 = p1[0] - p3[0];
    p1ap3_1 = p1[1] + p3[1];
    p1sp3_1 = p1[1] - p3[1];
    t2[0]   = p1sp3_0 + p2[1] - p4[1];
    t2[1]   = p1sp3_1 - p2[0] + p4[0];
    t3[0]   = p1ap3_0 - p2[0] - p4[0];
    t3[1]   = p1ap3_1 - p2[1] - p4[1];
    t4[0]   = p1sp3_0 - p2[1] + p4[1];
    t4[1]   = p1sp3_1 + p2[0] - p4[0];
    *p1++   = p1ap3_0 + p2[0] + p4[0];
    *p1++   = p1ap3_1 + p2[1] + p4[1];
    twR     = tw2[0];
    twI     = tw2[1];
    m0      = t2[0] * twR;
    m1      = t2[1] * twI;
    m2      = t2[1] * twR;
    m3      = t2[0] * twI;
    *p2++   = m0 + m1;
    *p2++   = m2 - m3;
    twR     = tw3[0];
    twI     = tw3[1];
    m0      = t3[0] * twR;
    m1      = t3[1] * twI;
    m2      = t3[1] * twR;
    m3      = t3[0] * twI;
    *p3++   = m0 + m1;
    *p3++   = m2 - m3;
    twR     = tw4[0];
    twI     = tw4[1];
    m0      = t4[0] * twR;
    m1      = t4[1] * twI;
    m2      = t4[1] * twR;
    m3      = t4[0] * twI;
    *p4++   = m0 + m1;
    *p4++   = m2 - m3;

    riscv_radix8_butterfly_f32(pCol1, L, (float32_t *)S->pTwiddle, 4U);
    riscv_radix8_butterfly_f32(pCol2, L, (float32_t *)S->pTwiddle, 4U);
    riscv_radix8_butterfly_f32(pCol3, L, (float32_t *)S->pTwiddle, 4U);
    riscv_radix8_butterfly_f32(pCol4, L, (float32_t *)S->pTwiddle, 4U);
}

void
riscv_cfft_f32(const riscv_cfft_instance_f32 *S,
               float32_t                     *p1,
               uint8_t                        ifftFlag,
               uint8_t                        bitReverseFlag)
{
    uint32_t  L = S->fftLen, l;
    float32_t invL, *pSrc;

    if (ifftFlag == 1U)
    {
        pSrc = p1 + 1;
        l    = L;
        while (l > 0)
        {
            size_t       vl = __riscv_vsetvl_e32m4(l);
            vfloat32m4_t v
                = __riscv_vlse32_v_f32m4(pSrc, 2 * sizeof(float32_t), vl);
            __riscv_vsse32_v_f32m4(
                pSrc, 2 * sizeof(float32_t), __riscv_vfneg_v_f32m4(v, vl), vl);
            pSrc += 2 * vl;
            l -= vl;
        }
    }

    switch (L)
    {
        case 16:
        case 128:
        case 1024:
            riscv_cfft_radix8by2_f32((riscv_cfft_instance_f32 *)S, p1);
            break;
        case 32:
        case 256:
        case 2048:
            riscv_cfft_radix8by4_f32((riscv_cfft_instance_f32 *)S, p1);
            break;
        case 64:
        case 512:
        case 4096:
            riscv_radix8_butterfly_f32(p1, L, (float32_t *)S->pTwiddle, 1);
            break;
    }

    if (bitReverseFlag)
    {
        riscv_bitreversal_32((uint32_t *)p1, S->bitRevLength, S->pBitRevTable);
    }

    if (ifftFlag == 1U)
    {
        invL = 1.0f / (float32_t)L;
        pSrc = p1;
        l    = L;
        while (l > 0)
        {
            size_t       vl = __riscv_vsetvl_e32m4(l);
            vfloat32m4_t vR, vI;
            vR = __riscv_vlse32_v_f32m4(pSrc, 2 * sizeof(float32_t), vl);
            vI = __riscv_vlse32_v_f32m4(pSrc + 1, 2 * sizeof(float32_t), vl);
            vR = __riscv_vfmul_vf_f32m4(vR, invL, vl);
            vI = __riscv_vfmul_vf_f32m4(
                __riscv_vfneg_v_f32m4(vI, vl), invL, vl);
            __riscv_vsse32_v_f32m4(pSrc, 2 * sizeof(float32_t), vR, vl);
            __riscv_vsse32_v_f32m4(pSrc + 1, 2 * sizeof(float32_t), vI, vl);
            pSrc += 2 * vl;
            l -= vl;
        }
    }
}
