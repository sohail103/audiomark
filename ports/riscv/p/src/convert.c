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

#include "convert.h"
#include "ee_api.h"
#include "th_types.h"
#include "rvp_support_guard.h"

void
riscv_q31_to_float(const q31_t *pSrc, float *pDst, uint32_t blockSize)
{
    uint32_t blkCnt         = blockSize >> 2U;
    uint32_t i              = 0;
    float    inv_multiplier = 1.0f / 2147483648.0f;

    while (blkCnt > 0U)
    {
        pDst[i] = (float)pSrc[i] * inv_multiplier;
        i++;
        pDst[i] = (float)pSrc[i] * inv_multiplier;
        i++;
        pDst[i] = (float)pSrc[i] * inv_multiplier;
        i++;
        pDst[i] = (float)pSrc[i] * inv_multiplier;
        i++;
        blkCnt--;
    }

    blkCnt = blockSize % 0x4U;
    while (blkCnt > 0U)
    {
        pDst[i] = (float)pSrc[i] * inv_multiplier;
        i++;
        blkCnt--;
    }
}

void
riscv_q31_to_float_unnormalize(const q31_t *pSrc,
                               float       *pDst,
                               uint32_t     blockSize,
                               float        inv_multiplier)
{
    uint32_t blkCnt = blockSize >> 2U;
    uint32_t i      = 0;

    while (blkCnt > 0U)
    {
        pDst[i] = (float)pSrc[i] * inv_multiplier;
        i++;
        pDst[i] = (float)pSrc[i] * inv_multiplier;
        i++;
        pDst[i] = (float)pSrc[i] * inv_multiplier;
        i++;
        pDst[i] = (float)pSrc[i] * inv_multiplier;
        i++;
        blkCnt--;
    }

    blkCnt = blockSize % 0x4U;
    while (blkCnt > 0U)
    {
        pDst[i] = (float)pSrc[i] * inv_multiplier;
        i++;
        blkCnt--;
    }
}

void
riscv_float_to_q31(const float *pSrc, q31_t *pDst, uint32_t blockSize)
{
    uint32_t blkCnt = blockSize >> 2U; /* Divide by 4 */
    uint32_t i      = 0;
    float    in;

    while (blkCnt > 0U)
    {
        in = pSrc[i] * 2147483648.0f;
        in += (in > 0.0f) ? 0.5f : -0.5f;
        pDst[i++] = (q31_t)in;
        in        = pSrc[i] * 2147483648.0f;
        in += (in > 0.0f) ? 0.5f : -0.5f;
        pDst[i++] = (q31_t)in;
        in        = pSrc[i] * 2147483648.0f;
        in += (in > 0.0f) ? 0.5f : -0.5f;
        pDst[i++] = (q31_t)in;
        in        = pSrc[i] * 2147483648.0f;
        in += (in > 0.0f) ? 0.5f : -0.5f;
        pDst[i++] = (q31_t)in;
        blkCnt--;
    }

    /* Tail handling for sizes not perfectly divisible by 4 */
    blkCnt = blockSize % 0x4U;
    while (blkCnt > 0U)
    {
        in = pSrc[i] * 2147483648.0f;
        in += (in > 0.0f) ? 0.5f : -0.5f;
        pDst[i++] = (q31_t)in;
        blkCnt--;
    }
}

float
riscv_float_to_q31_normalize(const float *pSrc, q31_t *pDst, uint32_t blockSize)
{
    float    max_val = 0.0f;
    uint32_t max_idx = 0;

    th_absmax_f32(pSrc, blockSize, &max_val, &max_idx);

    /* Calculate the Fused Multiplier */
    float scale_factor        = 1.0f;
    float combined_multiplier = 2147483648.0f;

    if (max_val > 0.0f)
    {
        scale_factor = 0.499999f / max_val;
        combined_multiplier *= scale_factor;
    }

    uint32_t blkCnt = blockSize >> 2U;
    uint32_t i      = 0;
    float    in;

    while (blkCnt > 0U)
    {
        in = pSrc[i] * combined_multiplier;
        in += (in > 0.0f) ? 0.5f : -0.5f;
        pDst[i++] = (q31_t)in;
        in        = pSrc[i] * combined_multiplier;
        in += (in > 0.0f) ? 0.5f : -0.5f;
        pDst[i++] = (q31_t)in;
        in        = pSrc[i] * combined_multiplier;
        in += (in > 0.0f) ? 0.5f : -0.5f;
        pDst[i++] = (q31_t)in;
        in        = pSrc[i] * combined_multiplier;
        in += (in > 0.0f) ? 0.5f : -0.5f;
        pDst[i++] = (q31_t)in;
        blkCnt--;
    }

    blkCnt = blockSize % 0x4U;
    while (blkCnt > 0U)
    {
        in = pSrc[i] * combined_multiplier;
        in += (in > 0.0f) ? 0.5f : -0.5f;
        pDst[i++] = (q31_t)in;
        blkCnt--;
    }

    return scale_factor;
}
