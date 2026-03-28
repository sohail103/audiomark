/**
 * Copyright 2026 Sohail Raj Satapathy
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ee_api.h"

#define Q31_MAX 0x7FFFFFFFL


void th_absmax_f32(
  const float * pSrc,
        uint32_t blockSize,
        float * pResult,
        uint32_t * pIndex)
{
    /* Unroll by 4 */
    uint32_t blkCnt = blockSize >> 2; 


    /* Pipeline 0: Processes elements 0, 1, 4, 5, 8, 9... */
    uint32x2_t maxValVec0     = __riscv_pjoin2_u32x2(0, 0);
    uint32x2_t maxIdxVec0     = __riscv_pjoin2_u32x2(0, 1);
    uint32x2_t currentIdxVec0 = __riscv_pjoin2_u32x2(0, 1);

    /* Pipeline 1: Processes elements 2, 3, 6, 7, 10, 11... */
    uint32x2_t maxValVec1     = __riscv_pjoin2_u32x2(0, 0);
    uint32x2_t maxIdxVec1     = __riscv_pjoin2_u32x2(2, 3);
    uint32x2_t currentIdxVec1 = __riscv_pjoin2_u32x2(2, 3);

    /* Both pipelines advance by 4 elements per loop */
    uint32x2_t incrementVec   = __riscv_pjoin2_u32x2(4, 4);

    /* "Do not apply strict aliasing optimization to this type" */
    typedef uint32_t __attribute__((__may_alias__)) alias_u32_t;
    const alias_u32_t *pIn = (const alias_u32_t *)pSrc;

    uint32x2_t absMask = __riscv_pjoin2_u32x2(0x7FFFFFFFU, 0x7FFFFFFFU);

    while (blkCnt > 0)
    {
        uint32x2_t val0 = __riscv_pload_u32x2(pIn);
        uint32x2_t val1 = __riscv_pload_u32x2(pIn + 2);
        pIn += 4;

        uint32x2_t abs0 = __riscv_pand_u32x2(val0, absMask);
        uint32x2_t abs1 = __riscv_pand_u32x2(val1, absMask);

        /* Generate Hardware Masks */
        uint32x2_t mask0 = __riscv_pmsltu_u32x2(maxValVec0, abs0);
        uint32x2_t mask1 = __riscv_pmsltu_u32x2(maxValVec1, abs1);

        /* Branchless Merges (Pipelines execute independently) */
        maxValVec0 = __riscv_pmerge_u32x2(mask0, maxValVec0, abs0);
        maxIdxVec0 = __riscv_pmerge_u32x2(mask0, maxIdxVec0, currentIdxVec0);

        maxValVec1 = __riscv_pmerge_u32x2(mask1, maxValVec1, abs1);
        maxIdxVec1 = __riscv_pmerge_u32x2(mask1, maxIdxVec1, currentIdxVec1);

        /* Advance Indices */
        currentIdxVec0 = __riscv_padd_u32x2(currentIdxVec0, incrementVec);
        currentIdxVec1 = __riscv_padd_u32x2(currentIdxVec1, incrementVec);

        blkCnt--;
    }


    /* Merge Pipeline 1 into Pipeline 0 */
    uint32x2_t mergeMask = __riscv_pmsltu_u32x2(maxValVec0, maxValVec1);
    maxValVec0 = __riscv_pmerge_u32x2(mergeMask, maxValVec0, maxValVec1);
    maxIdxVec0 = __riscv_pmerge_u32x2(mergeMask, maxIdxVec0, maxIdxVec1);

    /* Horizontal Reduction (Cross the lanes of the surviving pipeline) */
    uint32x2_t swappedMax = __riscv_ppairoe_u32x2(maxValVec0, maxValVec0);
    uint32x2_t swappedIdx = __riscv_ppairoe_u32x2(maxIdxVec0, maxIdxVec0);

    uint32x2_t finalMask = __riscv_pmsltu_u32x2(maxValVec0, swappedMax);

    uint32x2_t finalMaxVec = __riscv_pmerge_u32x2(finalMask, maxValVec0, swappedMax);
    uint32x2_t finalIdxVec = __riscv_pmerge_u32x2(finalMask, maxIdxVec0, swappedIdx);

    /* Extract the final scalar winner from Lane 0 */
    uint32_t finalMax = __riscv_pget_u32x2_u32(finalMaxVec, 0);
    uint32_t finalIdx = __riscv_pget_u32x2_u32(finalIdxVec, 0);

    /* We unrolled by 4, there could be 1, 2, or 3 elements left over */
    blkCnt = blockSize & 3U; 
    uint32_t tailIdx = blockSize - blkCnt;

    while (blkCnt > 0) {
        /* Safely read float as integer using the alias type */
        uint32_t tailVal = ((const alias_u32_t *)pSrc)[tailIdx] & 0x7FFFFFFFU;

        if (tailVal > finalMax && tailVal < 0x7F800001U) {
            finalMax = tailVal;
            finalIdx = tailIdx;
        }
        tailIdx++;
        blkCnt--;
    }

    union {
        uint32_t i;
        float f;
    } converter;

    converter.i = finalMax;
    *pResult = converter.f;
    *pIndex = finalIdx;

}
