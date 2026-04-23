/*
 * Copyright (C) 2022 Arm Limited or its affiliates.
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
 *
 * Modifications copyright (C) 2021-2022 Chair of Electronic Design Automation,
 * TUM
 * Modifications copyright (C) 2026 Sohail Raj Satapathy
 */

#include "functions.h"
#include "support_functions.h"

#define ACCUM_BITS 12
#define Q7_MAX     ((q7_t)(0x7F))
#define Q7_MIN     ((q7_t)(0x80))

/*
 * Softmax function with s8 input and output of s8.
 */

void
nn_softmax_row12_s8(const int8_t *input, int8_t *output)
{
    const int32_t mult     = 1881344896;
    const int32_t shift    = 24;
    const int32_t diff_min = -124;

    /* Safe mask: (1 << shift) */
    const int32_t mask = (int32_t)((int64_t)1 << shift);

    /* Step 1: find max */
    int8_t max = input[0];
    for (int i = 1; i < 12; i++)
    {
        if (input[i] > max)
        {
            max = input[i];
        }
    }

    /* Step 2: compute sum of exps */
    int32_t sum = 0;

    for (int i = 0; i < 12; i++)
    {
        int32_t diff = (int32_t)input[i] - (int32_t)max;

        if (diff >= diff_min)
        {
            /* safe scaling: diff * mask */
            int32_t scaled = (int32_t)((int64_t)diff * mask);

            int32_t prod = MUL_SAT(scaled, mult);
            int32_t expv = EXP_ON_NEG(prod);

            sum += DIV_POW2(expv, ACCUM_BITS);
        }
    }

    /* Step 3: normalization */
    int32_t headroom = __builtin_clz(sum);

    int32_t shifted_sum = (sum > 0) ? (int32_t)((int64_t)sum << headroom) : 0;

    int32_t shifted_scale
        = ONE_OVER1(shifted_sum - (int32_t)((uint32_t)1 << 31));

    int32_t bits_over_unit = ACCUM_BITS - headroom + 23;

    /* Step 4: output */
    for (int i = 0; i < 12; i++)
    {
        int32_t diff = (int32_t)input[i] - (int32_t)max;

        if (diff >= diff_min)
        {
            int32_t scaled = (int32_t)((int64_t)diff * mask);
            int32_t prod   = MUL_SAT(scaled, mult);
            int32_t expv   = EXP_ON_NEG(prod);

            int32_t res = DIV_POW2(MUL_SAT(shifted_scale, expv), bits_over_unit)
                          + Q7_MIN;

            /* clamp */
            if (res > Q7_MAX)
            {
                res = Q7_MAX;
            }
            if (res < Q7_MIN)
            {
                res = Q7_MIN;
            }

            output[i] = (int8_t)res;
        }
        else
        {
            output[i] = Q7_MIN;
        }
    }
}
