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
nn_softmax_s8(const int8_t *input,
              const int32_t num_rows,
              const int32_t row_size,
              const int32_t mult,
              const int32_t shift,
              const int32_t diff_min,
              void         *output)
{
    const int32_t mask = (1 << shift);

    int32_t col = 0;
    int32_t row_idx;

    for (row_idx = 0; row_idx < num_rows; ++row_idx)
    {
        /* Find the maximum value in order to ensure numerical stability */
        int8_t max = *input;

        for (col = 1; col < row_size; ++col)
        {
            max = MAX(max, input[col]);
        }

        int32_t diff = 0;
        int32_t sum  = 0;

        for (col = 0; col < row_size; ++col)
        {
            diff = input[col] - max;
            if (diff >= diff_min)
            {
                sum += DIV_POW2(EXP_ON_NEG(MUL_SAT(diff * mask, mult)),
                                ACCUM_BITS);
            }
        }

        const int32_t headroom = __builtin_clz(sum);
        const int32_t shifted_scale
            = ONE_OVER1((sum > 0 ? sum << headroom : 0) - (1 << 31));
        int32_t bits_over_unit;

        int8_t *output_s8 = (int8_t *)output + row_idx * row_size;

        bits_over_unit = ACCUM_BITS - headroom + 23;

        for (col = 0; col < row_size; ++col)
        {
            diff = input[col] - max;
            if (diff >= diff_min)
            {
                const int32_t res
                    = DIV_POW2(MUL_SAT(shifted_scale,
                                       EXP_ON_NEG(MUL_SAT(diff * mask, mult))),
                               bits_over_unit)
                      + Q7_MIN;
                output_s8[col]
                    = (int8_t)CLAMP(res, (int32_t)Q7_MAX, (int32_t)Q7_MIN);
            }
            else
            {
                output_s8[col] = Q7_MIN;
            }
        }

        input += row_size;
    }
}
