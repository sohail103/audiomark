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

#include "functions.h"
#include "support_functions.h"

#define Q7_MIN     ((q7_t)(0x80))
#define ACCUM_BITS 12

extern const int32_t EXP_LUT[256];
extern const int32_t RECIP_LUT[256];

/*
 * Softmax function with s8 input and output of s8 using lookup tables
 */

void
nn_softmax_row12_s8(const int8_t *__restrict input, int8_t *__restrict output)
{
    /* find max */
    int8_t max = input[0];
    for (uint8_t i = 1; i < 12; i++)
    {
        if (input[i] > max)
        {
            max = input[i];
        }
    }

    /* compute sum of exps */
    int32_t sum = 0;
    int32_t exp_cache[12];

    for (uint8_t i = 0; i < 12; i++)
    {
        /* diff_idx is strictly positive (0 to 255) */
        uint8_t diff_idx = (uint8_t)(max - input[i]);

        int32_t expv = EXP_LUT[diff_idx];

        exp_cache[i] = expv;
        sum += expv >> ACCUM_BITS;
    }

    /* normalization */
    int32_t headroom = __builtin_clz(sum);

    int32_t shifted_sum = (sum > 0) ? (int32_t)((uint32_t)sum << headroom) : 0;

    /* extract top 8 bits (excluding the mandatory MSB at bit 31) for the
     * reciprocal LUT index */
    uint8_t recip_idx     = ((uint32_t)shifted_sum & 0x7FFFFFFF) >> 23;
    int32_t shifted_scale = RECIP_LUT[recip_idx];

    int32_t bits_over_unit = ACCUM_BITS - headroom + 23;

    /* output */
    for (uint8_t i = 0; i < 12; i++)
    {
        int32_t expv = exp_cache[i];

        int32_t res
            = DIV_POW2(MUL_SAT(shifted_scale, expv), bits_over_unit) + Q7_MIN;

        /* catch overflow */
        res = (res > 0x7F) ? 0x7F : res;

        /* mask = 0 if expv == 0, else -1 */
        int32_t mask = -(expv != 0);

        /* select between res and Q7_MIN */
        res = (res & mask) | (Q7_MIN & ~mask);

        output[i] = (int8_t)res;
    }
}
