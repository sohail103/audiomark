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
#include "ee_api.h"
#include "ee_nn.h"

#include <stdint.h>

/*
 * Specialized pointwise (1x1) convolution for Layers 2, 4, 6, 8:
 *   Input : 1 x 25 x 5 x 64  ->  125 spatial pixels x 64 channels
 *   Filter: 64 x 1 x 1 x 64  ->  [64 out][64 in], q7, row-major
 *   Output: 1 x 25 x 5 x 64
 *   input_offset = 128, output_offset = -128
 */

int32_t
nn_conv1x1_s8(const nn_context                  *ctx,
              const nn_per_channel_quant_params *quant_params,
              const q7_t                        *input_data,
              const q7_t                        *filter_data,
              const int32_t                     *bias_data,
              q7_t                              *output_data)
{
    (void)ctx;

    static const uint16_t output_ch = CONV_2_OUT_CH;                   /* 64 */
    static const uint16_t input_ch  = CONV_2_IN_CH;                    /* 64 */
    static const int32_t  spatial = CONV_2_OUTPUT_H * CONV_2_OUTPUT_W; /* 125 */

    static const int32_t input_offset = CONV_2_INPUT_OFFSET;  /* 128 */
    static const int32_t out_offset   = CONV_2_OUTPUT_OFFSET; /* -128 */
    static const int32_t out_activation_min
        = CONV_2_OUT_ACTIVATION_MIN; /* -128 */
    static const int32_t out_activation_max
        = CONV_2_OUT_ACTIVATION_MAX; /*  127 */

    const int32_t *output_mult  = quant_params->multiplier;
    const int32_t *output_shift = quant_params->shift;

    q15_t buffer[2 * CONV_2_IN_CH];

    const q7_t *inp = input_data;
    q7_t       *out = output_data;
    int32_t     px  = 0;

    for (; px <= spatial - 2; px += 2)
    {
        /* Two adjacent pixels = 2*input_ch = 128 contiguous q7 bytes */
        nn_q7_to_q15_with_offset(inp, buffer, 2 * input_ch, input_offset);

        out = nn_mat_mult_kernel_s8_s16(filter_data,
                                        buffer,
                                        output_ch,
                                        output_shift,
                                        output_mult,
                                        out_offset,
                                        out_activation_min,
                                        out_activation_max,
                                        input_ch,
                                        bias_data,
                                        out);
        inp += 2 * input_ch;
    }

    /* Scalar tail: 125 is odd, always exactly one leftover pixel */
    if (px < spatial)
    {
        nn_q7_to_q15_with_offset(inp, buffer, input_ch, input_offset);

        const q7_t *ker_a = filter_data;
        for (int32_t o = 0; o < output_ch; o++)
        {
            q31_t sum = bias_data ? bias_data[o] : 0;

            const q15_t *col   = buffer;
            uint16_t     count = input_ch;
            while (count--)
            {
                sum += (q31_t)(*ker_a++) * (q31_t)(*col++);
            }

            sum = nn_requantize(sum, output_mult[o], output_shift[o]);
            sum += out_offset;
            sum    = MAX(sum, out_activation_min);
            sum    = MIN(sum, out_activation_max);
            *out++ = (q7_t)sum;
        }
    }

    return 0;
}
