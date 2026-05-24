/*
 * Copyright (C) 2010-2022 Arm Limited or its affiliates.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Modifications copyright (C) 2021-2024 Chair of Electronic Design Automation,
 * TUM
 * Modifications copyright (C) 2026 Sohail Raj Satapathy
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

    /*
     * 2-column im2col buffer: 2 * input_ch * sizeof(q15_t) = 256 bytes.
     * Stack allocation is fine at this size.
     */
    q15_t buffer[2 * CONV_2_IN_CH];

    const q7_t *inp = input_data;
    q7_t       *out = output_data;

    int32_t px = 0;

    /* Main loop: 2 spatial pixels per call to mat_mult_kernel_s8_s16 */
    for (; px <= spatial - 2; px += 2)
    {
        nn_q7_to_q15_with_offset(inp, buffer, input_ch, input_offset);
        nn_q7_to_q15_with_offset(
            inp + input_ch, buffer + input_ch, input_ch, input_offset);

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

    /* Scalar tail for the leftover pixel (125 is odd, always exactly 1) */
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
