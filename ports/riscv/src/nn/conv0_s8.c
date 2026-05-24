/*
 * Copyright (C) 2010-2022 Arm Limited or its affiliates.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Modifications copyright (C) 2021-2024 Chair of Electronic Design Automation,
 * TUM Modifications copyright (C) 2026 Sohail Raj Satapathy
 */

#include "functions.h"
#include "support_functions.h"
#include "ee_api.h"
#include "ee_nn.h"

#include <stdint.h>

/*
 * Specialized convolution for Layer 0:
 *   Input : 1 x 49 x 10 x 1, Filter: 64 x 10 x 4 x 1
 *   Stride: 2x2, Pad: H=4 W=1, Output: 1 x 25 x 5 x 64
 */
int32_t
nn_conv0_s8(const nn_context                  *ctx,
            const nn_per_channel_quant_params *quant_params,
            const q7_t                        *input_data,
            const q7_t                        *filter_data,
            const int32_t                     *bias_data,
            q7_t                              *output_data)
{
    static const uint16_t input_x   = CONV_0_INPUT_W;
    static const uint16_t input_y   = CONV_0_INPUT_H;
    static const uint16_t input_ch  = CONV_0_IN_CH;
    static const uint16_t kernel_x  = CONV_0_FILTER_W;
    static const uint16_t kernel_y  = CONV_0_FILTER_H;
    static const uint16_t output_x  = CONV_0_OUTPUT_W;
    static const uint16_t output_y  = CONV_0_OUTPUT_H;
    static const uint16_t output_ch = CONV_0_OUT_CH;

    static const uint16_t pad_x      = CONV_0_PAD_W;
    static const uint16_t pad_y      = CONV_0_PAD_H;
    static const uint16_t stride_x   = CONV_0_STRIDE_W;
    static const uint16_t stride_y   = CONV_0_STRIDE_H;
    static const uint16_t dilation_x = CONV_0_DILATION_W;
    static const uint16_t dilation_y = CONV_0_DILATION_H;

    static const int32_t input_offset       = CONV_0_INPUT_OFFSET;
    static const int32_t out_offset         = CONV_0_OUTPUT_OFFSET;
    static const int32_t out_activation_min = CONV_0_OUT_ACTIVATION_MIN;
    static const int32_t out_activation_max = CONV_0_OUT_ACTIVATION_MAX;

    const int32_t *output_mult  = quant_params->multiplier;
    const int32_t *output_shift = quant_params->shift;

    q15_t *buffer_a       = (q15_t *)ctx->buf;
    q15_t *two_column_buf = buffer_a;
    q7_t  *out            = output_data;

    for (int32_t i_out_y = 0; i_out_y < output_y; i_out_y++)
    {
        for (int32_t i_out_x = 0; i_out_x < output_x; i_out_x++)
        {
            const int32_t base_idx_y = stride_y * i_out_y - pad_y;
            const int32_t base_idx_x = stride_x * i_out_x - pad_x;

            for (int32_t i_ker_y = 0; i_ker_y < kernel_y; i_ker_y++)
            {
                for (int32_t i_ker_x = 0; i_ker_x < kernel_x; i_ker_x++)
                {
                    const int32_t k_y = base_idx_y + dilation_y * i_ker_y;
                    const int32_t k_x = base_idx_x + dilation_x * i_ker_x;

                    if (k_y < 0 || k_y >= input_y || k_x < 0 || k_x >= input_x)
                    {
                        th_memset((int8_t *)two_column_buf,
                                  0,
                                  sizeof(q15_t) * input_ch);
                    }
                    else
                    {
                        nn_q7_to_q15_with_offset(
                            input_data + (k_y * input_x + k_x) * input_ch,
                            two_column_buf,
                            input_ch,
                            input_offset);
                    }
                    two_column_buf += input_ch;
                }
            }

            if (two_column_buf == buffer_a + 2 * input_ch * kernel_y * kernel_x)
            {
                out = nn_mat_mult_kernel_s8_s16(filter_data,
                                                buffer_a,
                                                output_ch,
                                                output_shift,
                                                output_mult,
                                                out_offset,
                                                out_activation_min,
                                                out_activation_max,
                                                input_ch * kernel_y * kernel_x,
                                                bias_data,
                                                out);
                two_column_buf = buffer_a;
            }
        }
    }

    /* Left-over odd output pixel (25*5 = 125 is odd, so always fires) */
    if (two_column_buf != buffer_a)
    {
        const q7_t *ker_a = filter_data;

        for (int32_t i = 0; i < output_ch; i++)
        {
            q31_t sum = bias_data ? bias_data[i] : 0;

            const q15_t *ip_as_col = buffer_a;
            uint16_t     col_count = input_ch * kernel_y * kernel_x;

            while (col_count--)
            {
                sum += (*ker_a++) * (*ip_as_col++);
            }

            sum = nn_requantize(sum, output_mult[i], output_shift[i]);
            sum += out_offset;
            sum    = MAX(sum, out_activation_min);
            sum    = MIN(sum, out_activation_max);
            *out++ = (q7_t)sum;
        }
    }

    return 0;
}