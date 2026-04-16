/*
 * Copyright (C) 2010-2022 Arm Limited or its affiliates.
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
 * Modifications copyright (C) 2021-2024 Chair of Electronic Design Automation,
 * TUM
 * Modifications copyright (C) 2026 Sohail Raj Satapathy
 */

#include "functions.h"
#include "support_functions.h"
#include "ee_api.h"

#include <stdint.h>

/*
 * Basic s8 convolution function.
 *
 * Optimal use case for the vectorized implementation is when input and output
 * channels are large.
 *
 */

int32_t
nn_convolve_s8(const nn_context                  *ctx,
               const nn_conv_params              *conv_params,
               const nn_per_channel_quant_params *quant_params,
               const nn_dims                     *input_dims,
               const q7_t                        *input_data,
               const nn_dims                     *filter_dims,
               const q7_t                        *filter_data,
               const nn_dims                     *bias_dims,
               const int32_t                     *bias_data,
               const nn_dims                     *output_dims,
               q7_t                              *output_data)
{
    q15_t *buffer_a = (q15_t *)ctx->buf;

    const int32_t  input_batches = input_dims->n;
    const uint16_t input_x       = input_dims->w;
    const uint16_t input_y       = input_dims->h;
    const uint16_t input_ch      = input_dims->c;
    const uint16_t kernel_x      = filter_dims->w;
    const uint16_t kernel_y      = filter_dims->h;
    const uint16_t output_x      = output_dims->w;
    const uint16_t output_y      = output_dims->h;
    const uint16_t output_ch     = output_dims->c;

    const uint16_t pad_x    = conv_params->padding.w;
    const uint16_t pad_y    = conv_params->padding.h;
    const uint16_t stride_x = conv_params->stride.w;
    const uint16_t stride_y = conv_params->stride.h;

    const int32_t  input_offset       = conv_params->input_offset;
    const int32_t  out_offset         = conv_params->output_offset;
    const int32_t  out_activation_min = conv_params->activation.min;
    const int32_t  out_activation_max = conv_params->activation.max;
    const int32_t *output_mult        = quant_params->multiplier;
    const int32_t *output_shift       = quant_params->shift;

    for (int32_t i_batch = 0; i_batch < input_batches; i_batch++)
    {
        const uint16_t dilation_x = conv_params->dilation.w;
        const uint16_t dilation_y = conv_params->dilation.h;

        int32_t i_out_y, i_out_x, i_ker_y, i_ker_x;

        /* Generate two columns from the input tensor a GEMM computation */
        q15_t *two_column_buf = buffer_a;
        q7_t  *out            = output_data;

        /* This part implements the im2col function */
        for (i_out_y = 0; i_out_y < output_y; i_out_y++)
        {
            for (i_out_x = 0; i_out_x < output_x; i_out_x++)
            {
                const int32_t base_idx_y = stride_y * i_out_y - pad_y;
                const int32_t base_idx_x = stride_x * i_out_x - pad_x;

                for (i_ker_y = 0; i_ker_y < kernel_y; i_ker_y++)
                {
                    for (i_ker_x = 0; i_ker_x < kernel_x; i_ker_x++)
                    {
                        const int32_t k_y = base_idx_y + dilation_y * i_ker_y;
                        const int32_t k_x = base_idx_x + dilation_x * i_ker_x;

                        if (k_y < 0 || k_y >= input_y || k_x < 0
                            || k_x >= input_x)
                        {
                            /* Filling 0 for out-of-bound paddings */
                            th_memset((int8_t *)two_column_buf,
                                      0,
                                      sizeof(q15_t) * input_ch);
                        }
                        else
                        {
                            /* Copying the pixel data to column */
                            nn_q7_to_q15_with_offset(
                                input_data + (k_y * input_x + k_x) * input_ch,
                                two_column_buf,
                                input_ch,
                                input_offset);
                        }
                        two_column_buf += input_ch;
                    }
                }

                /* Computation is filed for every 2 columns */
                if (two_column_buf
                    == buffer_a + 2 * input_ch * kernel_y * kernel_x)
                {
                    out = nn_mat_mult_kernel_s8_s16(filter_data,
                                                    buffer_a,
                                                    output_ch,
                                                    output_shift,
                                                    output_mult,
                                                    out_offset,
                                                    out_activation_min,
                                                    out_activation_max,
                                                    input_ch * kernel_y
                                                        * kernel_x,
                                                    bias_data,
                                                    out);

                    /* counter reset */
                    two_column_buf = buffer_a;
                }
            }
        }

        /* left-over because odd number of output pixels */
        if (two_column_buf != buffer_a)
        {
            const q7_t *ker_a = filter_data;
            int32_t     i;

            for (i = 0; i < output_ch; i++)
            {
                /* Load the accumulator with bias first */
                q31_t sum = 0;
                if (bias_data)
                {
                    sum = bias_data[i];
                }

                /* Point to the beginning of the im2col buffer where the input
                 * is available as a rearranged column */
                const q15_t *ip_as_col = buffer_a;

                uint16_t col_count = input_ch * kernel_y * kernel_x;

                while (col_count)
                {
                    q7_t  ker_a1 = *ker_a++;
                    q15_t ip_b1  = *ip_as_col++;
                    sum += ker_a1 * ip_b1;
                    col_count--;
                }

                sum = nn_requantize(sum, output_mult[i], output_shift[i]);
                sum += out_offset;
                sum    = MAX(sum, out_activation_min);
                sum    = MIN(sum, out_activation_max);
                *out++ = (q7_t)sum;
            }
        }

        /* Advance to the next batch */
        input_data += (input_x * input_y * input_ch);
        output_data += (output_x * output_y * output_ch);
    }

    /* Return to application */
    return 0;
}
