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
 * Modifications copyright (C) 2021-2022 Chair of Electronic Design Automation,
 * TUM
 * Modifications copyright (C) 2026 Sohail Raj Satapathy
 */

#include "functions.h"
#include "support_functions.h"

#include <stdint.h>

/*
 * Basic s8 depthwise convolution function.
 */
int32_t
nn_depthwise_conv_s8(const nn_dw_conv_params           *dw_conv_params,
                     const nn_per_channel_quant_params *quant_params,
                     const nn_dims                     *input_dims,
                     const q7_t                        *input,
                     const nn_dims                     *filter_dims,
                     const q7_t                        *kernel,
                     const nn_dims                     *bias_dims,
                     const int32_t                     *bias,
                     const nn_dims                     *output_dims,
                     q7_t                              *output)
{
    const uint16_t dilation_x = dw_conv_params->dilation.w;
    const uint16_t dilation_y = dw_conv_params->dilation.h;

    const uint16_t input_batches         = input_dims->n;
    const uint16_t input_x               = input_dims->w;
    const uint16_t input_y               = input_dims->h;
    const uint16_t input_ch              = input_dims->c;
    const uint16_t ch_mult               = dw_conv_params->ch_mult;
    const uint16_t kernel_x              = filter_dims->w;
    const uint16_t kernel_y              = filter_dims->h;
    const uint16_t pad_x                 = dw_conv_params->padding.w;
    const uint16_t pad_y                 = dw_conv_params->padding.h;
    const uint16_t stride_x              = dw_conv_params->stride.w;
    const uint16_t stride_y              = dw_conv_params->stride.h;
    const int32_t *output_shift          = quant_params->shift;
    const int32_t *output_mult           = quant_params->multiplier;
    const uint16_t output_x              = output_dims->w;
    const uint16_t output_y              = output_dims->h;
    const int32_t  output_offset         = dw_conv_params->output_offset;
    const int32_t  input_offset          = dw_conv_params->input_offset;
    const int32_t  output_activation_min = dw_conv_params->activation.min;
    const int32_t  output_activation_max = dw_conv_params->activation.max;

    int32_t i_out = 0;
    int32_t i_batch;

    for (i_batch = 0; i_batch < input_batches; i_batch++)
    {
        for (int i_out_y = 0; i_out_y < output_y; i_out_y++)
        {
            const int16_t base_idx_y = (i_out_y * stride_y) - pad_y;
            for (int i_out_x = 0; i_out_x < output_x; i_out_x++)
            {
                const int16_t base_idx_x = (i_out_x * stride_x) - pad_x;
                for (int i_input_ch = 0; i_input_ch < input_ch; i_input_ch++)
                {
                    for (int i_ch_mult = 0; i_ch_mult < ch_mult; i_ch_mult++)
                    {
                        const int idx_out_ch = i_ch_mult + i_input_ch * ch_mult;
                        int32_t   acc_0      = 0;

                        int ker_y_start;
                        int ker_x_start;
                        int ker_y_end;
                        int ker_x_end;

                        if (dilation_x > 1)
                        {
                            const int32_t start_x_max
                                = (-base_idx_x + dilation_x - 1) / dilation_x;
                            ker_x_start = MAX(0, start_x_max);
                            const int32_t end_min_x
                                = (input_x - base_idx_x + dilation_x - 1)
                                  / dilation_x;
                            ker_x_end = MIN(kernel_x, end_min_x);
                        }
                        else
                        {
                            ker_x_start = MAX(0, -base_idx_x);
                            ker_x_end   = MIN(kernel_x, input_x - base_idx_x);
                        }

                        if (dilation_y > 1)
                        {
                            const int32_t start_y_max
                                = (-base_idx_y + dilation_y - 1) / dilation_y;
                            ker_y_start = MAX(0, start_y_max);
                            const int32_t end_min_y
                                = (input_y - base_idx_y + dilation_y - 1)
                                  / dilation_y;
                            ker_y_end = MIN(kernel_y, end_min_y);
                        }
                        else
                        {
                            ker_y_start = MAX(0, -base_idx_y);
                            ker_y_end   = MIN(kernel_y, input_y - base_idx_y);
                        }

                        if (bias)
                        {
                            acc_0 = bias[idx_out_ch];
                        }
                        /*
                         * TODO(fabianpedd): We currently
                         * can't expose enough parallelism to the vector
                         * extension using this naive convolution approach.
                         * Thus, the vector extension can only operate on 3-5
                         * values in parallel and is actually slower than the
                         * scalar implementation (due to the vector overhead).
                         * Most kernels are in the range of 3x3 or 5x5. Thus, we
                         * would at least need to expose both kernel dimensions
                         * to the vector extension in order to get any
                         * significant speedup. This, however, is very hard to
                         * achieve here, as we need to take into account both
                         * strides != 1 and dilation != 1
                         * TODO(fabianpedd): I lifted
                         * the index calculations from the inner loops to the
                         * outer loops. This slightly improves performance, as
                         * far as I could measure. However, in reality this
                         * could turn out not to be as efficient since (I
                         * think?!) more multiplications are required. So on HW
                         * with multipliers that need more than a single cycle
                         * this might not perform as well as the reference code
                         * from ARM                        ARMs code for (int
                         * i_ker_y = ker_y_start; i_ker_y < ker_y_end;
                         *                      i_ker_y++)
                         *                      {
                         *                          const int32_t idx_y =
                         * base_idx_y + dilation_y * i_ker_y; for (int i_ker_x =
                         * ker_x_start; i_ker_x < ker_x_end; i_ker_x++)
                         *                          {
                         *                              const int32_t idx_x =
                         * base_idx_x + dilation_x
                         *                              * i_ker_x; int32_t idx_0
                         * = (idx_y * input_x + idx_x) * input_ch + i_input_ch;
                         * int32_t ker_idx_0 = (i_ker_y * kernel_x + i_ker_x) *
                         *                              (input_ch * ch_mult) +
                         * idx_out_ch; acc_0 += (input[idx_0] + input_offset) *
                         *                              kernel[ker_idx_0];
                         *                          }
                         */
                        int8_t *input_idx
                            = (int8_t *)(base_idx_y * input_x * input_ch
                                         + base_idx_x * input_ch + i_input_ch
                                         + input
                                         + dilation_y * input_x * input_ch
                                               * ker_y_start
                                         + dilation_x * input_ch * ker_x_start);
                        int8_t *kernel_idx
                            = (int8_t *)(idx_out_ch + kernel
                                         + kernel_x * input_ch * ch_mult
                                               * ker_y_start
                                         + input_ch * ch_mult * ker_x_start);

                        for (int i_ker_y = ker_y_start; i_ker_y < ker_y_end;
                             i_ker_y++)
                        {
                            int8_t *input_idx_inner  = input_idx;
                            int8_t *kernel_idx_inner = kernel_idx;
                            for (int i_ker_x = ker_x_start; i_ker_x < ker_x_end;
                                 i_ker_x++)
                            {
                                acc_0 += (*input_idx_inner + input_offset)
                                         * *kernel_idx_inner;

                                input_idx_inner += input_ch * dilation_x;
                                kernel_idx_inner += input_ch * ch_mult;
                            }
                            input_idx += dilation_y * input_x * input_ch;
                            kernel_idx += kernel_x * input_ch * ch_mult;
                        }

                        /* Requantize and clamp output to provided range */
                        acc_0 = nn_requantize(acc_0,
                                              output_mult[idx_out_ch],
                                              output_shift[idx_out_ch]);
                        acc_0 += output_offset;
                        acc_0 = MAX(acc_0, output_activation_min);
                        acc_0 = MIN(acc_0, output_activation_max);

                        output[i_out++] = acc_0;
                    }
                }
            }
        }
        /* Advance to the next batch */
        input += (input_x * input_y * input_ch);
    }

    /* Return to application */
    return 0;
}
