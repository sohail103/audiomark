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
 * Modifications copyright (C) 2021-2023 Chair of Electronic Design Automation,
 * TUM
 * Modifications copyright (C) 2026 Sohail Raj Satapathy
 */

#include "functions.h"
#include "support_functions.h"
#include "convolve_config.h"

#include <stddef.h>

static void
nn_fold_input_offset_s8(int32_t       *corrected_bias,
                        const int32_t *bias_data,
                        const q7_t    *filter_data,
                        int32_t        input_ch,
                        int32_t        output_ch,
                        int32_t        input_offset)
{
    for (int32_t oc = 0; oc < output_ch; oc++)
    {
        const q7_t *w   = filter_data + (size_t)oc * input_ch;
        q31_t       sum = 0;

        for (int32_t k = 0; k < input_ch; k++)
        {
            sum += w[k];
        }

        corrected_bias[oc]
            = (bias_data ? bias_data[oc] : 0) + sum * input_offset;
    }
}

int32_t
nn_conv1x1_s8(const nn_context                  *ctx,
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
    const int32_t  col_len    = input_dims->w * input_dims->h * input_dims->n;
    const int32_t  output_ch  = output_dims->c;
    const int32_t  input_ch   = input_dims->c;
    const int32_t  out_offset = conv_params->output_offset;
    const int32_t  act_min    = conv_params->activation.min;
    const int32_t  act_max    = conv_params->activation.max;
    const int32_t *out_mult   = quant_params->multiplier;
    const int32_t *out_shift  = quant_params->shift;

    int32_t *corrected_bias = (int32_t *)ctx->buf;

    nn_fold_input_offset_s8(corrected_bias,
                            bias_data,
                            filter_data,
                            input_ch,
                            output_ch,
                            conv_params->input_offset);

    int32_t i_items = 0;
    for (; i_items <= col_len - NN_KERNEL_COLS; i_items += NN_KERNEL_COLS)
    {
        output_data
            = nn_mat_mult_kernel_s8_s8(filter_data,
                                       input_data + (size_t)i_items * input_ch,
                                       output_ch,
                                       out_shift,
                                       out_mult,
                                       out_offset,
                                       act_min,
                                       act_max,
                                       input_ch,
                                       corrected_bias,
                                       output_data);
    }

    for (; i_items < col_len; i_items++)
    {
        output_data
            = nn_mat_mult_core_1x1_s8(filter_data,
                                      input_data + (size_t)i_items * input_ch,
                                      output_ch,
                                      out_shift,
                                      out_mult,
                                      out_offset,
                                      act_min,
                                      act_max,
                                      input_ch,
                                      corrected_bias,
                                      output_data);
    }

    return 0;
}
