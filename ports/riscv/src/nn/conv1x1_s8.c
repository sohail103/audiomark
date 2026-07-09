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

#define INPUT_W     5
#define INPUT_H     25
#define INPUT_N     1
#define INPUT_CH    64
#define OUTPUT_CH   64
#define OUT_OFFSET  (-128)
#define OUT_ACT_MIN (-128)
#define OUT_ACT_MAX 127
#define COL_LEN     (INPUT_W * INPUT_H * INPUT_N)

static void
nn_fold_input_offset_s8(int32_t       *corrected_bias,
                        const int32_t *bias_data,
                        const q7_t    *filter_data,
                        int32_t        input_offset)
{
    for (int32_t oc = 0; oc < OUTPUT_CH; oc++)
    {
        const q7_t *w   = filter_data + (size_t)oc * INPUT_CH;
        q31_t       sum = 0;

        for (int32_t k = 0; k < INPUT_CH; k++)
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
              const q7_t                        *input_data,
              const q7_t                        *filter_data,
              const int32_t                     *bias_data,
              q7_t                              *output_data)
{
    const int32_t *out_mult  = quant_params->multiplier;
    const int32_t *out_shift = quant_params->shift;

    int32_t *corrected_bias = (int32_t *)ctx->buf;

    nn_fold_input_offset_s8(
        corrected_bias, bias_data, filter_data, conv_params->input_offset);

    int32_t i_items = 0;
    for (; i_items <= COL_LEN - NN_KERNEL_COLS; i_items += NN_KERNEL_COLS)
    {
        output_data
            = nn_mat_mult_kernel_s8_s8(filter_data,
                                       input_data + (size_t)i_items * INPUT_CH,
                                       OUTPUT_CH,
                                       out_shift,
                                       out_mult,
                                       OUT_OFFSET,
                                       OUT_ACT_MIN,
                                       OUT_ACT_MAX,
                                       INPUT_CH,
                                       corrected_bias,
                                       output_data);
    }

    for (; i_items < COL_LEN; i_items++)
    {
        output_data
            = nn_mat_mult_core_1x1_s8(filter_data,
                                      input_data + (size_t)i_items * INPUT_CH,
                                      OUTPUT_CH,
                                      out_shift,
                                      out_mult,
                                      OUT_OFFSET,
                                      OUT_ACT_MIN,
                                      OUT_ACT_MAX,
                                      INPUT_CH,
                                      corrected_bias,
                                      output_data);
    }

    return 0;
}
