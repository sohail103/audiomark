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
#include "convolve_config.h"

#include <stdint.h>

#define INPUT_X   5
#define INPUT_Y   25
#define INPUT_CH  64
#define OUTPUT_X  5
#define OUTPUT_Y  25
#define OUTPUT_CH 64

#define INPUT_OFFSET (128)
#define OUT_OFFSET   (-128)
#define OUT_ACT_MIN  (-128)
#define OUT_ACT_MAX  127

int32_t
nn_conv1x1_s8(const nn_context                  *ctx,
              const nn_per_channel_quant_params *quant_params,
              const q7_t                        *input_data,
              const q7_t                        *filter_data,
              const int32_t                     *bias_data,
              q7_t                              *output_data)
{
    q15_t *const im2col_buf = (q15_t *)ctx->buf;

    const int32_t *output_mult  = quant_params->multiplier;
    const int32_t *output_shift = quant_params->shift;

    q15_t *const col_buf_full = im2col_buf + (int32_t)NN_KERNEL_COLS * INPUT_CH;

    q15_t *col_buf = im2col_buf;
    q7_t  *out     = output_data;

    // Stride 1, no padding, input/output dims equal: pixel p maps 1:1
    // between input and output, so one flat loop replaces i_out_y/i_out_x.
    for (int32_t p = 0; p < OUTPUT_X * OUTPUT_Y; p++)
    {
        nn_q7_to_q15_with_offset(
            input_data + p * INPUT_CH, col_buf, INPUT_CH, INPUT_OFFSET);
        col_buf += INPUT_CH;

        if (col_buf == col_buf_full)
        {
            out = nn_mat_mult_kernel_s8_s16(filter_data,
                                            im2col_buf,
                                            OUTPUT_CH,
                                            output_shift,
                                            output_mult,
                                            OUT_OFFSET,
                                            OUT_ACT_MIN,
                                            OUT_ACT_MAX,
                                            INPUT_CH,
                                            bias_data,
                                            out);

            col_buf = im2col_buf;
        }
    }

    if (col_buf != im2col_buf)
    {
        const int32_t leftover_pixels = (OUTPUT_X * OUTPUT_Y) % NN_KERNEL_COLS;
        const q15_t  *patch           = im2col_buf;

        for (int32_t p = 0; p < leftover_pixels; p++, patch += INPUT_CH)
        {
            const q7_t *ker_a = filter_data;

            for (int32_t i = 0; i < OUTPUT_CH; i++)
            {
                q31_t        sum       = bias_data ? bias_data[i] : 0;
                const q15_t *col       = patch;
                uint16_t     col_count = INPUT_CH;

                while (col_count--)
                {
                    sum += (*ker_a++) * (*col++);
                }

                sum = nn_requantize(sum, output_mult[i], output_shift[i]);
                sum += OUT_OFFSET;
                sum    = MAX(sum, OUT_ACT_MIN);
                sum    = MIN(sum, OUT_ACT_MAX);
                *out++ = (q7_t)sum;
            }
        }
    }

    input_data += INPUT_X * INPUT_Y * INPUT_CH;
    output_data += OUTPUT_X * OUTPUT_Y * OUTPUT_CH;

    return 0;
}
