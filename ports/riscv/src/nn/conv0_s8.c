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
 * Specialized convolution for Layer 0:
 *   Input : 1 x 49 x 10 x 1, Filter: 64 x 10 x 4 x 1
 *   Stride: 2x2, Pad: H=4 W=1, Output: 1 x 25 x 5 x 64
 */

static const uint16_t input_x   = CONV_0_INPUT_W;      /* 10 */
static const uint16_t input_y   = CONV_0_INPUT_H;      /* 49 */
static const uint16_t input_ch  = CONV_0_IN_CH;        /*  1 */
static const uint16_t kernel_x  = CONV_0_FILTER_W;     /*  4 */
static const uint16_t kernel_y  = CONV_0_FILTER_H;     /* 10 */
static const uint16_t output_x  = CONV_0_OUTPUT_W;     /*  5 */
static const uint16_t output_y  = CONV_0_OUTPUT_H;     /* 25 */
static const uint16_t output_ch = CONV_0_OUT_CH;       /* 64 */
static const uint16_t pad_x     = CONV_0_PAD_W;        /*  1 */
static const uint16_t pad_y     = CONV_0_PAD_H;        /*  4 */
static const uint16_t stride_x  = CONV_0_STRIDE_W;     /*  2 */
static const uint16_t stride_y  = CONV_0_STRIDE_H;     /*  2 */
static const int32_t  in_off    = CONV_0_INPUT_OFFSET; /* -100 */

/*
 * Fill one im2col column for an interior pixel (no boundary checks needed).
 * input_ch = 1: each kernel row contributes exactly kernel_x = 4 q15 values
 * from a single contiguous input row.
 */
static inline void
fill_col_interior(q15_t *col, const q7_t *inp_row)
{
    /* 10 rows * 4 cols = 40 q15 values, all in-bounds, no branches */
    for (int32_t ky = 0; ky < kernel_y; ky++, inp_row += input_x)
    {
        col[0] = (q15_t)inp_row[0] + (q15_t)in_off;
        col[1] = (q15_t)inp_row[1] + (q15_t)in_off;
        col[2] = (q15_t)inp_row[2] + (q15_t)in_off;
        col[3] = (q15_t)inp_row[3] + (q15_t)in_off;
        col += kernel_x;
    }
}

/*
 * Fill one im2col column for a boundary pixel.
 * Computes valid kernel row/col ranges once, zero-fills padding in bulk.
 */
static inline void
fill_col_boundary(q15_t      *col,
                  int32_t     base_y,
                  int32_t     base_x,
                  const q7_t *input_data)
{
    const int32_t ky_start = (base_y < 0) ? -base_y : 0;
    const int32_t ky_end
        = (base_y + kernel_y > input_y) ? (int32_t)input_y - base_y : kernel_y;
    const int32_t kx_start = (base_x < 0) ? -base_x : 0;
    const int32_t kx_end
        = (base_x + kernel_x > input_x) ? (int32_t)input_x - base_x : kernel_x;

    /* Zero the padding rows at the top */
    if (ky_start > 0)
    {
        th_memset(
            (int8_t *)col, 0, sizeof(q15_t) * (uint32_t)ky_start * kernel_x);
    }

    for (int32_t ky = ky_start; ky < ky_end; ky++)
    {
        const q7_t *row = input_data + (base_y + ky) * input_x + base_x;
        q15_t      *dst = col + ky * kernel_x;

        /* Left padding */
        for (int32_t kx = 0; kx < kx_start; kx++)
        {
            dst[kx] = 0;
        }
        /* Valid columns */
        for (int32_t kx = kx_start; kx < kx_end; kx++)
        {
            dst[kx] = (q15_t)row[kx] + (q15_t)in_off;
        }
        /* Right padding */
        for (int32_t kx = kx_end; kx < kernel_x; kx++)
        {
            dst[kx] = 0;
        }
    }

    /* Zero the padding rows at the bottom */
    if (ky_end < kernel_y)
    {
        th_memset((int8_t *)(col + ky_end * kernel_x),
                  0,
                  sizeof(q15_t) * (uint32_t)(kernel_y - ky_end) * kernel_x);
    }
}

int32_t
nn_conv0_s8(const nn_context                  *ctx,
            const nn_per_channel_quant_params *quant_params,
            const q7_t                        *input_data,
            const q7_t                        *filter_data,
            const int32_t                     *bias_data,
            q7_t                              *output_data)
{
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
        const int32_t base_idx_y = stride_y * i_out_y - pad_y;
        /*
         * Interior y: i_out_y in [2, 21]
         * base_idx_y in [0, 38], kernel rows [base_idx_y, base_idx_y+9]
         * all within [0, 48] = [0, input_y-1].
         */
        const int y_interior = ((uint32_t)(i_out_y - 2) <= 19u);

        for (int32_t i_out_x = 0; i_out_x < output_x; i_out_x++)
        {
            const int32_t base_idx_x = stride_x * i_out_x - pad_x;
            /*
             * Interior x: i_out_x in [1, 3]
             * base_idx_x in [1, 5], kernel cols [base_idx_x, base_idx_x+3]
             * all within [0, 9] = [0, input_x-1].
             */
            const int x_interior = ((uint32_t)(i_out_x - 1) <= 2u);

            if (y_interior & x_interior)
            {
                fill_col_interior(two_column_buf,
                                  input_data + base_idx_y * input_x
                                      + base_idx_x);
            }
            else
            {
                fill_col_boundary(
                    two_column_buf, base_idx_y, base_idx_x, input_data);
            }
            two_column_buf += input_ch * kernel_y * kernel_x;

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

    /* Left-over odd output pixel (25*5 = 125 is odd, always fires once) */
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
