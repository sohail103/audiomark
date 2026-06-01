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

#include <stdint.h>
#include <stdbool.h>

/* Accumulate one kernel column over valid rows [kh_start, kh_end), 4 channels.
   ip/kp point to kernel row 0 of this column; row r lives at r * row_stride. */
static inline void
acc_col_4ch(int32_t      *b0,
            int32_t      *b1,
            int32_t      *b2,
            int32_t      *b3,
            const int8_t *ip,
            const int8_t *kp,
            int32_t       kh_start,
            int32_t       kh_end,
            int32_t       inp_row_stride,
            int32_t       ker_row_stride,
            int32_t       input_offset)
{
    int32_t start = kh_start < 0 ? 0 : kh_start;
    int32_t end   = kh_end > 3 ? 3 : kh_end;
    if (start >= end)
    {
        return;
    }
    switch (start)
    {
        case 0: {
            const int8_t *in  = ip;
            const int8_t *ker = kp;
            *b0 += ((int32_t)in[0] + input_offset) * (int32_t)ker[0];
            *b1 += ((int32_t)in[1] + input_offset) * (int32_t)ker[1];
            *b2 += ((int32_t)in[2] + input_offset) * (int32_t)ker[2];
            *b3 += ((int32_t)in[3] + input_offset) * (int32_t)ker[3];
            if (end == 1)
            {
                break;
            }
        }
        /* fall through */
        case 1: {
            const int8_t *in  = ip + inp_row_stride;
            const int8_t *ker = kp + ker_row_stride;
            *b0 += ((int32_t)in[0] + input_offset) * (int32_t)ker[0];
            *b1 += ((int32_t)in[1] + input_offset) * (int32_t)ker[1];
            *b2 += ((int32_t)in[2] + input_offset) * (int32_t)ker[2];
            *b3 += ((int32_t)in[3] + input_offset) * (int32_t)ker[3];
            if (end == 2)
            {
                break;
            }
        }
        /* fall through */
        case 2: {
            const int8_t *in  = ip + 2 * inp_row_stride;
            const int8_t *ker = kp + 2 * ker_row_stride;
            *b0 += ((int32_t)in[0] + input_offset) * (int32_t)ker[0];
            *b1 += ((int32_t)in[1] + input_offset) * (int32_t)ker[1];
            *b2 += ((int32_t)in[2] + input_offset) * (int32_t)ker[2];
            *b3 += ((int32_t)in[3] + input_offset) * (int32_t)ker[3];
            break;
        }
        default:
            break;
    }
}

static inline void
acc_col_1ch(int32_t      *b0,
            const int8_t *ip,
            const int8_t *kp,
            int32_t       kh_start,
            int32_t       kh_end,
            int32_t       inp_row_stride,
            int32_t       ker_row_stride,
            int32_t       input_offset)
{
    int32_t start = kh_start < 0 ? 0 : kh_start;
    int32_t end   = kh_end > 3 ? 3 : kh_end;
    if (start >= end)
    {
        return;
    }
    switch (start)
    {
        case 0: {
            *b0 += ((int32_t)ip[0] + input_offset) * (int32_t)kp[0];
            if (end == 1)
            {
                break;
            }
        }
        /* fall through */
        case 1: {
            const int8_t *in  = ip + inp_row_stride;
            const int8_t *ker = kp + ker_row_stride;
            *b0 += ((int32_t)in[0] + input_offset) * (int32_t)ker[0];
            if (end == 2)
            {
                break;
            }
        }
        /* fall through */
        case 2: {
            const int8_t *in  = ip + 2 * inp_row_stride;
            const int8_t *ker = kp + 2 * ker_row_stride;
            *b0 += ((int32_t)in[0] + input_offset) * (int32_t)ker[0];
            break;
        }
        default:
            break;
    }
}

int32_t
nn_depthwise_conv_3x3_s8(const nn_dw_conv_params           *dw_conv_params,
                         const nn_per_channel_quant_params *quant_params,
                         const nn_dims                     *input_dims,
                         const q7_t                        *input,
                         const q7_t                        *kernel,
                         const int32_t                     *bias,
                         const nn_dims                     *output_dims,
                         q7_t                              *output)
{
    const int32_t input_x  = input_dims->w;
    const int32_t input_y  = input_dims->h;
    const int32_t input_ch = input_dims->c;
    const int32_t pad_x    = dw_conv_params->padding.w;
    const int32_t pad_y    = dw_conv_params->padding.h;
    const int32_t stride_x = dw_conv_params->stride.w;
    const int32_t stride_y = dw_conv_params->stride.h;
    const int32_t output_x = output_dims->w;
    const int32_t output_y = output_dims->h;

    const int32_t  input_offset          = dw_conv_params->input_offset;
    const int32_t  output_offset         = dw_conv_params->output_offset;
    const int32_t  output_activation_min = dw_conv_params->activation.min;
    const int32_t  output_activation_max = dw_conv_params->activation.max;
    const int32_t *output_mult           = quant_params->multiplier;
    const int32_t *output_shift          = quant_params->shift;

    const int32_t inp_row_stride = input_ch * input_x;
    const int32_t ker_row_stride = input_ch * 3;
    const int32_t col_stride     = input_ch;

    /* Step size for advancing ip0 by one output column in the interior loop. */
    const int32_t input_x_step = stride_x * col_stride;

    /*
     * Compute interior rectangle boundaries directly by walking output
     * coordinates, avoiding closed-form rounding errors entirely.
     *
     * int_h0: first out_h where in_h = out_h*stride_y - pad_y >= 0
     * int_h1: first out_h where in_h + 2 >= input_y  (bottom tap out of bounds)
     * int_w0: first out_w where in_w = out_w*stride_x - pad_x >= 0
     * int_w1: first out_w where in_w + 2 >= input_x  (right tap out of bounds)
     */
    int32_t int_h0 = 0;
    while (int_h0 < output_y && (int_h0 * stride_y - pad_y) < 0)
    {
        ++int_h0;
    }

    int32_t int_h1 = int_h0;
    while (int_h1 < output_y && (int_h1 * stride_y - pad_y + 2) < input_y)
    {
        ++int_h1;
    }

    int32_t int_w0 = 0;
    while (int_w0 < output_x && (int_w0 * stride_x - pad_x) < 0)
    {
        ++int_w0;
    }

    int32_t int_w1 = int_w0;
    while (int_w1 < output_x && (int_w1 * stride_x - pad_x + 2) < input_x)
    {
        ++int_w1;
    }

    int32_t out_idx = 0;

    /* Top border rows */
    for (int32_t out_h = 0; out_h < int_h0; ++out_h)
    {
        const int32_t in_h     = out_h * stride_y - pad_y;
        const int32_t kh_start = -in_h;
        const int32_t kh_end   = MIN(3, input_y - in_h);

        for (int32_t out_w = 0; out_w < output_x; ++out_w)
        {
            const int32_t in_w     = out_w * stride_x - pad_x;
            const int32_t kw_start = (in_w < 0) ? -in_w : 0;
            const bool    right_ok = (in_w + 2) < input_x;
            const int8_t *inp_base
                = input + in_h * inp_row_stride + in_w * col_stride;

            int32_t ch = 0;
            for (; ch <= (input_ch - 4); ch += 4)
            {
                const int8_t *ip = inp_base + ch;
                const int8_t *kp = kernel + ch;
                int32_t       b0 = bias[ch + 0], b1 = bias[ch + 1];
                int32_t       b2 = bias[ch + 2], b3 = bias[ch + 3];

                if (kw_start == 0)
                {
                    acc_col_4ch(&b0,
                                &b1,
                                &b2,
                                &b3,
                                ip,
                                kp,
                                kh_start,
                                kh_end,
                                inp_row_stride,
                                ker_row_stride,
                                input_offset);
                }
                acc_col_4ch(&b0,
                            &b1,
                            &b2,
                            &b3,
                            ip + col_stride,
                            kp + col_stride,
                            kh_start,
                            kh_end,
                            inp_row_stride,
                            ker_row_stride,
                            input_offset);
                if (right_ok)
                {
                    acc_col_4ch(&b0,
                                &b1,
                                &b2,
                                &b3,
                                ip + 2 * col_stride,
                                kp + 2 * col_stride,
                                kh_start,
                                kh_end,
                                inp_row_stride,
                                ker_row_stride,
                                input_offset);
                }

                b0                = nn_requantize(
                                        b0, output_mult[ch + 0], output_shift[ch + 0])
                                    + output_offset;
                b1                = nn_requantize(
                                        b1, output_mult[ch + 1], output_shift[ch + 1])
                                    + output_offset;
                b2                = nn_requantize(
                                        b2, output_mult[ch + 2], output_shift[ch + 2])
                                    + output_offset;
                b3                = nn_requantize(
                                        b3, output_mult[ch + 3], output_shift[ch + 3])
                                    + output_offset;
                output[out_idx++] = (int8_t)MIN(MAX(b0, output_activation_min),
                                                output_activation_max);
                output[out_idx++] = (int8_t)MIN(MAX(b1, output_activation_min),
                                                output_activation_max);
                output[out_idx++] = (int8_t)MIN(MAX(b2, output_activation_min),
                                                output_activation_max);
                output[out_idx++] = (int8_t)MIN(MAX(b3, output_activation_min),
                                                output_activation_max);
            }
            for (; ch < input_ch; ++ch)
            {
                const int8_t *ip = inp_base + ch;
                const int8_t *kp = kernel + ch;
                int32_t       b0 = bias[ch];

                if (kw_start == 0)
                {
                    acc_col_1ch(&b0,
                                ip,
                                kp,
                                kh_start,
                                kh_end,
                                inp_row_stride,
                                ker_row_stride,
                                input_offset);
                }
                acc_col_1ch(&b0,
                            ip + col_stride,
                            kp + col_stride,
                            kh_start,
                            kh_end,
                            inp_row_stride,
                            ker_row_stride,
                            input_offset);
                if (right_ok)
                {
                    acc_col_1ch(&b0,
                                ip + 2 * col_stride,
                                kp + 2 * col_stride,
                                kh_start,
                                kh_end,
                                inp_row_stride,
                                ker_row_stride,
                                input_offset);
                }

                b0 = nn_requantize(b0, output_mult[ch], output_shift[ch])
                     + output_offset;
                output[out_idx++] = (int8_t)MIN(MAX(b0, output_activation_min),
                                                output_activation_max);
            }
        }
    }

    /* Interior rows */
    for (int32_t out_h = int_h0; out_h < int_h1; ++out_h)
    {
        const int32_t in_h = out_h * stride_y - pad_y;

        /* Left border columns: all rows valid, left column clipped */
        for (int32_t out_w = 0; out_w < int_w0; ++out_w)
        {
            const int32_t in_w     = out_w * stride_x - pad_x;
            const int32_t kw_start = -in_w;
            const bool    right_ok = (in_w + 2) < input_x;
            const int8_t *inp_base
                = input + in_h * inp_row_stride + in_w * col_stride;

            int32_t ch = 0;
            for (; ch <= (input_ch - 4); ch += 4)
            {
                const int8_t *ip = inp_base + ch;
                const int8_t *kp = kernel + ch;
                int32_t       b0 = bias[ch + 0], b1 = bias[ch + 1];
                int32_t       b2 = bias[ch + 2], b3 = bias[ch + 3];

                if (kw_start == 0)
                {
                    acc_col_4ch(&b0,
                                &b1,
                                &b2,
                                &b3,
                                ip,
                                kp,
                                0,
                                3,
                                inp_row_stride,
                                ker_row_stride,
                                input_offset);
                }
                acc_col_4ch(&b0,
                            &b1,
                            &b2,
                            &b3,
                            ip + col_stride,
                            kp + col_stride,
                            0,
                            3,
                            inp_row_stride,
                            ker_row_stride,
                            input_offset);
                if (right_ok)
                {
                    acc_col_4ch(&b0,
                                &b1,
                                &b2,
                                &b3,
                                ip + 2 * col_stride,
                                kp + 2 * col_stride,
                                0,
                                3,
                                inp_row_stride,
                                ker_row_stride,
                                input_offset);
                }

                b0                = nn_requantize(
                                        b0, output_mult[ch + 0], output_shift[ch + 0])
                                    + output_offset;
                b1                = nn_requantize(
                                        b1, output_mult[ch + 1], output_shift[ch + 1])
                                    + output_offset;
                b2                = nn_requantize(
                                        b2, output_mult[ch + 2], output_shift[ch + 2])
                                    + output_offset;
                b3                = nn_requantize(
                                        b3, output_mult[ch + 3], output_shift[ch + 3])
                                    + output_offset;
                output[out_idx++] = (int8_t)MIN(MAX(b0, output_activation_min),
                                                output_activation_max);
                output[out_idx++] = (int8_t)MIN(MAX(b1, output_activation_min),
                                                output_activation_max);
                output[out_idx++] = (int8_t)MIN(MAX(b2, output_activation_min),
                                                output_activation_max);
                output[out_idx++] = (int8_t)MIN(MAX(b3, output_activation_min),
                                                output_activation_max);
            }
            for (; ch < input_ch; ++ch)
            {
                const int8_t *ip = inp_base + ch;
                const int8_t *kp = kernel + ch;
                int32_t       b0 = bias[ch];

                if (kw_start == 0)
                {
                    acc_col_1ch(&b0,
                                ip,
                                kp,
                                0,
                                3,
                                inp_row_stride,
                                ker_row_stride,
                                input_offset);
                }
                acc_col_1ch(&b0,
                            ip + col_stride,
                            kp + col_stride,
                            0,
                            3,
                            inp_row_stride,
                            ker_row_stride,
                            input_offset);
                if (right_ok)
                {
                    acc_col_1ch(&b0,
                                ip + 2 * col_stride,
                                kp + 2 * col_stride,
                                0,
                                3,
                                inp_row_stride,
                                ker_row_stride,
                                input_offset);
                }

                b0 = nn_requantize(b0, output_mult[ch], output_shift[ch])
                     + output_offset;
                output[out_idx++] = (int8_t)MIN(MAX(b0, output_activation_min),
                                                output_activation_max);
            }
        }

        /* Interior columns: all 9 taps valid, fully unrolled.
         * ip0 is initialised to the first interior column and then stepped by
         * input_x_step each iteration, avoiding the multiply inside the loop.
         */
        const int8_t *ip0 = input + in_h * inp_row_stride
                            + int_w0 * stride_x * col_stride
                            - pad_x * col_stride;
        for (int32_t out_w = int_w0; out_w < int_w1;
             ++out_w, ip0 += input_x_step)
        {
            int32_t ch = 0;
            for (; ch <= (input_ch - 4); ch += 4)
            {
                const int8_t *ip = ip0 + ch;
                const int8_t *kp = kernel + ch;

                /* Row and column pointers so each load is base + small
                 * immediate. This maps directly onto RISC-V register+immediate
                 * load instructions and avoids repeated address
                 * rematerialization. */
                const int8_t *in0 = ip;
                const int8_t *in1 = ip + inp_row_stride;
                const int8_t *in2 = ip + 2 * inp_row_stride;
                const int8_t *k0  = kp;
                const int8_t *k1  = kp + ker_row_stride;
                const int8_t *k2  = kp + 2 * ker_row_stride;

                const int8_t *in00 = in0;
                const int8_t *in01 = in0 + col_stride;
                const int8_t *in02 = in0 + 2 * col_stride;
                const int8_t *in10 = in1;
                const int8_t *in11 = in1 + col_stride;
                const int8_t *in12 = in1 + 2 * col_stride;
                const int8_t *in20 = in2;
                const int8_t *in21 = in2 + col_stride;
                const int8_t *in22 = in2 + 2 * col_stride;
                const int8_t *k00  = k0;
                const int8_t *k01  = k0 + col_stride;
                const int8_t *k02  = k0 + 2 * col_stride;
                const int8_t *k10  = k1;
                const int8_t *k11  = k1 + col_stride;
                const int8_t *k12  = k1 + 2 * col_stride;
                const int8_t *k20  = k2;
                const int8_t *k21  = k2 + col_stride;
                const int8_t *k22  = k2 + 2 * col_stride;

                int32_t b0 = bias[ch + 0], b1 = bias[ch + 1];
                int32_t b2 = bias[ch + 2], b3 = bias[ch + 3];

                b0 += ((int32_t)in00[0] + input_offset) * (int32_t)k00[0];
                b1 += ((int32_t)in00[1] + input_offset) * (int32_t)k00[1];
                b2 += ((int32_t)in00[2] + input_offset) * (int32_t)k00[2];
                b3 += ((int32_t)in00[3] + input_offset) * (int32_t)k00[3];

                b0 += ((int32_t)in01[0] + input_offset) * (int32_t)k01[0];
                b1 += ((int32_t)in01[1] + input_offset) * (int32_t)k01[1];
                b2 += ((int32_t)in01[2] + input_offset) * (int32_t)k01[2];
                b3 += ((int32_t)in01[3] + input_offset) * (int32_t)k01[3];

                b0 += ((int32_t)in02[0] + input_offset) * (int32_t)k02[0];
                b1 += ((int32_t)in02[1] + input_offset) * (int32_t)k02[1];
                b2 += ((int32_t)in02[2] + input_offset) * (int32_t)k02[2];
                b3 += ((int32_t)in02[3] + input_offset) * (int32_t)k02[3];

                b0 += ((int32_t)in10[0] + input_offset) * (int32_t)k10[0];
                b1 += ((int32_t)in10[1] + input_offset) * (int32_t)k10[1];
                b2 += ((int32_t)in10[2] + input_offset) * (int32_t)k10[2];
                b3 += ((int32_t)in10[3] + input_offset) * (int32_t)k10[3];

                b0 += ((int32_t)in11[0] + input_offset) * (int32_t)k11[0];
                b1 += ((int32_t)in11[1] + input_offset) * (int32_t)k11[1];
                b2 += ((int32_t)in11[2] + input_offset) * (int32_t)k11[2];
                b3 += ((int32_t)in11[3] + input_offset) * (int32_t)k11[3];

                b0 += ((int32_t)in12[0] + input_offset) * (int32_t)k12[0];
                b1 += ((int32_t)in12[1] + input_offset) * (int32_t)k12[1];
                b2 += ((int32_t)in12[2] + input_offset) * (int32_t)k12[2];
                b3 += ((int32_t)in12[3] + input_offset) * (int32_t)k12[3];

                b0 += ((int32_t)in20[0] + input_offset) * (int32_t)k20[0];
                b1 += ((int32_t)in20[1] + input_offset) * (int32_t)k20[1];
                b2 += ((int32_t)in20[2] + input_offset) * (int32_t)k20[2];
                b3 += ((int32_t)in20[3] + input_offset) * (int32_t)k20[3];

                b0 += ((int32_t)in21[0] + input_offset) * (int32_t)k21[0];
                b1 += ((int32_t)in21[1] + input_offset) * (int32_t)k21[1];
                b2 += ((int32_t)in21[2] + input_offset) * (int32_t)k21[2];
                b3 += ((int32_t)in21[3] + input_offset) * (int32_t)k21[3];

                b0 += ((int32_t)in22[0] + input_offset) * (int32_t)k22[0];
                b1 += ((int32_t)in22[1] + input_offset) * (int32_t)k22[1];
                b2 += ((int32_t)in22[2] + input_offset) * (int32_t)k22[2];
                b3 += ((int32_t)in22[3] + input_offset) * (int32_t)k22[3];

                b0                = nn_requantize(
                                        b0, output_mult[ch + 0], output_shift[ch + 0])
                                    + output_offset;
                b1                = nn_requantize(
                                        b1, output_mult[ch + 1], output_shift[ch + 1])
                                    + output_offset;
                b2                = nn_requantize(
                                        b2, output_mult[ch + 2], output_shift[ch + 2])
                                    + output_offset;
                b3                = nn_requantize(
                                        b3, output_mult[ch + 3], output_shift[ch + 3])
                                    + output_offset;
                output[out_idx++] = (int8_t)MIN(MAX(b0, output_activation_min),
                                                output_activation_max);
                output[out_idx++] = (int8_t)MIN(MAX(b1, output_activation_min),
                                                output_activation_max);
                output[out_idx++] = (int8_t)MIN(MAX(b2, output_activation_min),
                                                output_activation_max);
                output[out_idx++] = (int8_t)MIN(MAX(b3, output_activation_min),
                                                output_activation_max);
            }
            for (; ch < input_ch; ++ch)
            {
                const int8_t *ip = ip0 + ch;
                const int8_t *kp = kernel + ch;

                const int8_t *in0 = ip;
                const int8_t *in1 = ip + inp_row_stride;
                const int8_t *in2 = ip + 2 * inp_row_stride;
                const int8_t *k0  = kp;
                const int8_t *k1  = kp + ker_row_stride;
                const int8_t *k2  = kp + 2 * ker_row_stride;

                int32_t b0 = bias[ch];

                b0 += ((int32_t)in0[0 * col_stride] + input_offset)
                      * (int32_t)k0[0 * col_stride];
                b0 += ((int32_t)in0[1 * col_stride] + input_offset)
                      * (int32_t)k0[1 * col_stride];
                b0 += ((int32_t)in0[2 * col_stride] + input_offset)
                      * (int32_t)k0[2 * col_stride];
                b0 += ((int32_t)in1[0 * col_stride] + input_offset)
                      * (int32_t)k1[0 * col_stride];
                b0 += ((int32_t)in1[1 * col_stride] + input_offset)
                      * (int32_t)k1[1 * col_stride];
                b0 += ((int32_t)in1[2 * col_stride] + input_offset)
                      * (int32_t)k1[2 * col_stride];
                b0 += ((int32_t)in2[0 * col_stride] + input_offset)
                      * (int32_t)k2[0 * col_stride];
                b0 += ((int32_t)in2[1 * col_stride] + input_offset)
                      * (int32_t)k2[1 * col_stride];
                b0 += ((int32_t)in2[2 * col_stride] + input_offset)
                      * (int32_t)k2[2 * col_stride];

                b0 = nn_requantize(b0, output_mult[ch], output_shift[ch])
                     + output_offset;
                output[out_idx++] = (int8_t)MIN(MAX(b0, output_activation_min),
                                                output_activation_max);
            }
        }

        /* Right border columns: all rows valid, right column clipped */
        for (int32_t out_w = int_w1; out_w < output_x; ++out_w)
        {
            const int32_t in_w     = out_w * stride_x - pad_x;
            const bool    right_ok = (in_w + 2) < input_x;
            const int8_t *inp_base
                = input + in_h * inp_row_stride + in_w * col_stride;

            int32_t ch = 0;
            for (; ch <= (input_ch - 4); ch += 4)
            {
                const int8_t *ip = inp_base + ch;
                const int8_t *kp = kernel + ch;
                int32_t       b0 = bias[ch + 0], b1 = bias[ch + 1];
                int32_t       b2 = bias[ch + 2], b3 = bias[ch + 3];

                acc_col_4ch(&b0,
                            &b1,
                            &b2,
                            &b3,
                            ip,
                            kp,
                            0,
                            3,
                            inp_row_stride,
                            ker_row_stride,
                            input_offset);
                acc_col_4ch(&b0,
                            &b1,
                            &b2,
                            &b3,
                            ip + col_stride,
                            kp + col_stride,
                            0,
                            3,
                            inp_row_stride,
                            ker_row_stride,
                            input_offset);
                if (right_ok)
                {
                    acc_col_4ch(&b0,
                                &b1,
                                &b2,
                                &b3,
                                ip + 2 * col_stride,
                                kp + 2 * col_stride,
                                0,
                                3,
                                inp_row_stride,
                                ker_row_stride,
                                input_offset);
                }

                b0                = nn_requantize(
                                        b0, output_mult[ch + 0], output_shift[ch + 0])
                                    + output_offset;
                b1                = nn_requantize(
                                        b1, output_mult[ch + 1], output_shift[ch + 1])
                                    + output_offset;
                b2                = nn_requantize(
                                        b2, output_mult[ch + 2], output_shift[ch + 2])
                                    + output_offset;
                b3                = nn_requantize(
                                        b3, output_mult[ch + 3], output_shift[ch + 3])
                                    + output_offset;
                output[out_idx++] = (int8_t)MIN(MAX(b0, output_activation_min),
                                                output_activation_max);
                output[out_idx++] = (int8_t)MIN(MAX(b1, output_activation_min),
                                                output_activation_max);
                output[out_idx++] = (int8_t)MIN(MAX(b2, output_activation_min),
                                                output_activation_max);
                output[out_idx++] = (int8_t)MIN(MAX(b3, output_activation_min),
                                                output_activation_max);
            }
            for (; ch < input_ch; ++ch)
            {
                const int8_t *ip = inp_base + ch;
                const int8_t *kp = kernel + ch;
                int32_t       b0 = bias[ch];

                acc_col_1ch(&b0,
                            ip,
                            kp,
                            0,
                            3,
                            inp_row_stride,
                            ker_row_stride,
                            input_offset);
                acc_col_1ch(&b0,
                            ip + col_stride,
                            kp + col_stride,
                            0,
                            3,
                            inp_row_stride,
                            ker_row_stride,
                            input_offset);
                if (right_ok)
                {
                    acc_col_1ch(&b0,
                                ip + 2 * col_stride,
                                kp + 2 * col_stride,
                                0,
                                3,
                                inp_row_stride,
                                ker_row_stride,
                                input_offset);
                }

                b0 = nn_requantize(b0, output_mult[ch], output_shift[ch])
                     + output_offset;
                output[out_idx++] = (int8_t)MIN(MAX(b0, output_activation_min),
                                                output_activation_max);
            }
        }
    }

    /* Bottom border rows */
    for (int32_t out_h = int_h1; out_h < output_y; ++out_h)
    {
        const int32_t in_h   = out_h * stride_y - pad_y;
        const int32_t kh_end = MIN(3, input_y - in_h);

        for (int32_t out_w = 0; out_w < output_x; ++out_w)
        {
            const int32_t in_w     = out_w * stride_x - pad_x;
            const int32_t kw_start = (in_w < 0) ? -in_w : 0;
            const bool    right_ok = (in_w + 2) < input_x;
            const int8_t *inp_base
                = input + in_h * inp_row_stride + in_w * col_stride;

            int32_t ch = 0;
            for (; ch <= (input_ch - 4); ch += 4)
            {
                const int8_t *ip = inp_base + ch;
                const int8_t *kp = kernel + ch;
                int32_t       b0 = bias[ch + 0], b1 = bias[ch + 1];
                int32_t       b2 = bias[ch + 2], b3 = bias[ch + 3];

                if (kw_start == 0)
                {
                    acc_col_4ch(&b0,
                                &b1,
                                &b2,
                                &b3,
                                ip,
                                kp,
                                0,
                                kh_end,
                                inp_row_stride,
                                ker_row_stride,
                                input_offset);
                }
                acc_col_4ch(&b0,
                            &b1,
                            &b2,
                            &b3,
                            ip + col_stride,
                            kp + col_stride,
                            0,
                            kh_end,
                            inp_row_stride,
                            ker_row_stride,
                            input_offset);
                if (right_ok)
                {
                    acc_col_4ch(&b0,
                                &b1,
                                &b2,
                                &b3,
                                ip + 2 * col_stride,
                                kp + 2 * col_stride,
                                0,
                                kh_end,
                                inp_row_stride,
                                ker_row_stride,
                                input_offset);
                }

                b0                = nn_requantize(
                                        b0, output_mult[ch + 0], output_shift[ch + 0])
                                    + output_offset;
                b1                = nn_requantize(
                                        b1, output_mult[ch + 1], output_shift[ch + 1])
                                    + output_offset;
                b2                = nn_requantize(
                                        b2, output_mult[ch + 2], output_shift[ch + 2])
                                    + output_offset;
                b3                = nn_requantize(
                                        b3, output_mult[ch + 3], output_shift[ch + 3])
                                    + output_offset;
                output[out_idx++] = (int8_t)MIN(MAX(b0, output_activation_min),
                                                output_activation_max);
                output[out_idx++] = (int8_t)MIN(MAX(b1, output_activation_min),
                                                output_activation_max);
                output[out_idx++] = (int8_t)MIN(MAX(b2, output_activation_min),
                                                output_activation_max);
                output[out_idx++] = (int8_t)MIN(MAX(b3, output_activation_min),
                                                output_activation_max);
            }
            for (; ch < input_ch; ++ch)
            {
                const int8_t *ip = inp_base + ch;
                const int8_t *kp = kernel + ch;
                int32_t       b0 = bias[ch];

                if (kw_start == 0)
                {
                    acc_col_1ch(&b0,
                                ip,
                                kp,
                                0,
                                kh_end,
                                inp_row_stride,
                                ker_row_stride,
                                input_offset);
                }
                acc_col_1ch(&b0,
                            ip + col_stride,
                            kp + col_stride,
                            0,
                            kh_end,
                            inp_row_stride,
                            ker_row_stride,
                            input_offset);
                if (right_ok)
                {
                    acc_col_1ch(&b0,
                                ip + 2 * col_stride,
                                kp + 2 * col_stride,
                                0,
                                kh_end,
                                inp_row_stride,
                                ker_row_stride,
                                input_offset);
                }

                b0 = nn_requantize(b0, output_mult[ch], output_shift[ch])
                     + output_offset;
                output[out_idx++] = (int8_t)MIN(MAX(b0, output_activation_min),
                                                output_activation_max);
            }
        }
    }

    return 0;
}
