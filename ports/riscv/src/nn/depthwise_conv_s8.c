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

/*
 * acc_col_1ch: accumulate one kernel column (up to 3 rows) into a single
 * channel accumulator. kh_start/kh_end select which rows are valid.
 * Keeping this as a macro rather than an inline function avoids the
 * parameter-passing register pressure that contributes to spills.
 */
#define ACC_COL(b, ip, kp, kh_start, kh_end, inp_rs, ker_rs, ioff)     \
    do                                                                 \
    {                                                                  \
        if ((kh_start) <= 0 && 0 < (kh_end))                           \
            (b) += ((ip)[0 * (inp_rs)] + (ioff)) * (kp)[0 * (ker_rs)]; \
        if ((kh_start) <= 1 && 1 < (kh_end))                           \
            (b) += ((ip)[1 * (inp_rs)] + (ioff)) * (kp)[1 * (ker_rs)]; \
        if ((kh_start) <= 2 && 2 < (kh_end))                           \
            (b) += ((ip)[2 * (inp_rs)] + (ioff)) * (kp)[2 * (ker_rs)]; \
    } while (0)

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

    /* Compute interior rectangle boundaries. */
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
            const int32_t right_ok = (in_w + 2) < input_x;
            const int8_t *inp_base
                = input + in_h * inp_row_stride + in_w * col_stride;

            for (int32_t ch = 0; ch < input_ch; ++ch)
            {
                const int8_t *ip = inp_base + ch;
                const int8_t *kp = kernel + ch;
                int32_t       b0 = bias[ch];

                if (kw_start == 0)
                {
                    ACC_COL(b0,
                            ip,
                            kp,
                            kh_start,
                            kh_end,
                            inp_row_stride,
                            ker_row_stride,
                            input_offset);
                }

                ACC_COL(b0,
                        ip + col_stride,
                        kp + col_stride,
                        kh_start,
                        kh_end,
                        inp_row_stride,
                        ker_row_stride,
                        input_offset);

                if (right_ok)
                {
                    ACC_COL(b0,
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

        /* Left border columns */
        for (int32_t out_w = 0; out_w < int_w0; ++out_w)
        {
            const int32_t in_w     = out_w * stride_x - pad_x;
            const int32_t kw_start = -in_w;
            const int32_t right_ok = (in_w + 2) < input_x;
            const int8_t *inp_base
                = input + in_h * inp_row_stride + in_w * col_stride;

            for (int32_t ch = 0; ch < input_ch; ++ch)
            {
                const int8_t *ip = inp_base + ch;
                const int8_t *kp = kernel + ch;
                int32_t       b0 = bias[ch];

                if (kw_start == 0)
                {
                    ACC_COL(b0,
                            ip,
                            kp,
                            0,
                            3,
                            inp_row_stride,
                            ker_row_stride,
                            input_offset);
                }

                ACC_COL(b0,
                        ip + col_stride,
                        kp + col_stride,
                        0,
                        3,
                        inp_row_stride,
                        ker_row_stride,
                        input_offset);

                if (right_ok)
                {
                    ACC_COL(b0,
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

        /* Interior columns: all 9 taps valid, fully unrolled, 1-channel */
        for (int32_t out_w = int_w0; out_w < int_w1; ++out_w)
        {
            const int32_t in_w = out_w * stride_x - pad_x;
            const int8_t *ip0
                = input + in_h * inp_row_stride + in_w * col_stride;

            for (int32_t ch = 0; ch < input_ch; ++ch)
            {
                const int8_t *ip = ip0 + ch;
                const int8_t *kp = kernel + ch;
                int32_t       b0 = bias[ch];

                b0 += (ip[0 * inp_row_stride + 0 * col_stride] + input_offset)
                      * kp[0 * ker_row_stride + 0 * col_stride];
                b0 += (ip[0 * inp_row_stride + 1 * col_stride] + input_offset)
                      * kp[0 * ker_row_stride + 1 * col_stride];
                b0 += (ip[0 * inp_row_stride + 2 * col_stride] + input_offset)
                      * kp[0 * ker_row_stride + 2 * col_stride];
                b0 += (ip[1 * inp_row_stride + 0 * col_stride] + input_offset)
                      * kp[1 * ker_row_stride + 0 * col_stride];
                b0 += (ip[1 * inp_row_stride + 1 * col_stride] + input_offset)
                      * kp[1 * ker_row_stride + 1 * col_stride];
                b0 += (ip[1 * inp_row_stride + 2 * col_stride] + input_offset)
                      * kp[1 * ker_row_stride + 2 * col_stride];
                b0 += (ip[2 * inp_row_stride + 0 * col_stride] + input_offset)
                      * kp[2 * ker_row_stride + 0 * col_stride];
                b0 += (ip[2 * inp_row_stride + 1 * col_stride] + input_offset)
                      * kp[2 * ker_row_stride + 1 * col_stride];
                b0 += (ip[2 * inp_row_stride + 2 * col_stride] + input_offset)
                      * kp[2 * ker_row_stride + 2 * col_stride];

                b0 = nn_requantize(b0, output_mult[ch], output_shift[ch])
                     + output_offset;
                output[out_idx++] = (int8_t)MIN(MAX(b0, output_activation_min),
                                                output_activation_max);
            }
        }

        /* Right border columns */
        for (int32_t out_w = int_w1; out_w < output_x; ++out_w)
        {
            const int32_t in_w     = out_w * stride_x - pad_x;
            const int32_t right_ok = (in_w + 2) < input_x;
            const int8_t *inp_base
                = input + in_h * inp_row_stride + in_w * col_stride;

            for (int32_t ch = 0; ch < input_ch; ++ch)
            {
                const int8_t *ip = inp_base + ch;
                const int8_t *kp = kernel + ch;
                int32_t       b0 = bias[ch];

                /* kw_start is always 0 in the right-border region */
                ACC_COL(b0,
                        ip,
                        kp,
                        0,
                        3,
                        inp_row_stride,
                        ker_row_stride,
                        input_offset);
                ACC_COL(b0,
                        ip + col_stride,
                        kp + col_stride,
                        0,
                        3,
                        inp_row_stride,
                        ker_row_stride,
                        input_offset);
                if (right_ok)
                {
                    ACC_COL(b0,
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
            const int32_t right_ok = (in_w + 2) < input_x;
            const int8_t *inp_base
                = input + in_h * inp_row_stride + in_w * col_stride;

            for (int32_t ch = 0; ch < input_ch; ++ch)
            {
                const int8_t *ip = inp_base + ch;
                const int8_t *kp = kernel + ch;
                int32_t       b0 = bias[ch];

                if (kw_start == 0)
                {
                    ACC_COL(b0,
                            ip,
                            kp,
                            0,
                            kh_end,
                            inp_row_stride,
                            ker_row_stride,
                            input_offset);
                }

                ACC_COL(b0,
                        ip + col_stride,
                        kp + col_stride,
                        0,
                        kh_end,
                        inp_row_stride,
                        ker_row_stride,
                        input_offset);

                if (right_ok)
                {
                    ACC_COL(b0,
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
