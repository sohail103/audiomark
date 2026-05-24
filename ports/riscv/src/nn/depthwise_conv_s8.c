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

/*
 * bundling stride layout constants so a single pointer can be passed instead of
 * 3 separate scalars
 */
typedef struct
{
    int32_t inp_row_stride; /* input_ch * input_x   */
    int32_t ker_row_stride; /* input_ch * 3         */
    int32_t col_stride;     /* input_ch             */
} nn_conv_strides;

/*
 * quantisation and activation scalars needed to finish every output sample
 */
typedef struct
{
    const int32_t *output_mult;
    const int32_t *output_shift;
    int32_t        output_offset;
    int32_t        output_activation_min;
    int32_t        output_activation_max;
} nn_conv_out_params;

/*
 * compute_interior_col_range - returns the [w0, w1) column range where all 3
 * kernel columns are fully in-bounds.
 */
static void
compute_interior_col_range(int32_t  output_x,
                           int32_t  stride_x,
                           int32_t  pad_x,
                           int32_t  input_x,
                           int32_t *out_w0,
                           int32_t *out_w1)
{
    int32_t w0 = 0;
    while (w0 < output_x && (w0 * stride_x - pad_x) < 0)
    {
        ++w0;
    }

    int32_t w1 = w0;
    while (w1 < output_x && (w1 * stride_x - pad_x + 2) < input_x)
    {
        ++w1;
    }

    *out_w0 = w0;
    *out_w1 = w1;
}

/*
 * process_border_pixel - computes one output pixel whose kernel window may be
 * partially out-of-bounds in either the horizontal or vertical dimension (or
 * both).
 */
static void
process_border_pixel(const int8_t             *inp_base,
                     const int8_t             *kernel,
                     const int32_t            *bias,
                     int32_t                   input_ch,
                     int32_t                   input_offset,
                     int32_t                   kh_start,
                     int32_t                   kh_end,
                     int32_t                   kw_start,
                     int32_t                   right_ok,
                     const nn_conv_strides    *strides,
                     const nn_conv_out_params *outp,
                     q7_t                    **out_ptr)
{
    const int32_t inp_rs = strides->inp_row_stride;
    const int32_t ker_rs = strides->ker_row_stride;
    const int32_t col_s  = strides->col_stride;

    for (int32_t ch = 0; ch < input_ch; ++ch)
    {
        const int8_t *ip = inp_base + ch;
        const int8_t *kp = kernel + ch;
        int32_t       b0 = bias[ch];

        if (kw_start == 0)
        {
            ACC_COL(b0, ip, kp, kh_start, kh_end, inp_rs, ker_rs, input_offset);
        }

        ACC_COL(b0,
                ip + col_s,
                kp + col_s,
                kh_start,
                kh_end,
                inp_rs,
                ker_rs,
                input_offset);

        if (right_ok)
        {
            ACC_COL(b0,
                    ip + 2 * col_s,
                    kp + 2 * col_s,
                    kh_start,
                    kh_end,
                    inp_rs,
                    ker_rs,
                    input_offset);
        }

        b0 = nn_requantize(b0, outp->output_mult[ch], outp->output_shift[ch])
             + outp->output_offset;
        *(*out_ptr)++ = (int8_t)MIN(MAX(b0, outp->output_activation_min),
                                    outp->output_activation_max);
    }
}

/*
 * all 9 kernel taps are in-bounds. fully unrolled fast path.
 */
static void
process_interior_pixel(const int8_t             *ip0,
                       const int8_t             *kernel,
                       const int32_t            *bias,
                       int32_t                   input_ch,
                       int32_t                   input_offset,
                       const nn_conv_strides    *strides,
                       const nn_conv_out_params *outp,
                       q7_t                    **out_ptr)
{
    const int32_t inp_rs = strides->inp_row_stride;
    const int32_t ker_rs = strides->ker_row_stride;
    const int32_t col_s  = strides->col_stride;

    for (int32_t ch = 0; ch < input_ch; ++ch)
    {
        const int8_t *ip = ip0 + ch;
        const int8_t *kp = kernel + ch;
        int32_t       b0 = bias[ch];

        b0 += (ip[0 * inp_rs + 0 * col_s] + input_offset)
              * kp[0 * ker_rs + 0 * col_s];
        b0 += (ip[0 * inp_rs + 1 * col_s] + input_offset)
              * kp[0 * ker_rs + 1 * col_s];
        b0 += (ip[0 * inp_rs + 2 * col_s] + input_offset)
              * kp[0 * ker_rs + 2 * col_s];
        b0 += (ip[1 * inp_rs + 0 * col_s] + input_offset)
              * kp[1 * ker_rs + 0 * col_s];
        b0 += (ip[1 * inp_rs + 1 * col_s] + input_offset)
              * kp[1 * ker_rs + 1 * col_s];
        b0 += (ip[1 * inp_rs + 2 * col_s] + input_offset)
              * kp[1 * ker_rs + 2 * col_s];
        b0 += (ip[2 * inp_rs + 0 * col_s] + input_offset)
              * kp[2 * ker_rs + 0 * col_s];
        b0 += (ip[2 * inp_rs + 1 * col_s] + input_offset)
              * kp[2 * ker_rs + 1 * col_s];
        b0 += (ip[2 * inp_rs + 2 * col_s] + input_offset)
              * kp[2 * ker_rs + 2 * col_s];

        b0 = nn_requantize(b0, outp->output_mult[ch], outp->output_shift[ch])
             + outp->output_offset;
        *(*out_ptr)++ = (int8_t)MIN(MAX(b0, outp->output_activation_min),
                                    outp->output_activation_max);
    }
}

/*
 * handles one output row whose vertical kernel window is clipped
 * by the top or bottom border
 */
static void
process_border_row(int32_t                   out_w,     /* first col  */
                   int32_t                   out_w_end, /* past-end   */
                   int32_t                   stride_x,
                   int32_t                   pad_x,
                   int32_t                   input_x,
                   int32_t                   input_ch,
                   int32_t                   input_offset,
                   int32_t                   kh_start,
                   int32_t                   kh_end,
                   const int8_t             *input_row, /* row base   */
                   const int8_t             *kernel,
                   const int32_t            *bias,
                   const nn_conv_strides    *strides,
                   const nn_conv_out_params *outp,
                   q7_t                    **out_ptr)
{
    for (; out_w < out_w_end; ++out_w)
    {
        const int32_t in_w     = out_w * stride_x - pad_x;
        const int32_t kw_start = (in_w < 0) ? -in_w : 0;
        const int32_t right_ok = (in_w + 2) < input_x;
        const int8_t *inp_base = input_row + in_w * strides->col_stride;

        process_border_pixel(inp_base,
                             kernel,
                             bias,
                             input_ch,
                             input_offset,
                             kh_start,
                             kh_end,
                             kw_start,
                             right_ok,
                             strides,
                             outp,
                             out_ptr);
    }
}

/*
 * process_interior_row
 *
 * handles one complete interior output row, split into three column bands:
 *   left border  [0,       int_w0)  — horizontal kernel may be clipped
 *   interior     [int_w0,  int_w1)  — all 9 taps in-bounds (fast path)
 *   right border [int_w1,  out_x)   — right kernel column may be clipped
 */
static void
process_interior_row(int32_t                   out_h,
                     int32_t                   output_x,
                     int32_t                   stride_x,
                     int32_t                   pad_x,
                     int32_t                   input_x,
                     int32_t                   input_ch,
                     int32_t                   input_offset,
                     const int8_t             *input,
                     const int8_t             *kernel,
                     const int32_t            *bias,
                     const nn_conv_strides    *strides,
                     const nn_conv_out_params *outp,
                     int32_t                   in_h,
                     q7_t                    **out_ptr)
{
    int32_t int_w0, int_w1;
    compute_interior_col_range(
        output_x, stride_x, pad_x, input_x, &int_w0, &int_w1);

    const int8_t *input_row = input + in_h * strides->inp_row_stride;

    /* --- left border columns (kh always full, kw may be clipped left) --- */
    for (int32_t out_w = 0; out_w < int_w0; ++out_w)
    {
        const int32_t in_w     = out_w * stride_x - pad_x;
        const int32_t kw_start = -in_w; /* in_w < 0 guaranteed here */
        const int32_t right_ok = (in_w + 2) < input_x;
        const int8_t *inp_base = input_row + in_w * strides->col_stride;

        process_border_pixel(inp_base,
                             kernel,
                             bias,
                             input_ch,
                             input_offset,
                             /*kh_start=*/0,
                             /*kh_end=*/3,
                             kw_start,
                             right_ok,
                             strides,
                             outp,
                             out_ptr);
    }

    /* fully interior columns: all 9 taps valid  */
    for (int32_t out_w = int_w0; out_w < int_w1; ++out_w)
    {
        const int32_t in_w = out_w * stride_x - pad_x;
        const int8_t *ip0  = input_row + in_w * strides->col_stride;

        process_interior_pixel(
            ip0, kernel, bias, input_ch, input_offset, strides, outp, out_ptr);
    }

    /* right border columns (kw may be clipped right) */
    for (int32_t out_w = int_w1; out_w < output_x; ++out_w)
    {
        const int32_t in_w     = out_w * stride_x - pad_x;
        const int32_t right_ok = (in_w + 2) < input_x;
        const int8_t *inp_base = input_row + in_w * strides->col_stride;

        /* kw_start is always 0 in the right-border region */
        process_border_pixel(inp_base,
                             kernel,
                             bias,
                             input_ch,
                             input_offset,
                             /*kh_start=*/0,
                             /*kh_end=*/3,
                             /*kw_start=*/0,
                             right_ok,
                             strides,
                             outp,
                             out_ptr);
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

    const nn_conv_strides strides = {
        .inp_row_stride = input_ch * input_x,
        .ker_row_stride = input_ch * 3,
        .col_stride     = input_ch,
    };

    const nn_conv_out_params outp = {
        .output_mult           = quant_params->multiplier,
        .output_shift          = quant_params->shift,
        .output_offset         = dw_conv_params->output_offset,
        .output_activation_min = dw_conv_params->activation.min,
        .output_activation_max = dw_conv_params->activation.max,
    };

    /* Compute interior rectangle boundaries (vertical). */
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

    q7_t *out_ptr = output;

    /* Top border rows (kernel clipped at the top) */
    for (int32_t out_h = 0; out_h < int_h0; ++out_h)
    {
        const int32_t in_h      = out_h * stride_y - pad_y;
        const int32_t kh_start  = -in_h;
        const int32_t kh_end    = MIN(3, input_y - in_h);
        const int8_t *input_row = input + in_h * strides.inp_row_stride;

        process_border_row(0,
                           output_x,
                           stride_x,
                           pad_x,
                           input_x,
                           input_ch,
                           dw_conv_params->input_offset,
                           kh_start,
                           kh_end,
                           input_row,
                           kernel,
                           bias,
                           &strides,
                           &outp,
                           &out_ptr);
    }

    /* Interior rows (full vertical kernel) */
    for (int32_t out_h = int_h0; out_h < int_h1; ++out_h)
    {
        const int32_t in_h = out_h * stride_y - pad_y;
        process_interior_row(out_h,
                             output_x,
                             stride_x,
                             pad_x,
                             input_x,
                             input_ch,
                             dw_conv_params->input_offset,
                             input,
                             kernel,
                             bias,
                             &strides,
                             &outp,
                             in_h,
                             &out_ptr);
    }

    /* Bottom border rows (kernel clipped at the bottom) */
    for (int32_t out_h = int_h1; out_h < output_y; ++out_h)
    {
        const int32_t in_h      = out_h * stride_y - pad_y;
        const int32_t kh_end    = MIN(3, input_y - in_h);
        const int8_t *input_row = input + in_h * strides.inp_row_stride;

        process_border_row(0,
                           output_x,
                           stride_x,
                           pad_x,
                           input_x,
                           input_ch,
                           dw_conv_params->input_offset,
                           /*kh_start=*/0,
                           kh_end,
                           input_row,
                           kernel,
                           bias,
                           &strides,
                           &outp,
                           &out_ptr);
    }

    return 0;
}
