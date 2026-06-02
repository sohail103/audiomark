/*
 * Copyright 2026 Google LLC
 * Copyright 2026 Sohail Raj Satapathy
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include "functions.h"
#include "support_functions.h"
#include "ee_api.h"

#include <riscv_vector.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

/* ---- constants ---------------------------------------------------------- */
#define INPUT_BUFFER_SIZE (64u * 1024u)
#define PAD_PIXEL         4
#define MAX_OUTPUT_DEPTH  256

/* ---- helpers ------------------------------------------------------------ */

/* Align a pointer up to the next multiple of `align` bytes. */
static inline uint8_t *
align_ptr(uint8_t *p, size_t align)
{
    size_t r = (size_t)p % align;
    return r ? p + (align - r) : p;
}

/* Flat index into a 4-D NHWC buffer. */
static inline int32_t
offset4(int32_t n_dim,
        int32_t h_dim,
        int32_t w_dim,
        int32_t c_dim,
        int32_t n,
        int32_t h,
        int32_t w,
        int32_t c)
{
    return ((n * h_dim + h) * w_dim + w) * c_dim + c;
}

static void
prepare_shift_params(uint8_t       *shift_left,
                     uint8_t       *shift_right,
                     const int32_t *output_shift,
                     int32_t        depth)
{
    for (int32_t i = 0; i < depth; i++)
    {
        shift_left[i]  = (uint8_t)(output_shift[i] > 0 ? output_shift[i] : 0);
        shift_right[i] = (uint8_t)(output_shift[i] < 0 ? -output_shift[i] : 0);
    }
}

#define CONV_MAC(in_ptr, fil, acc)                                         \
    do                                                                     \
    {                                                                      \
        vint8m1_t  _in8  = __riscv_vle8_v_i8m1((in_ptr), vl);              \
        vint16m2_t _in16 = __riscv_vsext_vf2_i16m2(_in8, vl);              \
        _in16 = __riscv_vadd_vx_i16m2(_in16, (int16_t)(input_offset), vl); \
        vint16m2_t _w16 = __riscv_vsext_vf2_i16m2((fil), vl);              \
        (acc)           = __riscv_vwmacc_vv_i32m4((acc), _in16, _w16, vl); \
    } while (0)

#define CONV_MAC_2X(in_ptr, fil_for_acc1, fil_for_acc2)                         \
    do                                                                          \
    {                                                                           \
        vint8m1_t  _in8  = __riscv_vle8_v_i8m1((in_ptr), vl);                   \
        vint16m2_t _in16 = __riscv_vsext_vf2_i16m2(_in8, vl);                   \
        _in16 = __riscv_vadd_vx_i16m2(_in16, (int16_t)(input_offset), vl);      \
        vint16m2_t _w16a = __riscv_vsext_vf2_i16m2((fil_for_acc1), vl);         \
        vint16m2_t _w16b = __riscv_vsext_vf2_i16m2((fil_for_acc2), vl);         \
        mul_acc1         = __riscv_vwmacc_vv_i32m4(mul_acc1, _in16, _w16a, vl); \
        mul_acc2         = __riscv_vwmacc_vv_i32m4(mul_acc2, _in16, _w16b, vl); \
    } while (0)

static void
postprocess_acc(const int32_t *accs_buf,
                const int32_t *bias_data,
                const uint8_t *shift_left,
                const int32_t *output_mult,
                const uint8_t *shift_right,
                int32_t        out_offset,
                int32_t        out_activation_min,
                int32_t        out_activation_max,
                q7_t          *output_data,
                int32_t        num_pixels,
                int32_t        output_ch)
{
    for (int32_t pix = 0; pix < num_pixels; pix++)
    {
        for (int32_t oc = 0; oc < output_ch; oc++)
        {
            int32_t acc = accs_buf[pix * output_ch + oc];
            if (bias_data)
            {
                acc += bias_data[oc];
            }

            /* Left shift before multiplier */
            acc <<= shift_left[oc];
            /* Rounding multiply (nn_requantize-compatible) */
            acc = nn_requantize(
                acc, output_mult[oc], -(int32_t)shift_right[oc]);

            acc += out_offset;
            acc = acc > out_activation_max ? out_activation_max : acc;
            acc = acc < out_activation_min ? out_activation_min : acc;
            output_data[pix * output_ch + oc] = (q7_t)acc;
        }
    }
}

static const int8_t *
pad_input(int8_t       *dst_buffer,
          const int8_t *src,
          int32_t       input_height,
          int32_t       input_width,
          int32_t       input_depth,
          int32_t       input_offset)
{
    memset(dst_buffer, (int)(-input_offset), INPUT_BUFFER_SIZE);

    int32_t row_bytes     = input_width * input_depth;
    int32_t pad_bytes     = PAD_PIXEL * input_depth;
    int32_t stride_padded = row_bytes + 2 * pad_bytes;

    int32_t       required_size = input_height * stride_padded;
    const int32_t kMargin       = 1024;

    if ((size_t)(required_size + kMargin) > INPUT_BUFFER_SIZE)
    {
        return NULL;
    }

    int8_t *buffer_start = dst_buffer + kMargin;

    for (int32_t y = 0; y < input_height; y++)
    {
        int8_t       *dst_row = buffer_start + y * stride_padded + pad_bytes;
        const int8_t *src_row = src + y * row_bytes;
        memcpy(dst_row, src_row, (size_t)row_bytes);
    }

    return buffer_start + pad_bytes;
}

static const int8_t *
tiled_pad_input(int8_t       *dst_buffer,
                const int8_t *src_batch,
                int32_t       input_height,
                int32_t       input_width,
                int32_t       input_depth,
                int32_t       input_offset)
{
    const int32_t margin_h = 4;
    const int32_t margin_w = 4;

    int32_t padded_width = input_width + 2 * margin_w;
    int32_t dst_stride   = padded_width * input_depth;
    int32_t src_stride   = input_width * input_depth;

    memset(dst_buffer,
           (int)((int8_t)(-input_offset)),
           (size_t)((input_height + 2 * margin_h) * dst_stride));

    int8_t *dst_base
        = dst_buffer + margin_h * dst_stride + margin_w * input_depth;

    for (int32_t y = 0; y < input_height; y++)
    {
        const int8_t *src_row = src_batch + y * src_stride;
        int8_t       *dst_row = dst_base + y * dst_stride;
        memcpy(dst_row, src_row, (size_t)src_stride);
    }

    return dst_buffer;
}

static void
repack_weights_d48(const int8_t *src,
                   int16_t      *dst,
                   int32_t      *weight_sums,
                   int32_t       output_depth,
                   int32_t       filter_height,
                   int32_t       filter_width,
                   int32_t       input_depth)
{
    const int32_t oc_block_size = 16;

    for (int32_t oc = 0; oc < output_depth; oc++)
    {
        weight_sums[oc] = 0;
    }

    for (int32_t oc_block = 0; oc_block < output_depth;
         oc_block += oc_block_size)
    {
        for (int32_t ky = 0; ky < filter_height; ky++)
        {
            for (int32_t kx = 0; kx < filter_width; kx++)
            {
                for (int32_t ic = 0; ic < input_depth; ic++)
                {
                    for (int32_t oc_inner = 0; oc_inner < oc_block_size;
                         oc_inner++)
                    {
                        int32_t oc = oc_block + oc_inner;
                        if (oc < output_depth)
                        {
                            int32_t src_idx
                                = oc
                                      * (filter_height * filter_width
                                         * input_depth)
                                  + ky * (filter_width * input_depth)
                                  + kx * input_depth + ic;
                            int8_t val = src[src_idx];
                            *dst++     = (int16_t)val;
                            weight_sums[oc] += (int32_t)val;
                        }
                        else
                        {
                            *dst++ = 0;
                        }
                    }
                }
            }
        }
    }
}

static void
conv2d_4x4(const nn_conv_params              *conv_params,
           const nn_per_channel_quant_params *quant_params,
           const nn_dims                     *input_dims,
           const int8_t                      *input_data,
           const nn_dims                     *filter_dims,
           const int8_t                      *filter_data,
           const int32_t                     *bias_data,
           const nn_dims                     *output_dims,
           q7_t                              *output_data,
           /* generic repacked weights [ky][kx][ic][oc] */
           const int8_t *repacked_weights,
           /* scratch buffer for padded input (INPUT_BUFFER_SIZE) */
           int8_t *generic_tiled_buffer,
           /* per-channel shift arrays */
           const uint8_t *shift_left_arr,
           const uint8_t *shift_right_arr)
{
    const int32_t stride_width  = conv_params->stride.w;
    const int32_t stride_height = conv_params->stride.h;
    const int32_t dilation_w    = conv_params->dilation.w;
    const int32_t dilation_h    = conv_params->dilation.h;
    const int32_t pad_width     = conv_params->padding.w;
    const int32_t pad_height    = conv_params->padding.h;
    const int32_t input_offset  = conv_params->input_offset;
    const int32_t output_offset = conv_params->output_offset;

    const int32_t batches      = input_dims->n;
    const int32_t input_height = input_dims->h;
    const int32_t input_width  = input_dims->w;
    const int32_t input_depth  = input_dims->c;

    /* filter_height and filter_width are fixed 4x4 */
    const int32_t filter_depth  = filter_dims->c; /* == input_depth */
    const int32_t output_depth  = filter_dims->n; /* number of filters */
    const int32_t stride_filter = 4 * 4 * filter_depth;

    const int32_t output_height = output_dims->h;
    const int32_t output_width  = output_dims->w;

    for (int32_t out_channel_start = 0; out_channel_start < output_depth;)
    {

        int32_t rem_out_channels = output_depth - out_channel_start;
        size_t  vl = __riscv_vsetvl_e32m4((size_t)rem_out_channels);

        /* --- Load bias & quantisation parameters for this OC block --- */
        vint32m4_t bias_v
            = __riscv_vle32_v_i32m4(bias_data + out_channel_start, vl);
        vint32m4_t mult_v = __riscv_vle32_v_i32m4(
            quant_params->multiplier + out_channel_start, vl);
        vint32m4_t shift_v = __riscv_vle32_v_i32m4(
            quant_params->shift + out_channel_start, vl);

        /* Decompose shift into left / right components */
        vint32m4_t left_shift_v = __riscv_vmax_vx_i32m4(shift_v, 0, vl);
        vint32m4_t right_shift_v
            = __riscv_vmax_vx_i32m4(__riscv_vneg_v_i32m4(shift_v, vl), 0, vl);

        /* Rounding nudge: nudge = (right_shift > 0) ? (1 << (right_shift-1)) :
         * 0 */
        vint32m4_t shift_m1_v = __riscv_vsub_vx_i32m4(right_shift_v, 1, vl);
        vbool8_t   mask_gt0   = __riscv_vmsgt_vx_i32m4_b8(right_shift_v, 0, vl);
        vint32m4_t nudge_v    = __riscv_vmv_v_x_i32m4(0, vl);
        nudge_v               = __riscv_vsll_vv_i32m4_mu(
            mask_gt0,
            nudge_v,
            __riscv_vmv_v_x_i32m4(1, vl),
            __riscv_vreinterpret_v_i32m4_u32m4(shift_m1_v),
            vl);

        /* Scalar activation bounds */
        const int32_t out_act_min = conv_params->activation.min;
        const int32_t out_act_max = conv_params->activation.max;

        for (int32_t batch = 0; batch < batches; batch++)
        {

            const int8_t *batch_base
                = input_data + batch * input_height * input_width * input_depth;

            /* Optionally pad the batch into the scratch buffer */
            const int8_t *padded_input_base = NULL;
            if (generic_tiled_buffer)
            {
                padded_input_base = pad_input(generic_tiled_buffer,
                                              batch_base,
                                              input_height,
                                              input_width,
                                              input_depth,
                                              input_offset);
            }

            for (int32_t out_y = 0; out_y < output_height; out_y++)
            {
                const int32_t in_y_origin = out_y * stride_height - pad_height;

                int32_t out_x = 0;

                /* --- 4-pixel-wide unrolled X loop --- */
                for (; out_x <= output_width - 4; out_x += 4)
                {

                    const int32_t in_x0
                        = (out_x + 0) * stride_width - pad_width;
                    const int32_t in_x1
                        = (out_x + 1) * stride_width - pad_width;
                    const int32_t in_x2
                        = (out_x + 2) * stride_width - pad_width;
                    const int32_t in_x3
                        = (out_x + 3) * stride_width - pad_width;

                    vint32m4_t acc0 = bias_v;
                    vint32m4_t acc1 = bias_v;
                    vint32m4_t acc2 = bias_v;
                    vint32m4_t acc3 = bias_v;

                    for (int32_t ky = 0; ky < 4; ky++)
                    {
                        const int32_t in_y = in_y_origin + dilation_h * ky;
                        if (in_y < 0 || in_y >= input_height)
                        {
                            continue;
                        }

                        for (int32_t kx = 0; kx < 4; kx++)
                        {
                            const int32_t ix0 = in_x0 + dilation_w * kx;
                            const int32_t ix1 = in_x1 + dilation_w * kx;
                            const int32_t ix2 = in_x2 + dilation_w * kx;
                            const int32_t ix3 = in_x3 + dilation_w * kx;

                            const int32_t row_offset
                                = in_y * input_width * input_depth;
                            const int8_t *val_ptr = batch_base + row_offset;

                            const int32_t pad_bytes_per_row
                                = PAD_PIXEL * input_depth;
                            const int32_t stride_padded
                                = input_width * input_depth
                                  + 2 * pad_bytes_per_row;
                            const int8_t *pad_ptr
                                = padded_input_base ? (padded_input_base
                                                       + in_y * stride_padded)
                                                    : NULL;

                            if (repacked_weights)
                            {
                                /* Optimised path: weights already in
                                 * [ky][kx][ic][oc] order */
                                const int8_t *packed_ptr
                                    = repacked_weights
                                      + ky * 4 * input_depth * output_depth
                                      + kx * input_depth * output_depth
                                      + out_channel_start;

                                for (int32_t kc = 0; kc < input_depth; kc++)
                                {
                                    vint8m1_t w = __riscv_vle8_v_i8m1(
                                        packed_ptr + kc * output_depth, vl);
                                    vint16m2_t w16
                                        = __riscv_vsext_vf2_i16m2(w, vl);

                                    if (pad_ptr)
                                    {
                                        /* Branchless: padding holds
                                         * -input_offset so OOB → 0 contribution
                                         */
                                        int16_t v0
                                            = (int16_t)(pad_ptr
                                                            [ix0 * input_depth
                                                             + kc]
                                                        + input_offset);
                                        int16_t v1
                                            = (int16_t)(pad_ptr
                                                            [ix1 * input_depth
                                                             + kc]
                                                        + input_offset);
                                        int16_t v2
                                            = (int16_t)(pad_ptr
                                                            [ix2 * input_depth
                                                             + kc]
                                                        + input_offset);
                                        int16_t v3
                                            = (int16_t)(pad_ptr
                                                            [ix3 * input_depth
                                                             + kc]
                                                        + input_offset);
                                        acc0 = __riscv_vwmacc_vx_i32m4(
                                            acc0, v0, w16, vl);
                                        acc1 = __riscv_vwmacc_vx_i32m4(
                                            acc1, v1, w16, vl);
                                        acc2 = __riscv_vwmacc_vx_i32m4(
                                            acc2, v2, w16, vl);
                                        acc3 = __riscv_vwmacc_vx_i32m4(
                                            acc3, v3, w16, vl);
                                    }
                                    else
                                    {
                                        /* Fallback: explicit boundary guards */
                                        if (ix0 >= 0 && ix0 < input_width)
                                        {
                                            int16_t v
                                                = (int16_t)(val_ptr
                                                                [ix0 * input_depth
                                                                 + kc]
                                                            + input_offset);
                                            acc0 = __riscv_vwmacc_vx_i32m4(
                                                acc0, v, w16, vl);
                                        }
                                        if (ix1 >= 0 && ix1 < input_width)
                                        {
                                            int16_t v
                                                = (int16_t)(val_ptr
                                                                [ix1 * input_depth
                                                                 + kc]
                                                            + input_offset);
                                            acc1 = __riscv_vwmacc_vx_i32m4(
                                                acc1, v, w16, vl);
                                        }
                                        if (ix2 >= 0 && ix2 < input_width)
                                        {
                                            int16_t v
                                                = (int16_t)(val_ptr
                                                                [ix2 * input_depth
                                                                 + kc]
                                                            + input_offset);
                                            acc2 = __riscv_vwmacc_vx_i32m4(
                                                acc2, v, w16, vl);
                                        }
                                        if (ix3 >= 0 && ix3 < input_width)
                                        {
                                            int16_t v
                                                = (int16_t)(val_ptr
                                                                [ix3 * input_depth
                                                                 + kc]
                                                            + input_offset);
                                            acc3 = __riscv_vwmacc_vx_i32m4(
                                                acc3, v, w16, vl);
                                        }
                                    }
                                }
                            }
                            else
                            {
                                /* Original path: strided loads from filter
                                 * tensor */
                                const int8_t *f_ptr
                                    = filter_data
                                      + out_channel_start * stride_filter
                                      + ky * 4 * filter_depth
                                      + kx * filter_depth;

                                for (int32_t kc = 0; kc < input_depth; kc++)
                                {
                                    vint8m1_t w = __riscv_vlse8_v_i8m1(
                                        f_ptr + kc, (size_t)stride_filter, vl);
                                    vint16m2_t w16
                                        = __riscv_vsext_vf2_i16m2(w, vl);

                                    if (ix0 >= 0 && ix0 < input_width)
                                    {
                                        int16_t v
                                            = (int16_t)(val_ptr
                                                            [ix0 * input_depth
                                                             + kc]
                                                        + input_offset);
                                        acc0 = __riscv_vwmacc_vx_i32m4(
                                            acc0, v, w16, vl);
                                    }
                                    if (ix1 >= 0 && ix1 < input_width)
                                    {
                                        int16_t v
                                            = (int16_t)(val_ptr
                                                            [ix1 * input_depth
                                                             + kc]
                                                        + input_offset);
                                        acc1 = __riscv_vwmacc_vx_i32m4(
                                            acc1, v, w16, vl);
                                    }
                                    if (ix2 >= 0 && ix2 < input_width)
                                    {
                                        int16_t v
                                            = (int16_t)(val_ptr
                                                            [ix2 * input_depth
                                                             + kc]
                                                        + input_offset);
                                        acc2 = __riscv_vwmacc_vx_i32m4(
                                            acc2, v, w16, vl);
                                    }
                                    if (ix3 >= 0 && ix3 < input_width)
                                    {
                                        int16_t v
                                            = (int16_t)(val_ptr
                                                            [ix3 * input_depth
                                                             + kc]
                                                        + input_offset);
                                        acc3 = __riscv_vwmacc_vx_i32m4(
                                            acc3, v, w16, vl);
                                    }
                                }
                            }
                        }
                    }

/* Quantise a single accumulator and write vl bytes of int8 output. */
#define QUANTIZE_AND_STORE_4X4(ACC_V, X_OFF)                                   \
    do                                                                         \
    {                                                                          \
        vint32m4_t _a = (ACC_V);                                               \
        _a            = __riscv_vsll_vv_i32m4(                                 \
            _a, __riscv_vreinterpret_v_i32m4_u32m4(left_shift_v), vl);         \
        _a = __riscv_vsmul_vv_i32m4(_a, mult_v, 0, vl);                        \
        _a = __riscv_vadd_vv_i32m4(_a, nudge_v, vl);                           \
        _a = __riscv_vsra_vv_i32m4(                                            \
            _a, __riscv_vreinterpret_v_i32m4_u32m4(right_shift_v), vl);        \
        _a              = __riscv_vadd_vx_i32m4(_a, output_offset, vl);        \
        _a              = __riscv_vmax_vx_i32m4(_a, out_act_min, vl);          \
        _a              = __riscv_vmin_vx_i32m4(_a, out_act_max, vl);          \
        vint16m2_t _a16 = __riscv_vnclip_wx_i16m2(_a, 0, 0, vl);               \
        vint8m1_t  _a8  = __riscv_vnclip_wx_i8m1(_a16, 0, 0, vl);              \
        q7_t *_dst = output_data                                               \
                     + (batch * output_height * output_width * output_depth)   \
                     + (out_y * output_width * output_depth)                   \
                     + ((out_x + (X_OFF)) * output_depth) + out_channel_start; \
        __riscv_vse8_v_i8m1(_dst, _a8, vl);                                    \
    } while (0)

                    QUANTIZE_AND_STORE_4X4(acc0, 0);
                    QUANTIZE_AND_STORE_4X4(acc1, 1);
                    QUANTIZE_AND_STORE_4X4(acc2, 2);
                    QUANTIZE_AND_STORE_4X4(acc3, 3);
#undef QUANTIZE_AND_STORE_4X4
                }

                /* --- Remainder X loop (0..3 leftover columns) --- */
                for (; out_x < output_width; out_x++)
                {
                    const int32_t in_x_origin
                        = out_x * stride_width - pad_width;
                    vint32m4_t acc = bias_v;

                    for (int32_t ky = 0; ky < 4; ky++)
                    {
                        const int32_t in_y = in_y_origin + dilation_h * ky;
                        if (in_y < 0 || in_y >= input_height)
                        {
                            continue;
                        }

                        for (int32_t kx = 0; kx < 4; kx++)
                        {
                            const int32_t in_x = in_x_origin + dilation_w * kx;
                            if (in_x < 0 || in_x >= input_width)
                            {
                                continue;
                            }

                            const int8_t *val_ptr
                                = input_data
                                  + (batch * input_height * input_width
                                     * input_depth)
                                  + (in_y * input_width * input_depth);

                            const int8_t *f_ptr
                                = filter_data
                                  + out_channel_start * stride_filter
                                  + ky * 4 * filter_depth + kx * filter_depth;

                            for (int32_t kc = 0; kc < input_depth; kc++)
                            {
                                vint8m1_t w = __riscv_vlse8_v_i8m1(
                                    f_ptr + kc, (size_t)stride_filter, vl);
                                vint16m2_t w16 = __riscv_vsext_vf2_i16m2(w, vl);
                                int16_t    v
                                    = (int16_t)(val_ptr[in_x * input_depth + kc]
                                                + input_offset);
                                acc = __riscv_vwmacc_vx_i32m4(acc, v, w16, vl);
                            }
                        }
                    }

#define QUANTIZE_AND_STORE_4X4_1(ACC_V)                                      \
    do                                                                       \
    {                                                                        \
        vint32m4_t _a = (ACC_V);                                             \
        _a            = __riscv_vsll_vv_i32m4(                               \
            _a, __riscv_vreinterpret_v_i32m4_u32m4(left_shift_v), vl);       \
        _a = __riscv_vsmul_vv_i32m4(_a, mult_v, 0, vl);                      \
        _a = __riscv_vadd_vv_i32m4(_a, nudge_v, vl);                         \
        _a = __riscv_vsra_vv_i32m4(                                          \
            _a, __riscv_vreinterpret_v_i32m4_u32m4(right_shift_v), vl);      \
        _a              = __riscv_vadd_vx_i32m4(_a, output_offset, vl);      \
        _a              = __riscv_vmax_vx_i32m4(_a, out_act_min, vl);        \
        _a              = __riscv_vmin_vx_i32m4(_a, out_act_max, vl);        \
        vint16m2_t _a16 = __riscv_vnclip_wx_i16m2(_a, 0, 0, vl);             \
        vint8m1_t  _a8  = __riscv_vnclip_wx_i8m1(_a16, 0, 0, vl);            \
        q7_t *_dst = output_data                                             \
                     + (batch * output_height * output_width * output_depth) \
                     + (out_y * output_width * output_depth)                 \
                     + (out_x * output_depth) + out_channel_start;           \
        __riscv_vse8_v_i8m1(_dst, _a8, vl);                                  \
    } while (0)

                    QUANTIZE_AND_STORE_4X4_1(acc);
#undef QUANTIZE_AND_STORE_4X4_1
                }
            } /* out_y */
        } /* batch */

        out_channel_start += (int32_t)vl;
    } /* out_channel_start */
}

static void
conv_4_4_16_strideN(const nn_conv_params              *conv_params,
                    const nn_per_channel_quant_params *quant_params,
                    const nn_dims                     *input_dims,
                    const int8_t                      *input_data,
                    const nn_dims                     *filter_dims,
                    const int8_t                      *filter_data,
                    const int32_t                     *bias_data,
                    const nn_dims                     *output_dims,
                    q7_t                              *output_data,
                    int32_t                           *accs_buf,
                    const int32_t                     *output_mult,
                    const uint8_t                     *shift_left,
                    const uint8_t                     *shift_right)
{
    const int32_t batches      = input_dims->n;
    const int16_t input_offset = (int16_t)conv_params->input_offset;
    const int32_t out_offset   = conv_params->output_offset;
    const int32_t out_act_min  = conv_params->activation.min;
    const int32_t out_act_max  = conv_params->activation.max;
    const int32_t stride_w     = conv_params->stride.w;
    const int32_t stride_h     = conv_params->stride.h;
    const int32_t pad_w        = conv_params->padding.w;
    const int32_t pad_h        = conv_params->padding.h;

    const int32_t input_height = input_dims->h;
    const int32_t input_width  = input_dims->w;
    const int32_t input_depth  = input_dims->c;

    const int32_t filter_height = filter_dims->h; /* asserted == 4 */
    const int32_t filter_width  = filter_dims->w; /* asserted == 4 */
    assert(filter_height == 4 && filter_width == 4);
    assert(input_depth <= 16);

    const int32_t output_height = output_dims->h;
    const int32_t output_width  = output_dims->w;
    const int32_t output_depth  = output_dims->c;

    size_t vl = __riscv_vsetvl_e8m1((size_t)input_depth);

    const int32_t row_stride        = input_width * input_depth;
    const int32_t col_stride        = input_depth;
    const int32_t row_step          = stride_h * row_stride;
    const int32_t col_step          = stride_w * col_stride;
    const int32_t filter_row_stride = filter_width * input_depth;
    const int32_t filter_col_stride = input_depth;

    /* Clear accumulator buffer */
    memset(accs_buf,
           0,
           (size_t)(batches * output_height * output_width * output_depth)
               * sizeof(int32_t));

    for (int32_t out_channel = 0; out_channel < output_depth; out_channel++)
    {

        const int8_t *filter_base = filter_data
                                    + offset4(output_depth,
                                              filter_height,
                                              filter_width,
                                              input_depth,
                                              out_channel,
                                              0,
                                              0,
                                              0);

        vint8m1_t fil00 = __riscv_vle8_v_i8m1(filter_base, vl);
        vint8m1_t fil01
            = __riscv_vle8_v_i8m1(filter_base + 1 * filter_col_stride, vl);
        vint8m1_t fil02
            = __riscv_vle8_v_i8m1(filter_base + 2 * filter_col_stride, vl);
        vint8m1_t fil03
            = __riscv_vle8_v_i8m1(filter_base + 3 * filter_col_stride, vl);
        vint8m1_t fil10
            = __riscv_vle8_v_i8m1(filter_base + filter_row_stride, vl);
        vint8m1_t fil11 = __riscv_vle8_v_i8m1(
            filter_base + filter_row_stride + 1 * filter_col_stride, vl);
        vint8m1_t fil12 = __riscv_vle8_v_i8m1(
            filter_base + filter_row_stride + 2 * filter_col_stride, vl);
        vint8m1_t fil13 = __riscv_vle8_v_i8m1(
            filter_base + filter_row_stride + 3 * filter_col_stride, vl);
        vint8m1_t fil20
            = __riscv_vle8_v_i8m1(filter_base + 2 * filter_row_stride, vl);
        vint8m1_t fil21 = __riscv_vle8_v_i8m1(
            filter_base + 2 * filter_row_stride + 1 * filter_col_stride, vl);
        vint8m1_t fil22 = __riscv_vle8_v_i8m1(
            filter_base + 2 * filter_row_stride + 2 * filter_col_stride, vl);
        vint8m1_t fil23 = __riscv_vle8_v_i8m1(
            filter_base + 2 * filter_row_stride + 3 * filter_col_stride, vl);
        vint8m1_t fil30
            = __riscv_vle8_v_i8m1(filter_base + 3 * filter_row_stride, vl);
        vint8m1_t fil31 = __riscv_vle8_v_i8m1(
            filter_base + 3 * filter_row_stride + 1 * filter_col_stride, vl);
        vint8m1_t fil32 = __riscv_vle8_v_i8m1(
            filter_base + 3 * filter_row_stride + 2 * filter_col_stride, vl);
        vint8m1_t fil33 = __riscv_vle8_v_i8m1(
            filter_base + 3 * filter_row_stride + 3 * filter_col_stride, vl);

        for (int32_t batch = 0; batch < batches; batch++)
        {

            const int8_t *batch_base = input_data
                                       + offset4(batches,
                                                 input_height,
                                                 input_width,
                                                 input_depth,
                                                 batch,
                                                 0,
                                                 0,
                                                 0);

            /* Start row pointer at (-pad_h, -pad_w) in the padded coordinate
             * system */
            const int8_t *row_ptr
                = batch_base - pad_h * row_stride - pad_w * col_stride;

            for (int32_t out_y = 0; out_y < output_height; out_y++)
            {
                const int32_t in_y_origin = out_y * stride_h - pad_h;
                const int8_t *base_ptr    = row_ptr;

                for (int32_t out_x = 0; out_x < output_width; out_x++)
                {
                    const int32_t in_x_origin = out_x * stride_w - pad_w;

                    vint32m4_t mul_acc1 = __riscv_vmv_v_x_i32m4(0, vl);

                    const int8_t *in_ptrs[4][4];
                    for (int32_t r = 0; r < 4; r++)
                    {
                        for (int32_t c = 0; c < 4; c++)
                        {
                            in_ptrs[r][c]
                                = base_ptr + r * row_stride + c * col_stride;
                        }
                    }

                    if (in_y_origin >= 0 && in_y_origin + 3 < input_height
                        && in_x_origin >= 0 && in_x_origin + 3 < input_width)
                    {
                        /* Fast path: all 4x4 kernel positions are in-bounds */
                        CONV_MAC(in_ptrs[0][0], fil00, mul_acc1);
                        CONV_MAC(in_ptrs[0][1], fil01, mul_acc1);
                        CONV_MAC(in_ptrs[0][2], fil02, mul_acc1);
                        CONV_MAC(in_ptrs[0][3], fil03, mul_acc1);
                        CONV_MAC(in_ptrs[1][0], fil10, mul_acc1);
                        CONV_MAC(in_ptrs[1][1], fil11, mul_acc1);
                        CONV_MAC(in_ptrs[1][2], fil12, mul_acc1);
                        CONV_MAC(in_ptrs[1][3], fil13, mul_acc1);
                        CONV_MAC(in_ptrs[2][0], fil20, mul_acc1);
                        CONV_MAC(in_ptrs[2][1], fil21, mul_acc1);
                        CONV_MAC(in_ptrs[2][2], fil22, mul_acc1);
                        CONV_MAC(in_ptrs[2][3], fil23, mul_acc1);
                        CONV_MAC(in_ptrs[3][0], fil30, mul_acc1);
                        CONV_MAC(in_ptrs[3][1], fil31, mul_acc1);
                        CONV_MAC(in_ptrs[3][2], fil32, mul_acc1);
                        CONV_MAC(in_ptrs[3][3], fil33, mul_acc1);
                    }
                    else
                    {
                        /* Slow path: per-row and per-column boundary guards */
                        const int rv0 = (in_y_origin + 0 >= 0)
                                        && (in_y_origin + 0 < input_height);
                        const int rv1 = (in_y_origin + 1 >= 0)
                                        && (in_y_origin + 1 < input_height);
                        const int rv2 = (in_y_origin + 2 >= 0)
                                        && (in_y_origin + 2 < input_height);
                        const int rv3 = (in_y_origin + 3 >= 0)
                                        && (in_y_origin + 3 < input_height);
                        const int cv0 = (in_x_origin + 0 >= 0)
                                        && (in_x_origin + 0 < input_width);
                        const int cv1 = (in_x_origin + 1 >= 0)
                                        && (in_x_origin + 1 < input_width);
                        const int cv2 = (in_x_origin + 2 >= 0)
                                        && (in_x_origin + 2 < input_width);
                        const int cv3 = (in_x_origin + 3 >= 0)
                                        && (in_x_origin + 3 < input_width);

                        if (rv0)
                        {
                            if (cv0)
                            {
                                CONV_MAC(in_ptrs[0][0], fil00, mul_acc1);
                            }
                            if (cv1)
                            {
                                CONV_MAC(in_ptrs[0][1], fil01, mul_acc1);
                            }
                            if (cv2)
                            {
                                CONV_MAC(in_ptrs[0][2], fil02, mul_acc1);
                            }
                            if (cv3)
                            {
                                CONV_MAC(in_ptrs[0][3], fil03, mul_acc1);
                            }
                        }
                        if (rv1)
                        {
                            if (cv0)
                            {
                                CONV_MAC(in_ptrs[1][0], fil10, mul_acc1);
                            }
                            if (cv1)
                            {
                                CONV_MAC(in_ptrs[1][1], fil11, mul_acc1);
                            }
                            if (cv2)
                            {
                                CONV_MAC(in_ptrs[1][2], fil12, mul_acc1);
                            }
                            if (cv3)
                            {
                                CONV_MAC(in_ptrs[1][3], fil13, mul_acc1);
                            }
                        }
                        if (rv2)
                        {
                            if (cv0)
                            {
                                CONV_MAC(in_ptrs[2][0], fil20, mul_acc1);
                            }
                            if (cv1)
                            {
                                CONV_MAC(in_ptrs[2][1], fil21, mul_acc1);
                            }
                            if (cv2)
                            {
                                CONV_MAC(in_ptrs[2][2], fil22, mul_acc1);
                            }
                            if (cv3)
                            {
                                CONV_MAC(in_ptrs[2][3], fil23, mul_acc1);
                            }
                        }
                        if (rv3)
                        {
                            if (cv0)
                            {
                                CONV_MAC(in_ptrs[3][0], fil30, mul_acc1);
                            }
                            if (cv1)
                            {
                                CONV_MAC(in_ptrs[3][1], fil31, mul_acc1);
                            }
                            if (cv2)
                            {
                                CONV_MAC(in_ptrs[3][2], fil32, mul_acc1);
                            }
                            if (cv3)
                            {
                                CONV_MAC(in_ptrs[3][3], fil33, mul_acc1);
                            }
                        }
                    }

                    /* Horizontal reduction: sum all input-channel partial
                     * products */
                    int32_t temp_acc = __riscv_vmv_x_s_i32m1_i32(
                        __riscv_vredsum_vs_i32m4_i32m1(
                            mul_acc1, __riscv_vmv_v_x_i32m1(0, 1), vl));

                    accs_buf[offset4(batches,
                                     output_height,
                                     output_width,
                                     output_depth,
                                     batch,
                                     out_y,
                                     out_x,
                                     out_channel)] = temp_acc;

                    base_ptr += col_step;
                }
                row_ptr += row_step;
            }
        }
    }

    /* Requantise and write output */
    postprocess_acc(accs_buf,
                    bias_data,
                    shift_left,
                    output_mult,
                    shift_right,
                    out_offset,
                    out_act_min,
                    out_act_max,
                    output_data,
                    batches * output_height * output_width,
                    output_depth);
}

/* Thin wrapper, matching Conv_4_4_16 in the original. */
static void
conv_4_4_16(const nn_conv_params              *conv_params,
            const nn_per_channel_quant_params *quant_params,
            const nn_dims                     *input_dims,
            const int8_t                      *input_data,
            const nn_dims                     *filter_dims,
            const int8_t                      *filter_data,
            const int32_t                     *bias_data,
            const nn_dims                     *output_dims,
            q7_t                              *output_data,
            int32_t                           *accs_buf,
            const int32_t                     *output_mult,
            const uint8_t                     *shift_left,
            const uint8_t                     *shift_right)
{
    conv_4_4_16_strideN(conv_params,
                        quant_params,
                        input_dims,
                        input_data,
                        filter_dims,
                        filter_data,
                        bias_data,
                        output_dims,
                        output_data,
                        accs_buf,
                        output_mult,
                        shift_left,
                        shift_right);
}

static void
conv_4_4_48_stride1(const nn_conv_params              *conv_params,
                    const nn_per_channel_quant_params *quant_params,
                    const nn_dims                     *input_dims,
                    const int8_t                      *input_data,
                    const nn_dims                     *filter_dims,
                    const int8_t                      *filter_data,
                    const int32_t                     *bias_data,
                    const nn_dims                     *output_dims,
                    q7_t                              *output_data,
                    int32_t                           *accs_buf,
                    const int32_t                     *output_mult,
                    const uint8_t                     *shift_left,
                    const uint8_t                     *shift_right)
{
    const int32_t batches       = input_dims->n;
    const int32_t input_height  = input_dims->h;
    const int32_t input_width   = input_dims->w;
    const int32_t input_depth   = input_dims->c;
    const int32_t output_height = output_dims->h;
    const int32_t output_width  = output_dims->w;
    const int32_t output_depth  = output_dims->c;
    const int32_t out_offset    = conv_params->output_offset;
    const int32_t out_act_min   = conv_params->activation.min;
    const int32_t out_act_max   = conv_params->activation.max;
    const int32_t input_offset  = conv_params->input_offset;
    const int32_t pad_w         = conv_params->padding.w;
    const int32_t pad_h         = conv_params->padding.h;

    const int32_t filter_row_stride = filter_dims->w * input_depth;
    const int32_t filter_col_stride = input_depth;
    const int32_t row_stride        = input_width * input_depth;
    const int32_t col_stride        = input_depth;

    /* Clear accumulator buffer */
    memset(accs_buf,
           0,
           (size_t)(batches * output_height * output_width * output_depth)
               * sizeof(int32_t));

    for (int32_t batch = 0; batch < batches; batch++)
    {
        for (int32_t out_channel = 0; out_channel < output_depth; out_channel++)
        {

            const int8_t *filter_base
                = filter_data
                  + offset4(
                      output_depth, 4, 4, input_depth, out_channel, 0, 0, 0);

            int32_t rem_channels = input_depth;
            int32_t num_chunks   = (input_depth + 15) / 16;

            for (int32_t chunk = 0; chunk < num_chunks; chunk++)
            {

                const int8_t *chunk_ptr = filter_base + chunk * 16;
                size_t        vl        = __riscv_vsetvl_e8m1(
                    (size_t)(rem_channels > 16 ? 16 : rem_channels));
                rem_channels -= 16;

                /* Pin this chunk's 4x4 filter tiles */
                vint8m1_t fil00 = __riscv_vle8_v_i8m1(chunk_ptr, vl);
                vint8m1_t fil01 = __riscv_vle8_v_i8m1(
                    chunk_ptr + 1 * filter_col_stride, vl);
                vint8m1_t fil02 = __riscv_vle8_v_i8m1(
                    chunk_ptr + 2 * filter_col_stride, vl);
                vint8m1_t fil03 = __riscv_vle8_v_i8m1(
                    chunk_ptr + 3 * filter_col_stride, vl);
                vint8m1_t fil10
                    = __riscv_vle8_v_i8m1(chunk_ptr + filter_row_stride, vl);
                vint8m1_t fil11 = __riscv_vle8_v_i8m1(
                    chunk_ptr + filter_row_stride + 1 * filter_col_stride, vl);
                vint8m1_t fil12 = __riscv_vle8_v_i8m1(
                    chunk_ptr + filter_row_stride + 2 * filter_col_stride, vl);
                vint8m1_t fil13 = __riscv_vle8_v_i8m1(
                    chunk_ptr + filter_row_stride + 3 * filter_col_stride, vl);
                vint8m1_t fil20 = __riscv_vle8_v_i8m1(
                    chunk_ptr + 2 * filter_row_stride, vl);
                vint8m1_t fil21 = __riscv_vle8_v_i8m1(
                    chunk_ptr + 2 * filter_row_stride + 1 * filter_col_stride,
                    vl);
                vint8m1_t fil22 = __riscv_vle8_v_i8m1(
                    chunk_ptr + 2 * filter_row_stride + 2 * filter_col_stride,
                    vl);
                vint8m1_t fil23 = __riscv_vle8_v_i8m1(
                    chunk_ptr + 2 * filter_row_stride + 3 * filter_col_stride,
                    vl);
                vint8m1_t fil30 = __riscv_vle8_v_i8m1(
                    chunk_ptr + 3 * filter_row_stride, vl);
                vint8m1_t fil31 = __riscv_vle8_v_i8m1(
                    chunk_ptr + 3 * filter_row_stride + 1 * filter_col_stride,
                    vl);
                vint8m1_t fil32 = __riscv_vle8_v_i8m1(
                    chunk_ptr + 3 * filter_row_stride + 2 * filter_col_stride,
                    vl);
                vint8m1_t fil33 = __riscv_vle8_v_i8m1(
                    chunk_ptr + 3 * filter_row_stride + 3 * filter_col_stride,
                    vl);

                const int8_t *base_ptr = input_data
                                         + offset4(batches,
                                                   input_height,
                                                   input_width,
                                                   input_depth,
                                                   batch,
                                                   0,
                                                   0,
                                                   chunk * 16);

                for (int32_t out_y = 0; out_y < output_height; out_y++)
                {
                    const int32_t in_y_origin = out_y - pad_h;
                    const int8_t *row_ptr = base_ptr + in_y_origin * row_stride;

                    for (int32_t out_x = 0; out_x < output_width; out_x += 2)
                    {
                        const int32_t in_x_origin1 = out_x - pad_w;
                        const int32_t in_x_origin2 = out_x + 1 - pad_w;

                        const int8_t *curr_ptr
                            = row_ptr + in_x_origin1 * col_stride;

                        const int8_t *in_ptrs[4][5];
                        for (int32_t r = 0; r < 4; r++)
                        {
                            for (int32_t c = 0; c < 5; c++)
                            {
                                in_ptrs[r][c] = curr_ptr + r * row_stride
                                                + c * col_stride;
                            }
                        }

                        /*
                         * mul_acc1 / mul_acc2 are the fixed names expected by
                         * CONV_MAC_2X (matching the original macro convention).
                         */
                        vint32m4_t mul_acc1 = __riscv_vmv_v_x_i32m4(0, 16);
                        vint32m4_t mul_acc2 = __riscv_vmv_v_x_i32m4(0, 16);

                        if (in_y_origin >= 0 && in_y_origin + 3 < input_height
                            && in_x_origin1 >= 0
                            && in_x_origin2 + 3 < input_width)
                        {
                            /* Fast path */
                            CONV_MAC(in_ptrs[0][0], fil00, mul_acc1);
                            CONV_MAC_2X(in_ptrs[0][1], fil01, fil00);
                            CONV_MAC_2X(in_ptrs[0][2], fil02, fil01);
                            CONV_MAC_2X(in_ptrs[0][3], fil03, fil02);
                            CONV_MAC(in_ptrs[0][4], fil03, mul_acc2);

                            CONV_MAC(in_ptrs[1][0], fil10, mul_acc1);
                            CONV_MAC_2X(in_ptrs[1][1], fil11, fil10);
                            CONV_MAC_2X(in_ptrs[1][2], fil12, fil11);
                            CONV_MAC_2X(in_ptrs[1][3], fil13, fil12);
                            CONV_MAC(in_ptrs[1][4], fil13, mul_acc2);

                            CONV_MAC(in_ptrs[2][0], fil20, mul_acc1);
                            CONV_MAC_2X(in_ptrs[2][1], fil21, fil20);
                            CONV_MAC_2X(in_ptrs[2][2], fil22, fil21);
                            CONV_MAC_2X(in_ptrs[2][3], fil23, fil22);
                            CONV_MAC(in_ptrs[2][4], fil23, mul_acc2);

                            CONV_MAC(in_ptrs[3][0], fil30, mul_acc1);
                            CONV_MAC_2X(in_ptrs[3][1], fil31, fil30);
                            CONV_MAC_2X(in_ptrs[3][2], fil32, fil31);
                            CONV_MAC_2X(in_ptrs[3][3], fil33, fil32);
                            CONV_MAC(in_ptrs[3][4], fil33, mul_acc2);
                        }
                        else
                        {
                            /* Slow path: separate guards per pixel */
                            const int rv[4]
                                = { (in_y_origin + 0 >= 0)
                                        && (in_y_origin + 0 < input_height),
                                    (in_y_origin + 1 >= 0)
                                        && (in_y_origin + 1 < input_height),
                                    (in_y_origin + 2 >= 0)
                                        && (in_y_origin + 2 < input_height),
                                    (in_y_origin + 3 >= 0)
                                        && (in_y_origin + 3 < input_height) };
                            const int cv1[4]
                                = { (in_x_origin1 + 0 >= 0)
                                        && (in_x_origin1 + 0 < input_width),
                                    (in_x_origin1 + 1 >= 0)
                                        && (in_x_origin1 + 1 < input_width),
                                    (in_x_origin1 + 2 >= 0)
                                        && (in_x_origin1 + 2 < input_width),
                                    (in_x_origin1 + 3 >= 0)
                                        && (in_x_origin1 + 3 < input_width) };
                            const int cv2[4]
                                = { (in_x_origin2 + 0 >= 0)
                                        && (in_x_origin2 + 0 < input_width),
                                    (in_x_origin2 + 1 >= 0)
                                        && (in_x_origin2 + 1 < input_width),
                                    (in_x_origin2 + 2 >= 0)
                                        && (in_x_origin2 + 2 < input_width),
                                    (in_x_origin2 + 3 >= 0)
                                        && (in_x_origin2 + 3 < input_width) };

                            const int8_t *in_ptrs1[4][4], *in_ptrs2[4][4];
                            for (int32_t r = 0; r < 4; r++)
                            {
                                for (int32_t c = 0; c < 4; c++)
                                {
                                    in_ptrs1[r][c] = curr_ptr + r * row_stride
                                                     + c * col_stride;
                                    in_ptrs2[r][c]
                                        = in_ptrs1[r][c] + col_stride;
                                }
                            }

                            if (rv[0])
                            {
                                if (cv1[0])
                                {
                                    CONV_MAC(in_ptrs1[0][0], fil00, mul_acc1);
                                }
                                if (cv1[1])
                                {
                                    CONV_MAC(in_ptrs1[0][1], fil01, mul_acc1);
                                }
                                if (cv1[2])
                                {
                                    CONV_MAC(in_ptrs1[0][2], fil02, mul_acc1);
                                }
                                if (cv1[3])
                                {
                                    CONV_MAC(in_ptrs1[0][3], fil03, mul_acc1);
                                }
                            }
                            if (rv[1])
                            {
                                if (cv1[0])
                                {
                                    CONV_MAC(in_ptrs1[1][0], fil10, mul_acc1);
                                }
                                if (cv1[1])
                                {
                                    CONV_MAC(in_ptrs1[1][1], fil11, mul_acc1);
                                }
                                if (cv1[2])
                                {
                                    CONV_MAC(in_ptrs1[1][2], fil12, mul_acc1);
                                }
                                if (cv1[3])
                                {
                                    CONV_MAC(in_ptrs1[1][3], fil13, mul_acc1);
                                }
                            }
                            if (rv[2])
                            {
                                if (cv1[0])
                                {
                                    CONV_MAC(in_ptrs1[2][0], fil20, mul_acc1);
                                }
                                if (cv1[1])
                                {
                                    CONV_MAC(in_ptrs1[2][1], fil21, mul_acc1);
                                }
                                if (cv1[2])
                                {
                                    CONV_MAC(in_ptrs1[2][2], fil22, mul_acc1);
                                }
                                if (cv1[3])
                                {
                                    CONV_MAC(in_ptrs1[2][3], fil23, mul_acc1);
                                }
                            }
                            if (rv[3])
                            {
                                if (cv1[0])
                                {
                                    CONV_MAC(in_ptrs1[3][0], fil30, mul_acc1);
                                }
                                if (cv1[1])
                                {
                                    CONV_MAC(in_ptrs1[3][1], fil31, mul_acc1);
                                }
                                if (cv1[2])
                                {
                                    CONV_MAC(in_ptrs1[3][2], fil32, mul_acc1);
                                }
                                if (cv1[3])
                                {
                                    CONV_MAC(in_ptrs1[3][3], fil33, mul_acc1);
                                }
                            }

                            if (rv[0])
                            {
                                if (cv2[0])
                                {
                                    CONV_MAC(in_ptrs2[0][0], fil00, mul_acc2);
                                }
                                if (cv2[1])
                                {
                                    CONV_MAC(in_ptrs2[0][1], fil01, mul_acc2);
                                }
                                if (cv2[2])
                                {
                                    CONV_MAC(in_ptrs2[0][2], fil02, mul_acc2);
                                }
                                if (cv2[3])
                                {
                                    CONV_MAC(in_ptrs2[0][3], fil03, mul_acc2);
                                }
                            }
                            if (rv[1])
                            {
                                if (cv2[0])
                                {
                                    CONV_MAC(in_ptrs2[1][0], fil10, mul_acc2);
                                }
                                if (cv2[1])
                                {
                                    CONV_MAC(in_ptrs2[1][1], fil11, mul_acc2);
                                }
                                if (cv2[2])
                                {
                                    CONV_MAC(in_ptrs2[1][2], fil12, mul_acc2);
                                }
                                if (cv2[3])
                                {
                                    CONV_MAC(in_ptrs2[1][3], fil13, mul_acc2);
                                }
                            }
                            if (rv[2])
                            {
                                if (cv2[0])
                                {
                                    CONV_MAC(in_ptrs2[2][0], fil20, mul_acc2);
                                }
                                if (cv2[1])
                                {
                                    CONV_MAC(in_ptrs2[2][1], fil21, mul_acc2);
                                }
                                if (cv2[2])
                                {
                                    CONV_MAC(in_ptrs2[2][2], fil22, mul_acc2);
                                }
                                if (cv2[3])
                                {
                                    CONV_MAC(in_ptrs2[2][3], fil23, mul_acc2);
                                }
                            }
                            if (rv[3])
                            {
                                if (cv2[0])
                                {
                                    CONV_MAC(in_ptrs2[3][0], fil30, mul_acc2);
                                }
                                if (cv2[1])
                                {
                                    CONV_MAC(in_ptrs2[3][1], fil31, mul_acc2);
                                }
                                if (cv2[2])
                                {
                                    CONV_MAC(in_ptrs2[3][2], fil32, mul_acc2);
                                }
                                if (cv2[3])
                                {
                                    CONV_MAC(in_ptrs2[3][3], fil33, mul_acc2);
                                }
                            }
                        }

                        /* Reduce and accumulate into buffer */
                        int32_t acc1_val = __riscv_vmv_x_s_i32m1_i32(
                            __riscv_vredsum_vs_i32m4_i32m1(
                                mul_acc1, __riscv_vmv_v_x_i32m1(0, 1), vl));
                        int32_t acc2_val = __riscv_vmv_x_s_i32m1_i32(
                            __riscv_vredsum_vs_i32m4_i32m1(
                                mul_acc2, __riscv_vmv_v_x_i32m1(0, 1), vl));

                        int32_t idx1 = offset4(batches,
                                               output_height,
                                               output_width,
                                               output_depth,
                                               batch,
                                               out_y,
                                               out_x,
                                               out_channel);
                        int32_t idx2 = offset4(batches,
                                               output_height,
                                               output_width,
                                               output_depth,
                                               batch,
                                               out_y,
                                               out_x + 1,
                                               out_channel);

                        if (chunk == 0)
                        {
                            accs_buf[idx1] = acc1_val;
                            if (out_x + 1 < output_width)
                            {
                                accs_buf[idx2] = acc2_val;
                            }
                        }
                        else
                        {
                            accs_buf[idx1] += acc1_val;
                            if (out_x + 1 < output_width)
                            {
                                accs_buf[idx2] += acc2_val;
                            }
                        }
                    }
                }
            }
        }
    }

    postprocess_acc(accs_buf,
                    bias_data,
                    shift_left,
                    output_mult,
                    shift_right,
                    out_offset,
                    out_act_min,
                    out_act_max,
                    output_data,
                    batches * output_height * output_width,
                    output_depth);
}

static void
conv_4x4_oc_vectorized(const nn_conv_params              *conv_params,
                       const nn_per_channel_quant_params *quant_params,
                       const nn_dims                     *input_dims,
                       const int8_t                      *input_data,
                       const nn_dims                     *filter_dims,
                       const int32_t                     *bias_data,
                       const nn_dims                     *output_dims,
                       q7_t                              *output_data,
                       const int16_t                     *repacked_weights,
                       const int32_t                     *weight_sums,
                       int8_t                            *tiled_input_buffer)
{
    const int32_t output_height = output_dims->h;
    const int32_t output_width  = output_dims->w;
    const int32_t output_depth  = output_dims->c;
    const int32_t input_depth   = input_dims->c;
    const int32_t batches       = input_dims->n;

    const int32_t stride_w      = conv_params->stride.w;
    const int32_t stride_h      = conv_params->stride.h;
    const int32_t pad_w         = conv_params->padding.w;
    const int32_t pad_h         = conv_params->padding.h;
    const int32_t out_offset    = conv_params->output_offset;
    const int32_t input_off_val = conv_params->input_offset;

    /* Pre-adjust bias: bias_adj[i] = bias[i] + input_offset * weight_sum[i] */
    int32_t adjusted_bias[MAX_OUTPUT_DEPTH];
    for (int32_t i = 0; i < output_depth; i++)
    {
        adjusted_bias[i] = bias_data[i] + input_off_val * weight_sums[i];
    }

    const int32_t padded_width        = input_dims->w + 8; /* 2 * margin_w */
    const int32_t stride_padded_bytes = padded_width * input_depth;
    /* Offset from buffer start to pixel (0,0): margin_h rows + margin_w pixels
     */
    const int32_t pad_margin_offset = 4 * stride_padded_bytes + 4 * input_depth;

    const int32_t s_step = stride_w * input_depth; /* bytes per output-X step */

    for (int32_t batch = 0; batch < batches; batch++)
    {

        const int8_t *batch_src
            = input_data + batch * input_dims->h * input_dims->w * input_depth;

        tiled_pad_input(tiled_input_buffer,
                        batch_src,
                        input_dims->h,
                        input_dims->w,
                        input_depth,
                        input_off_val);

        const int8_t *input_base = tiled_input_buffer + pad_margin_offset;

        for (int32_t oc_block = 0; oc_block < output_depth;)
        {

            size_t vl = __riscv_vsetvl_e32m4((size_t)(output_depth - oc_block));

            /* Load bias and quantisation parameters for this OC block */
            vint32m4_t bias_v
                = __riscv_vle32_v_i32m4(adjusted_bias + oc_block, vl);
            vint32m4_t mult_v = __riscv_vle32_v_i32m4(
                quant_params->multiplier + oc_block, vl);
            vint32m4_t shift_v
                = __riscv_vle32_v_i32m4(quant_params->shift + oc_block, vl);

            vint32m4_t left_shift_v  = __riscv_vmax_vx_i32m4(shift_v, 0, vl);
            vint32m4_t right_shift_v = __riscv_vmax_vx_i32m4(
                __riscv_vneg_v_i32m4(shift_v, vl), 0, vl);

            vint32m4_t shift_m1_v = __riscv_vsub_vx_i32m4(right_shift_v, 1, vl);
            vbool8_t mask_gt0 = __riscv_vmsgt_vx_i32m4_b8(right_shift_v, 0, vl);
            vint32m4_t nudge_v = __riscv_vmv_v_x_i32m4(0, vl);
            nudge_v            = __riscv_vsll_vv_i32m4_mu(
                mask_gt0,
                nudge_v,
                __riscv_vmv_v_x_i32m4(1, vl),
                __riscv_vreinterpret_v_i32m4_u32m4(shift_m1_v),
                vl);

            const int32_t out_act_min = conv_params->activation.min;
            const int32_t out_act_max = conv_params->activation.max;

            for (int32_t out_y = 0; out_y < output_height; out_y++)
            {

                const int32_t in_y_origin = out_y * stride_h - pad_h;
                const int8_t *row_ptr_base
                    = input_base + in_y_origin * stride_padded_bytes;

                const int8_t *r0 = row_ptr_base;
                const int8_t *r1 = row_ptr_base + stride_padded_bytes;
                const int8_t *r2 = row_ptr_base + stride_padded_bytes * 2;
                const int8_t *r3 = row_ptr_base + stride_padded_bytes * 3;

                int32_t out_x = 0;

                /* --- 4-pixel-wide unrolled X loop --- */
                for (; out_x <= output_width - 4; out_x += 4)
                {

                    vint32m4_t acc0 = bias_v;
                    vint32m4_t acc1 = bias_v;
                    vint32m4_t acc2 = bias_v;
                    vint32m4_t acc3 = bias_v;

                    const int32_t in_x_offset
                        = (out_x * stride_w - pad_w) * input_depth;

                    /* Weights for this OC block, all 4x4 spatial positions */
                    const int16_t *w_ptr
                        = repacked_weights + oc_block * 4 * 4 * input_depth;

                    const int8_t *p0_base = r0 + in_x_offset;
                    const int8_t *p1_base = r1 + in_x_offset;
                    const int8_t *p2_base = r2 + in_x_offset;
                    const int8_t *p3_base = r3 + in_x_offset;

                    for (int32_t ky = 0; ky < 4; ky++)
                    {
                        const int8_t *p0 = (ky == 0)   ? p0_base
                                           : (ky == 1) ? p1_base
                                           : (ky == 2) ? p2_base
                                                       : p3_base;

                        const int8_t *p_sp0 = p0;
                        const int8_t *p_sp1 = p0 + s_step;
                        const int8_t *p_sp2 = p0 + s_step * 2;
                        const int8_t *p_sp3 = p0 + s_step * 3;

                        for (int32_t kx = 0; kx < 4; kx++)
                        {
                            /* Unroll input-channel loop by 4 */
                            for (int32_t ic = 0; ic < input_depth; ic += 4)
                            {
                                vint16m2_t w_0
                                    = __riscv_vle16_v_i16m2(w_ptr, vl);
                                vint16m2_t w_1
                                    = __riscv_vle16_v_i16m2(w_ptr + vl, vl);
                                vint16m2_t w_2
                                    = __riscv_vle16_v_i16m2(w_ptr + 2 * vl, vl);
                                vint16m2_t w_3
                                    = __riscv_vle16_v_i16m2(w_ptr + 3 * vl, vl);
                                w_ptr += 4 * vl;

                                acc0 = __riscv_vwmacc_vx_i32m4(
                                    acc0, *p_sp0++, w_0, vl);
                                acc1 = __riscv_vwmacc_vx_i32m4(
                                    acc1, *p_sp1++, w_0, vl);
                                acc2 = __riscv_vwmacc_vx_i32m4(
                                    acc2, *p_sp2++, w_0, vl);
                                acc3 = __riscv_vwmacc_vx_i32m4(
                                    acc3, *p_sp3++, w_0, vl);

                                acc0 = __riscv_vwmacc_vx_i32m4(
                                    acc0, *p_sp0++, w_1, vl);
                                acc1 = __riscv_vwmacc_vx_i32m4(
                                    acc1, *p_sp1++, w_1, vl);
                                acc2 = __riscv_vwmacc_vx_i32m4(
                                    acc2, *p_sp2++, w_1, vl);
                                acc3 = __riscv_vwmacc_vx_i32m4(
                                    acc3, *p_sp3++, w_1, vl);

                                acc0 = __riscv_vwmacc_vx_i32m4(
                                    acc0, *p_sp0++, w_2, vl);
                                acc1 = __riscv_vwmacc_vx_i32m4(
                                    acc1, *p_sp1++, w_2, vl);
                                acc2 = __riscv_vwmacc_vx_i32m4(
                                    acc2, *p_sp2++, w_2, vl);
                                acc3 = __riscv_vwmacc_vx_i32m4(
                                    acc3, *p_sp3++, w_2, vl);

                                acc0 = __riscv_vwmacc_vx_i32m4(
                                    acc0, *p_sp0++, w_3, vl);
                                acc1 = __riscv_vwmacc_vx_i32m4(
                                    acc1, *p_sp1++, w_3, vl);
                                acc2 = __riscv_vwmacc_vx_i32m4(
                                    acc2, *p_sp2++, w_3, vl);
                                acc3 = __riscv_vwmacc_vx_i32m4(
                                    acc3, *p_sp3++, w_3, vl);
                            }
                        }
                    }

/* Quantise and store one accumulator for this oc_block */
#define QUANT_STORE(acc, idx)                                                \
    do                                                                       \
    {                                                                        \
        (acc) = __riscv_vsll_vv_i32m4(                                       \
            (acc), __riscv_vreinterpret_v_i32m4_u32m4(left_shift_v), vl);    \
        (acc) = __riscv_vsmul_vv_i32m4((acc), mult_v, 0, vl);                \
        (acc) = __riscv_vadd_vv_i32m4((acc), nudge_v, vl);                   \
        (acc) = __riscv_vsra_vv_i32m4(                                       \
            (acc), __riscv_vreinterpret_v_i32m4_u32m4(right_shift_v), vl);   \
        (acc)           = __riscv_vadd_vx_i32m4((acc), out_offset, vl);      \
        (acc)           = __riscv_vmax_vx_i32m4((acc), out_act_min, vl);     \
        (acc)           = __riscv_vmin_vx_i32m4((acc), out_act_max, vl);     \
        vint16m2_t _a16 = __riscv_vnclip_wx_i16m2((acc), 0, 0, vl);          \
        vint8m1_t  _a8  = __riscv_vnclip_wx_i8m1(_a16, 0, 0, vl);            \
        q7_t *_dst = output_data                                             \
                     + (batch * output_height * output_width * output_depth) \
                     + (out_y * output_width * output_depth)                 \
                     + ((out_x + (idx)) * output_depth) + oc_block;          \
        __riscv_vse8_v_i8m1(_dst, _a8, vl);                                  \
    } while (0)

                    QUANT_STORE(acc0, 0);
                    QUANT_STORE(acc1, 1);
                    QUANT_STORE(acc2, 2);
                    QUANT_STORE(acc3, 3);
#undef QUANT_STORE
                }

                /* --- Remainder columns --- */
                for (; out_x < output_width; out_x++)
                {
                    vint32m4_t acc = bias_v;

                    const int32_t in_x_offset
                        = (out_x * stride_w - pad_w) * input_depth;
                    const int16_t *w_ptr
                        = repacked_weights + oc_block * 4 * 4 * input_depth;

                    const int8_t *r_ptrs[4] = { r0 + in_x_offset,
                                                r1 + in_x_offset,
                                                r2 + in_x_offset,
                                                r3 + in_x_offset };

                    for (int32_t ky = 0; ky < 4; ky++)
                    {
                        const int8_t *p = r_ptrs[ky];
                        for (int32_t kx = 0; kx < 4; kx++)
                        {
                            for (int32_t ic = 0; ic < input_depth; ic += 4)
                            {
                                vint16m2_t w_0
                                    = __riscv_vle16_v_i16m2(w_ptr, vl);
                                vint16m2_t w_1
                                    = __riscv_vle16_v_i16m2(w_ptr + vl, vl);
                                vint16m2_t w_2
                                    = __riscv_vle16_v_i16m2(w_ptr + 2 * vl, vl);
                                vint16m2_t w_3
                                    = __riscv_vle16_v_i16m2(w_ptr + 3 * vl, vl);
                                w_ptr += 4 * vl;
                                acc = __riscv_vwmacc_vx_i32m4(
                                    acc, *p++, w_0, vl);
                                acc = __riscv_vwmacc_vx_i32m4(
                                    acc, *p++, w_1, vl);
                                acc = __riscv_vwmacc_vx_i32m4(
                                    acc, *p++, w_2, vl);
                                acc = __riscv_vwmacc_vx_i32m4(
                                    acc, *p++, w_3, vl);
                            }
                        }
                    }

#define QUANT_STORE_1(acc)                                                   \
    do                                                                       \
    {                                                                        \
        (acc) = __riscv_vsll_vv_i32m4(                                       \
            (acc), __riscv_vreinterpret_v_i32m4_u32m4(left_shift_v), vl);    \
        (acc) = __riscv_vsmul_vv_i32m4((acc), mult_v, 0, vl);                \
        (acc) = __riscv_vadd_vv_i32m4((acc), nudge_v, vl);                   \
        (acc) = __riscv_vsra_vv_i32m4(                                       \
            (acc), __riscv_vreinterpret_v_i32m4_u32m4(right_shift_v), vl);   \
        (acc)           = __riscv_vadd_vx_i32m4((acc), out_offset, vl);      \
        (acc)           = __riscv_vmax_vx_i32m4((acc), out_act_min, vl);     \
        (acc)           = __riscv_vmin_vx_i32m4((acc), out_act_max, vl);     \
        vint16m2_t _a16 = __riscv_vnclip_wx_i16m2((acc), 0, 0, vl);          \
        vint8m1_t  _a8  = __riscv_vnclip_wx_i8m1(_a16, 0, 0, vl);            \
        q7_t *_dst = output_data                                             \
                     + (batch * output_height * output_width * output_depth) \
                     + (out_y * output_width * output_depth)                 \
                     + (out_x * output_depth) + oc_block;                    \
        __riscv_vse8_v_i8m1(_dst, _a8, vl);                                  \
    } while (0)

                    QUANT_STORE_1(acc);
#undef QUANT_STORE_1
                }
            } /* out_y */

            oc_block += (int32_t)vl;
        } /* oc_block */
    } /* batch */
}

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
    (void)bias_dims; /* unused; output_depth is read from filter_dims->n */

    const int32_t batches       = input_dims->n;
    const int32_t input_height  = input_dims->h;
    const int32_t input_width   = input_dims->w;
    const int32_t input_depth   = input_dims->c;
    const int32_t filter_height = filter_dims->h;
    const int32_t filter_width  = filter_dims->w;
    const int32_t output_depth  = output_dims->c;
    const int32_t output_height = output_dims->h;
    const int32_t output_width  = output_dims->w;

    uint8_t *buf = (uint8_t *)ctx->buf;

    /* 1. Accumulator buffer (always required) */
    int32_t *accs_buf = (int32_t *)(void *)buf;
    buf               = align_ptr(
        buf
            + (size_t)(batches * output_height * output_width * output_depth)
                  * sizeof(int32_t),
        16);

    /* 2. OC-vectorised repacked weights + weight sums + tiled input
          (only for 4x4 & input_depth%4==0 & output_depth<=MAX_OUTPUT_DEPTH) */
    int16_t *repacked_weights = NULL;
    int32_t *weight_sums      = NULL;
    int8_t  *tiled_input_buf  = NULL;

    const int use_oc_vec
        = (filter_height == 4 && filter_width == 4 && (input_depth % 4) == 0
           && output_depth <= MAX_OUTPUT_DEPTH);
    if (use_oc_vec)
    {
        repacked_weights = (int16_t *)(void *)buf;
        buf = align_ptr(buf
                            + (size_t)(output_depth * 4 * 4 * input_depth)
                                  * sizeof(int16_t),
                        16);

        weight_sums = (int32_t *)(void *)buf;
        buf = align_ptr(buf + (size_t)output_depth * sizeof(int32_t), 16);

        const int32_t padded_w = input_width + 8;
        const int32_t padded_h = input_height + 8;
        tiled_input_buf        = (int8_t *)(void *)buf;
        buf = align_ptr(buf + (size_t)(padded_h * padded_w * input_depth), 16);
    }

    /* 3. Generic tiled buffer and repacked generic weights (for 4x4 fallback)
     */
    int8_t *generic_tiled_buf = NULL;
    int8_t *repacked_wgen     = NULL;

    const int use_4x4 = (filter_height == 4 && filter_width == 4);
    if (use_4x4)
    {
        generic_tiled_buf = (int8_t *)(void *)buf;
        buf               = align_ptr(buf + INPUT_BUFFER_SIZE, 16);

        repacked_wgen = (int8_t *)(void *)buf;
        buf = align_ptr(buf + (size_t)(output_depth * 4 * 4 * input_depth), 16);
    }

    /* 4. Per-channel shift arrays */
    uint8_t *shift_left  = buf;
    buf                  = align_ptr(buf + (size_t)output_depth, 16);
    uint8_t *shift_right = buf;
    buf                  = align_ptr(buf + (size_t)output_depth, 16);

    /* 5. Aligned copies of filter and bias */
    int8_t       *filter_copy = (int8_t *)(void *)align_ptr(buf, 16);
    const int32_t filter_flat
        = filter_height * filter_width * input_depth * output_depth;
    buf = align_ptr((uint8_t *)filter_copy + (size_t)filter_flat, 16);

    int32_t *bias_copy
        = (bias_data) ? (int32_t *)(void *)align_ptr(buf, 16) : NULL;
    memcpy(filter_copy, filter_data, (size_t)filter_flat);
    if (bias_data && bias_copy)
    {
        memcpy(bias_copy, bias_data, (size_t)output_depth * sizeof(int32_t));
    }

    prepare_shift_params(
        shift_left, shift_right, quant_params->shift, output_depth);

    /* Build repacked generic weights for conv2d_4x4 (oc-innermost layout) */
    if (use_4x4 && repacked_wgen)
    {
        int8_t *dst = repacked_wgen;
        for (int32_t ky = 0; ky < filter_height; ky++)
        {
            for (int32_t kx = 0; kx < filter_width; kx++)
            {
                for (int32_t ic = 0; ic < input_depth; ic++)
                {
                    for (int32_t oc = 0; oc < output_depth; oc++)
                    {
                        int32_t src_idx
                            = oc * (filter_height * filter_width * input_depth)
                              + ky * (filter_width * input_depth)
                              + kx * input_depth + ic;
                        *dst++ = filter_copy[src_idx];
                    }
                }
            }
        }
    }

    /* Build int16 repacked weights + weight sums for conv_4x4_oc_vectorized */
    if (use_oc_vec && repacked_weights && weight_sums)
    {
        repack_weights_d48(filter_copy,
                           repacked_weights,
                           weight_sums,
                           output_depth,
                           filter_height,
                           filter_width,
                           input_depth);
    }

    if (filter_height == 4 && filter_width == 4 && (input_depth % 4) == 0
        && repacked_weights != NULL)
    {

        /* Best path: full OC-vectorised kernel with int16 weights */
        conv_4x4_oc_vectorized(conv_params,
                               quant_params,
                               input_dims,
                               input_data,
                               filter_dims,
                               bias_copy ? bias_copy : bias_data,
                               output_dims,
                               output_data,
                               repacked_weights,
                               weight_sums,
                               tiled_input_buf);
    }
    else if (filter_height == 4 && filter_width == 4 && repacked_wgen != NULL
             && output_depth >= 32)
    {

        /* Generic 4x4 kernel preferred for large OC with generic repacked
         * weights */
        conv2d_4x4(conv_params,
                   quant_params,
                   input_dims,
                   input_data,
                   filter_dims,
                   filter_copy,
                   bias_copy ? bias_copy : bias_data,
                   output_dims,
                   output_data,
                   repacked_wgen,
                   generic_tiled_buf,
                   shift_left,
                   shift_right);
    }
    else if (filter_height == 4 && filter_width == 4 && input_depth <= 16)
    {

        /* 4x4 x 16-channel specialisation */
        conv_4_4_16(conv_params,
                    quant_params,
                    input_dims,
                    input_data,
                    filter_dims,
                    filter_copy,
                    bias_copy ? bias_copy : bias_data,
                    output_dims,
                    output_data,
                    accs_buf,
                    quant_params->multiplier,
                    shift_left,
                    shift_right);
    }
    else if (filter_height == 4 && filter_width == 4 && input_depth <= 48
             && conv_params->stride.w == 1 && conv_params->stride.h == 1)
    {

        /* 4x4 x 48-channel stride-1 specialisation */
        conv_4_4_48_stride1(conv_params,
                            quant_params,
                            input_dims,
                            input_data,
                            filter_dims,
                            filter_copy,
                            bias_copy ? bias_copy : bias_data,
                            output_dims,
                            output_data,
                            accs_buf,
                            quant_params->multiplier,
                            shift_left,
                            shift_right);
    }
    else if (filter_height == 4 && filter_width == 4 && repacked_wgen != NULL)
    {

        /* Generic 4x4 with repacked weights (smaller OC) */
        conv2d_4x4(conv_params,
                   quant_params,
                   input_dims,
                   input_data,
                   filter_dims,
                   filter_copy,
                   bias_copy ? bias_copy : bias_data,
                   output_dims,
                   output_data,
                   repacked_wgen,
                   generic_tiled_buf,
                   shift_left,
                   shift_right);
    }
    else
    {

        /* Fallback: reference im2col + GEMM kernel from document 2 */
        q15_t *buffer_a = (q15_t *)(void *)
            accs_buf; /* reuse accs_buf as im2col workspace */

        const int32_t  input_offset = conv_params->input_offset;
        const int32_t  out_offset   = conv_params->output_offset;
        const int32_t  out_act_min  = conv_params->activation.min;
        const int32_t  out_act_max  = conv_params->activation.max;
        const int32_t  stride_x     = conv_params->stride.w;
        const int32_t  stride_y     = conv_params->stride.h;
        const int32_t  pad_x        = conv_params->padding.w;
        const int32_t  pad_y        = conv_params->padding.h;
        const uint16_t dilation_x   = (uint16_t)conv_params->dilation.w;
        const uint16_t dilation_y   = (uint16_t)conv_params->dilation.h;
        const int32_t *output_mult  = quant_params->multiplier;
        const int32_t *output_shift = quant_params->shift;

        for (int32_t i_batch = 0; i_batch < batches; i_batch++)
        {

            q15_t *two_column_buf = buffer_a;
            q7_t  *out            = output_data;

            for (int32_t i_out_y = 0; i_out_y < output_height; i_out_y++)
            {
                for (int32_t i_out_x = 0; i_out_x < output_width; i_out_x++)
                {

                    const int32_t base_idx_y = stride_y * i_out_y - pad_y;
                    const int32_t base_idx_x = stride_x * i_out_x - pad_x;

                    for (int32_t i_ker_y = 0; i_ker_y < filter_height;
                         i_ker_y++)
                    {
                        for (int32_t i_ker_x = 0; i_ker_x < filter_width;
                             i_ker_x++)
                        {

                            const int32_t k_y
                                = base_idx_y + dilation_y * i_ker_y;
                            const int32_t k_x
                                = base_idx_x + dilation_x * i_ker_x;

                            if (k_y < 0 || k_y >= input_height || k_x < 0
                                || k_x >= input_width)
                            {
                                th_memset((int8_t *)two_column_buf,
                                          0,
                                          sizeof(q15_t) * input_depth);
                            }
                            else
                            {
                                nn_q7_to_q15_with_offset(
                                    input_data
                                        + (k_y * input_width + k_x)
                                              * input_depth,
                                    two_column_buf,
                                    input_depth,
                                    input_offset);
                            }
                            two_column_buf += input_depth;
                        }
                    }

                    /* Every 2 output pixels: run the GEMM kernel */
                    if (two_column_buf
                        == buffer_a
                               + 2 * input_depth * filter_height * filter_width)
                    {

                        out = nn_mat_mult_kernel_s8_s16(
                            filter_data,
                            buffer_a,
                            (uint16_t)output_depth,
                            output_shift,
                            output_mult,
                            out_offset,
                            (int16_t)out_act_min,
                            (int16_t)out_act_max,
                            (uint16_t)(input_depth * filter_height
                                       * filter_width),
                            bias_data,
                            out);

                        two_column_buf = buffer_a;
                    }
                }
            }

            /* Handle odd-pixel leftover */
            if (two_column_buf != buffer_a)
            {
                const q7_t *ker_a = filter_data;
                for (int32_t i = 0; i < output_depth; i++)
                {
                    int32_t      sum       = bias_data ? bias_data[i] : 0;
                    const q15_t *ip_as_col = buffer_a;
                    int32_t      col_count
                        = input_depth * filter_height * filter_width;
                    while (col_count--)
                    {
                        sum += (int32_t)(*ker_a++) * (int32_t)(*ip_as_col++);
                    }
                    sum = nn_requantize(sum, output_mult[i], output_shift[i]);
                    sum += out_offset;
                    sum    = sum > out_act_max ? out_act_max : sum;
                    sum    = sum < out_act_min ? out_act_min : sum;
                    *out++ = (q7_t)sum;
                }
            }

            input_data += input_height * input_width * input_depth;
            output_data += output_height * output_width * output_depth;
        }
    }

    return 0;
}
