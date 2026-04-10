/**
 * Copyright (C) 2024 SPEC Embedded Group
 * Copyright (C) 2022-2023 EEMBC
 * Copyright (C) 2022-2024 Arm Limited
 *
 * All EEMBC Benchmark Software are products of EEMBC and are provided under the
 * terms of the EEMBC Benchmark License Agreements. The EEMBC Benchmark Software
 * are proprietary intellectual properties of EEMBC and its Members and is
 * protected under all applicable laws, including all applicable copyright laws.
 *
 * If you received this EEMBC Benchmark Software without having a currently
 * effective EEMBC Benchmark License Agreement, you must discontinue use.
 *
 * Modifications copyright (C) 2026 Sohail Raj Satapathy
 */

#include "ee_audiomark.h"
#include "ee_api.h"
#include "nn/functions.h"

/* Get size of additional buffers required by library/framework */
static int
ds_cnn_s_s8_get_buffer_size(void)
{
    /* Custom function based on knowledge that only select layers of DS_CNN_S
     * network require additional buffers. */
    int            max_buffer = 0;
    nn_conv_params conv_params;
    nn_dims        input_dims;
    nn_dims        filter_dims;
    nn_dims        output_dims;

    /* Layer 0 - Conv */
    conv_params.padding.h  = CONV_0_PAD_H;
    conv_params.padding.w  = CONV_0_PAD_W;
    conv_params.stride.h   = CONV_0_STRIDE_H;
    conv_params.stride.w   = CONV_0_STRIDE_W;
    conv_params.dilation.h = CONV_0_DILATION_H;
    conv_params.dilation.w = CONV_0_DILATION_W;

    input_dims.n = CONV_0_INPUT_BATCHES;
    input_dims.h = CONV_0_INPUT_H;
    input_dims.w = CONV_0_INPUT_W;
    input_dims.c = CONV_0_IN_CH;

    filter_dims.h = CONV_0_FILTER_H;
    filter_dims.w = CONV_0_FILTER_W;

    output_dims.n = input_dims.n;
    output_dims.h = CONV_0_OUTPUT_H;
    output_dims.w = CONV_0_OUTPUT_W;
    output_dims.c = CONV_0_OUT_CH;

    int32_t size = nn_convolve_s8_get_buffer_size(&input_dims, &filter_dims);

    max_buffer = size > max_buffer ? size : max_buffer;

    /* Layer 0 - DW Conv */
    nn_dw_conv_params dw_conv_params;
    dw_conv_params.activation.min = DW_CONV_1_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = DW_CONV_1_OUT_ACTIVATION_MAX;
    dw_conv_params.ch_mult        = 1;
    dw_conv_params.dilation.h     = DW_CONV_1_DILATION_H;
    dw_conv_params.dilation.w     = DW_CONV_1_DILATION_W;
    dw_conv_params.input_offset   = DW_CONV_1_INPUT_OFFSET;
    dw_conv_params.output_offset  = DW_CONV_1_OUTPUT_OFFSET;
    dw_conv_params.padding.h      = DW_CONV_1_PAD_H;
    dw_conv_params.padding.w      = DW_CONV_1_PAD_W;
    dw_conv_params.stride.h       = DW_CONV_1_STRIDE_H;
    dw_conv_params.stride.w       = DW_CONV_1_STRIDE_W;

    filter_dims.h = DW_CONV_1_FILTER_H;
    filter_dims.w = DW_CONV_1_FILTER_W;

    input_dims.n = 1;
    input_dims.h = DW_CONV_1_INPUT_H;
    input_dims.w = DW_CONV_1_INPUT_W;
    input_dims.c = DW_CONV_1_OUT_CH;

    output_dims.h = DW_CONV_1_OUTPUT_H;
    output_dims.w = DW_CONV_1_OUTPUT_W;
    output_dims.c = DW_CONV_1_OUT_CH;

    return max_buffer;
}

/* Test for a complete int8 DS_CNN_S keyword spotting network from
 * https://github.com/ARM-software/ML-zoo & Tag: 22.02 */
nn_context ctx;

void
th_nn_init(void)
{
    ctx.size = ds_cnn_s_s8_get_buffer_size();

    /* N.B. The developer owns this file so they can allocate how they like. */
    ctx.buf = th_malloc(ctx.size, COMPONENT_KWS);

    /* we don't free in audiomark */
}
