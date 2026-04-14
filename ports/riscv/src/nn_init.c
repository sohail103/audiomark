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
#include "nn/support_functions.h"

/* this function computes the maximum scratch buffer required across all
 * layers/kernels. if a new kernel uses ctx.buf, its buffer requirement must be
 * added here.
 */
static int
ds_cnn_s_s8_get_buffer_size(void)
{
    int32_t size       = 0;
    int32_t max_buffer = 0;
    nn_dims input_dims;
    nn_dims filter_dims;

    /* currently, all conv layers use the same generic convolution
     * implementation. if a different convolution implementation is used for any
     * specific layer, change the corresponding get_buffer_size function
     */

    // Conv0
    filter_dims.w = CONV_0_FILTER_W;
    filter_dims.h = CONV_0_FILTER_H;
    input_dims.c  = CONV_0_IN_CH;
    size          = nn_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    max_buffer    = MAX(max_buffer, size);

    // Conv2
    filter_dims.w = CONV_2_FILTER_W;
    filter_dims.h = CONV_2_FILTER_H;
    input_dims.c  = CONV_2_IN_CH;
    size          = nn_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    max_buffer    = MAX(max_buffer, size);

    // Conv4
    filter_dims.w = CONV_4_FILTER_W;
    filter_dims.h = CONV_4_FILTER_H;
    input_dims.c  = CONV_4_IN_CH;
    size          = nn_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    max_buffer    = MAX(max_buffer, size);

    // Conv6
    filter_dims.w = CONV_6_FILTER_W;
    filter_dims.h = CONV_6_FILTER_H;
    input_dims.c  = CONV_6_IN_CH;
    size          = nn_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    max_buffer    = MAX(max_buffer, size);

    // Conv8
    filter_dims.w = CONV_8_FILTER_W;
    filter_dims.h = CONV_8_FILTER_H;
    input_dims.c  = CONV_8_IN_CH;
    size          = nn_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    max_buffer    = MAX(max_buffer, size);

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
