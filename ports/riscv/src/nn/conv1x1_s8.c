/*
 * Copyright (C) 2010-2022 Arm Limited or its affiliates.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Modifications copyright (C) 2021-2024 Chair of Electronic Design Automation,
 * TUM Modifications copyright (C) 2026 Sohail Raj Satapathy
 */

#include "functions.h"
#include "support_functions.h"
#include "ee_api.h"
#include "ee_nn.h"

#include <stdint.h>

/*
 * Specialized pointwise (1x1) convolution for Layers 2, 4, 6, 8:
 *   Input : 1 x 25 x 5 x 64, Filter: 64 x 1 x 1 x 64, Output: 1 x 25 x 5 x 64
 *
 * With a 1x1 kernel, stride 1, and no padding the im2col is an identity, so
 * the input can be passed directly to nn_mat_mult_nt_t_s8 as the lhs matrix.
 * Constants from CONV_2_* apply equally to CONV_4, CONV_6, CONV_8.
 */
int32_t
nn_conv1x1_s8(const nn_context                  *ctx,
              const nn_per_channel_quant_params *quant_params,
              const q7_t                        *input_data,
              const q7_t                        *filter_data,
              const int32_t                     *bias_data,
              q7_t                              *output_data)
{
    (void)ctx;

    static const int32_t output_h  = CONV_2_OUTPUT_H;
    static const int32_t output_w  = CONV_2_OUTPUT_W;
    static const int32_t output_ch = CONV_2_OUT_CH;
    static const int32_t input_ch  = CONV_2_IN_CH;

    static const int32_t input_offset       = CONV_2_INPUT_OFFSET;
    static const int32_t out_offset         = CONV_2_OUTPUT_OFFSET;
    static const int32_t out_activation_min = CONV_2_OUT_ACTIVATION_MIN;
    static const int32_t out_activation_max = CONV_2_OUT_ACTIVATION_MAX;

    return nn_mat_mult_nt_t_s8(input_data,
                               filter_data,
                               bias_data,
                               output_data,
                               quant_params->multiplier,
                               quant_params->shift,
                               output_h * output_w, /* lhs_rows */
                               output_ch,           /* rhs_rows */
                               input_ch,            /* rhs_cols */
                               input_offset,
                               out_offset,
                               out_activation_min,
                               out_activation_max,
                               0,       /* row_address_offset (unused) */
                               input_ch /* lhs_cols_offset */
    );
}
