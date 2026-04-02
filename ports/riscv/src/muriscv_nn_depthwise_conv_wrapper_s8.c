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
 * Modifications copyright (C) 2021-2022 Chair of Electronic Design Automation,
 * TUM
 * Modifications copyright (C) 2026 Sohail Raj Satapathy
 */

#include "muriscv_nn_functions.h"

/*
 *  s8 Depthwise conv wrapper function
 *
 *  Refer header file for details.
 *
 */
muriscv_nn_status
muriscv_nn_depthwise_conv_wrapper_s8(
    const muriscv_nn_context                  *ctx,
    const muriscv_nn_dw_conv_params           *dw_conv_params,
    const muriscv_nn_per_channel_quant_params *quant_params,
    const muriscv_nn_dims                     *input_dims,
    const q7_t                                *input,
    const muriscv_nn_dims                     *filter_dims,
    const q7_t                                *filter,
    const muriscv_nn_dims                     *bias_dims,
    const int32_t                             *bias,
    const muriscv_nn_dims                     *output_dims,
    q7_t                                      *output)
{
    muriscv_nn_status status = MURISCV_NN_SUCCESS;

    status = muriscv_nn_depthwise_conv_s8(ctx,
                                          dw_conv_params,
                                          quant_params,
                                          input_dims,
                                          input,
                                          filter_dims,
                                          filter,
                                          bias_dims,
                                          bias,
                                          output_dims,
                                          output);

    /* Return to application */
    return status;
}
