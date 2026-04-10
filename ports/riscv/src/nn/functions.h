// Modifications copyright (C) 2024 Chair of Electronic Design Automation, TUM
/*
 * SPDX-FileCopyrightText: Copyright 2010-2024 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
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
 * Modifications copyright (C) 2026 Sohail Raj Satapathy
 */

/**
 * A collection of functions to perform basic operations for neural network
 * layers.
 */

#ifndef NN_FUNCTIONS_H
#define NN_FUNCTIONS_H

#include "math_types.h"
#include "types.h"

int32_t nn_convolve_s8(const nn_context                  *ctx,
                       const nn_conv_params              *conv_params,
                       const nn_per_channel_quant_params *quant_params,
                       const nn_dims                     *input_dims,
                       const int8_t                      *input_data,
                       const nn_dims                     *filter_dims,
                       const int8_t                      *filter_data,
                       const nn_dims                     *bias_dims,
                       const int32_t                     *bias_data,
                       const nn_dims                     *output_dims,
                       int8_t                            *output_data);

int32_t nn_convolve_s8_get_buffer_size(const nn_dims *input_dims,
                                       const nn_dims *filter_dims);

int32_t nn_depthwise_conv_s8(const nn_dw_conv_params           *dw_conv_params,
                             const nn_per_channel_quant_params *quant_params,
                             const nn_dims                     *input_dims,
                             const int8_t                      *input_data,
                             const nn_dims                     *filter_dims,
                             const int8_t                      *filter_data,
                             const nn_dims                     *bias_dims,
                             const int32_t                     *bias_data,
                             const nn_dims                     *output_dims,
                             int8_t                            *output_data);

int32_t nn_fully_connected_s8(const nn_fc_params               *fc_params,
                              const nn_per_tensor_quant_params *quant_params,
                              const nn_dims                    *input_dims,
                              const int8_t                     *input_data,
                              const nn_dims                    *filter_dims,
                              const int8_t                     *filter_data,
                              const nn_dims                    *bias_dims,
                              const int32_t                    *bias_data,
                              const nn_dims                    *output_dims,
                              int8_t                           *output_data);

int32_t nn_avgpool_s8(const nn_pool_params *pool_params,
                      const nn_dims        *input_dims,
                      const int8_t         *input_data,
                      const nn_dims        *filter_dims,
                      const nn_dims        *output_dims,
                      int8_t               *output_data);

void nn_softmax_s8(const int8_t *input,
                   const int32_t num_rows,
                   const int32_t row_size,
                   const int32_t mult,
                   const int32_t shift,
                   const int32_t diff_min,
                   void         *output);

#endif
