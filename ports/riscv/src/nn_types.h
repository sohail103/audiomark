/*
 * Copyright (C) 2020-2022 Arm Limited or its affiliates.
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
 * Modifications copyright (C) 2021-2024 Chair of Electronic Design Automation,
 * TUM
 * Modifications copyright (C) 2026 Sohail Raj Satapathy
 */

#ifndef _MURISCV_NN_TYPES_H
#define _MURISCV_NN_TYPES_H

#include <stdint.h>

/** MURISCV-NN object to contain the width and height of a tile */
typedef struct
{
    int32_t w; /**< Width */
    int32_t h; /**< Height */
} muriscv_nn_tile;

/** MURISCV-NN object used for the function context. */
typedef struct
{
    void   *buf;  /**< Pointer to a buffer needed for the optimization */
    int32_t size; /**< Buffer size */
} muriscv_nn_context;

/** MURISCV-NN object to contain the dimensions of the tensors */
typedef struct
{
    int32_t n; /**< Generic dimension to contain either the batch size or
                  output channels. Please refer to the function
                  documentation for more information */
    int32_t h; /**< Height */
    int32_t w; /**< Width */
    int32_t c; /**< Input channels */
} muriscv_nn_dims;

/** MURISCV-NN object for the per-channel quantization parameters */
typedef struct
{
    int32_t *multiplier; /**< Multiplier values */
    int32_t *shift;      /**< Shift values */
} muriscv_nn_per_channel_quant_params;

/** MURISCV-NN object for the per-tensor quantization parameters */
typedef struct
{
    int32_t multiplier; /**< Multiplier value */
    int32_t shift;      /**< Shift value */
} muriscv_nn_per_tensor_quant_params;

/** MURISCV-NN object for the quantized Relu activation */
typedef struct
{
    int32_t min; /**< Min value used to clamp the result */
    int32_t max; /**< Max value used to clamp the result */
} muriscv_nn_activation;

/** MURISCV-NN object for the convolution layer parameters */
typedef struct
{
    int32_t input_offset;  /**< Negative of the Zero value for the input
                              tensor */
    int32_t output_offset; /**< negative of the Zero value for the output
                              tensor */
    muriscv_nn_tile       stride;
    muriscv_nn_tile       padding;
    muriscv_nn_tile       dilation;
    muriscv_nn_activation activation;
} muriscv_nn_conv_params;

/** MURISCV-NN object for Depthwise convolution layer parameters */
typedef struct
{
    int32_t input_offset;  /**< negative of the Zero value for the input
                              tensor */
    int32_t output_offset; /**< negative of the Zero value for the output
                              tensor */
    int32_t ch_mult;       /**< Channel Multiplier. ch_mult * in_ch = out_ch */
    muriscv_nn_tile       stride;
    muriscv_nn_tile       padding;
    muriscv_nn_tile       dilation;
    muriscv_nn_activation activation;
} muriscv_nn_dw_conv_params;

/** MURISCV-NN object for pooling layer parameters */
typedef struct
{
    muriscv_nn_tile       stride;
    muriscv_nn_tile       padding;
    muriscv_nn_activation activation;
} muriscv_nn_pool_params;

/** MURISCV-NN object for Fully Connected layer parameters */
typedef struct
{
    int32_t input_offset;  /**< Zero value for the input tensor */
    int32_t output_offset; /**< Zero value for the output tensor */
    muriscv_nn_activation activation;
} muriscv_nn_fc_params;

#endif /* _MURISCV_NN_TYPES_H */
