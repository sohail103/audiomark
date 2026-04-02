// Modifications copyright (C) 2023 Chair of Electronic Design Automation, TUM
/*
 * SPDX-FileCopyrightText: Copyright 2023 Arm Limited and/or its affiliates
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

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        muriscv_nn_depthwise_conv_get_buffer_sizes_s8.c
 * Description:  Collection of get buffer size functions for the various s8
 * convolution layer functions.
 *
 * $Date:        30 October 2023
 * $Revision:    V.1.1.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "muriscv_nn_functions.h"
#include "muriscv_nn_support_functions.h"

int32_t
muriscv_nn_depthwise_conv_wrapper_s8_get_buffer_size(
    const muriscv_nn_dw_conv_params *dw_conv_params,
    const muriscv_nn_dims           *input_dims,
    const muriscv_nn_dims           *filter_dims,
    const muriscv_nn_dims           *output_dims)
{
    return 0;
}
