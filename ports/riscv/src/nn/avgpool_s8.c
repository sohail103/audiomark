/*
 * Copyright (C) 2010-2021 Arm Limited or its affiliates.
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

#include "functions.h"
#include "support_functions.h"

int32_t
nn_avgpool_s8(const nn_pool_params *pool_params,
              const nn_dims        *input_dims,
              const q7_t           *src,
              const nn_dims        *filter_dims,
              const nn_dims        *output_dims,
              q7_t                 *dst)
{
    const int32_t input_y  = input_dims->h;
    const int32_t input_x  = input_dims->w;
    const int32_t output_y = output_dims->h;
    const int32_t output_x = output_dims->w;
    const int32_t stride_y = pool_params->stride.h;
    const int32_t stride_x = pool_params->stride.w;
    const int32_t kernel_y = filter_dims->h;
    const int32_t kernel_x = filter_dims->w;
    const int32_t pad_y    = pool_params->padding.h;
    const int32_t pad_x    = pool_params->padding.w;
    const int32_t act_min  = pool_params->activation.min;
    const int32_t act_max  = pool_params->activation.max;
    const int32_t ch_src   = input_dims->c;

    for (int32_t i_y = 0; i_y < output_y; i_y++)
    {
        for (int32_t i_x = 0; i_x < output_x; i_x++)
        {
            const int32_t k_y_start = MAX(0, i_y * stride_y - pad_y);
            const int32_t k_y_end
                = MIN(i_y * stride_y - pad_y + kernel_y, input_y);

            const int32_t k_x_start = MAX(0, i_x * stride_x - pad_x);
            const int32_t k_x_end
                = MIN(i_x * stride_x - pad_x + kernel_x, input_x);

            for (int32_t i_ch_in = 0; i_ch_in < ch_src; i_ch_in++)
            {
                int32_t sum   = 0;
                int32_t count = 0;
                for (int32_t k_y = k_y_start; k_y < k_y_end; k_y++)
                {
                    for (int32_t k_x = k_x_start; k_x < k_x_end; k_x++)
                    {
                        sum += src[i_ch_in + ch_src * (k_x + k_y * input_x)];
                        count++;
                    }
                }

                sum = sum > 0 ? (sum + count / 2) / count
                              : (sum - count / 2) / count;
                sum = MAX(sum, act_min);
                sum = MIN(sum, act_max);

                dst[i_ch_in + ch_src * (i_x + i_y * output_x)] = sum;
            }
        }
    }
    return 0;
}
