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

#include "support_functions.h"
#include "convolve_config.h"
#include <stddef.h>

q7_t *
nn_mat_mult_kernel_s8_s8(const q7_t          *input_a,
                         const q7_t          *input_b,
                         const uint16_t       output_ch,
                         const int32_t       *out_shift,
                         const int32_t       *out_mult,
                         const int32_t        out_offset,
                         const int16_t        activation_min,
                         const int16_t        activation_max,
                         const uint16_t       num_col_a,
                         const int32_t *const output_bias,
                         q7_t                *out_0)
{
    const q7_t *ip_b[NN_KERNEL_COLS];
    q7_t       *out_row[NN_KERNEL_COLS];

    for (int r = 0; r < NN_KERNEL_COLS; r++)
    {
        ip_b[r]    = input_b + (size_t)r * num_col_a;
        out_row[r] = out_0 + (size_t)r * output_ch;
    }

    for (uint16_t oc = 0; oc < output_ch; oc++)
    {
        const q7_t *w = input_a + (size_t)oc * num_col_a;
        q31_t       sum[NN_KERNEL_COLS];

        for (int r = 0; r < NN_KERNEL_COLS; r++)
        {
            sum[r] = output_bias ? output_bias[oc] : 0;
        }

        for (uint16_t k = 0; k < num_col_a; k++)
        {
            const q7_t wk = w[k];

            for (int r = 0; r < NN_KERNEL_COLS; r++)
            {
                sum[r] += wk * ip_b[r][k];
            }
        }

        const int32_t mult  = out_mult[oc];
        const int32_t shift = out_shift[oc];

        for (int r = 0; r < NN_KERNEL_COLS; r++)
        {
            q31_t s        = nn_requantize(sum[r], mult, shift) + out_offset;
            s              = MIN(MAX(s, activation_min), activation_max);
            out_row[r][oc] = (q7_t)s;
        }
    }

    return out_0 + (size_t)NN_KERNEL_COLS * output_ch;
}
