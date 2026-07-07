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

#include <stddef.h>

q7_t *
nn_mat_mult_core_1x1_s8(const q7_t          *input_a,
                        const q7_t          *act_row,
                        const uint16_t       output_ch,
                        const int32_t       *out_shift,
                        const int32_t       *out_mult,
                        const int32_t        out_offset,
                        const int16_t        activation_min,
                        const int16_t        activation_max,
                        const uint16_t       num_col_a,
                        const int32_t *const bias,
                        q7_t                *out)
{
    for (int oc = 0; oc < output_ch; oc++)
    {
        const q7_t *w   = input_a + (size_t)oc * num_col_a;
        q31_t       sum = bias ? bias[oc] : 0;

        for (int k = 0; k < num_col_a; k++)
        {
            sum += w[k] * act_row[k];
        }

        sum     = nn_requantize(sum, out_mult[oc], out_shift[oc]) + out_offset;
        sum     = MIN(MAX(sum, activation_min), activation_max);
        out[oc] = (q7_t)sum;
    }

    return out + output_ch;
}
