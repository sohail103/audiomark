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

#include "math_types.h"
#include "functions.h"

#include <stdint.h>

/*
 * Specialized for the KWS model:
 * input: 25x5x64, global avgpool
 */
#define INPUT_Y  25
#define INPUT_X  5
#define CHANNELS 64
#define COUNT    (INPUT_Y * INPUT_X)
#define HALF     (COUNT / 2)

int32_t
nn_avgpool_global_s8(const q7_t *input_data, q7_t *output_data)
{
    for (int32_t c = 0; c < CHANNELS; c++)
    {
        int16_t sum = 0;

        const q7_t *base = input_data + c;

        /* accumulate over spatial */
        for (int32_t idx = 0; idx < COUNT; idx++)
        {
            sum += *base;
            base += CHANNELS; /* move to next pixel, same channel */
        }

        /* symmetric remove rounding */
        if (sum >= 0)
        {
            sum = (sum + HALF) / COUNT;
        }
        else
        {
            sum = (sum - HALF) / COUNT;
        }

        output_data[c] = (q7_t)sum;
    }

    return 0;
}
