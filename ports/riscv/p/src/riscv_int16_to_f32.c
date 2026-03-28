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

#include "ee_api.h"

void th_int16_to_f32(const int16_t *p_src, float *p_dst, uint32_t len)
{
    if (!p_src || !p_dst || len == 0)
    {
        return;
    }

    /* Multiply by 1 / 2^15 to map [-32768, 32767] -> [-1.0, 1.0) */
    float inv_scale = 1.0f / 32768.0f;

    for (uint32_t i = 0; i < len; i++)
    {
        p_dst[i] = (float)p_src[i] * inv_scale;
    }
}
