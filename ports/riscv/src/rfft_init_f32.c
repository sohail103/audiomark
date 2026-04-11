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

#include "ee_audiomark.h"
#include "ee_api.h"

ee_status_t
th_rfft_init_f32(ee_rfft_f32_t *p_instance, int fft_length)
{
    p_instance->fft_len = fft_length;

    p_instance->work_real
        = th_malloc(sizeof(ee_f32_t) * fft_length, COMPONENT_KWS);
    p_instance->work_imag
        = th_malloc(sizeof(ee_f32_t) * fft_length, COMPONENT_KWS);

    if (!p_instance->work_real || !p_instance->work_imag)
    {
        return EE_STATUS_ERROR;
    }

    return EE_STATUS_OK;
}
