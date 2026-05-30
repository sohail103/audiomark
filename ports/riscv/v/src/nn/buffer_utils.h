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

#ifndef BUFFER_UTILS_H
#define BUFFER_UTILS_H

#include "support_functions.h"

#include <stdint.h>

#define NN_CONV_S8_BUF_SIZE(in_ch, k_h, k_w, out_ch, in_h, in_w, out_h, out_w) \
    (((k_h) == 4 && (k_w) == 4)                                                \
         ? ((int32_t)(out_h) * (int32_t)(out_w) * (int32_t)(out_ch) * 4        \
            + 64 * (int32_t)(in_ch) * (int32_t)(out_ch)                        \
            + 10 * (int32_t)(out_ch)                                           \
            + ((int32_t)(in_h) + 8) * ((int32_t)(in_w) + 8) * (int32_t)(in_ch) \
            + 65536 + 256)                                                     \
         : (MAX((int32_t)(out_h) * (int32_t)(out_w) * (int32_t)(out_ch) * 4,   \
                4 * (int32_t)(k_h) * (int32_t)(k_w) * (int32_t)(in_ch))        \
            + (int32_t)(k_h) * (int32_t)(k_w) * (int32_t)(in_ch)               \
                  * (int32_t)(out_ch)                                          \
            + 6 * (int32_t)(out_ch) + 256))

#endif /* BUFFER_UTILS_H */
