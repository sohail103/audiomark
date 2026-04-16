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

#include <stdint.h>

#define ALIGN4(x) (((x) + 3) & ~3)

#define NN_CONV_S8_BUF_SIZE(in_ch, k_w, k_h) \
    (2 * ALIGN4((int32_t)(in_ch) * (k_w) * (k_h)) * sizeof(int16_t))

#endif