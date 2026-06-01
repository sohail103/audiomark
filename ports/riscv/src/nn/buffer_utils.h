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

#define BUF_CONV0 \
    NN_CONV_S8_BUF_SIZE(CONV_0_IN_CH, CONV_0_FILTER_W, CONV_0_FILTER_H)
#define BUF_CONV2 \
    NN_CONV_S8_BUF_SIZE(CONV_2_IN_CH, CONV_2_FILTER_W, CONV_2_FILTER_H)
#define BUF_CONV4 \
    NN_CONV_S8_BUF_SIZE(CONV_4_IN_CH, CONV_4_FILTER_W, CONV_4_FILTER_H)
#define BUF_CONV6 \
    NN_CONV_S8_BUF_SIZE(CONV_6_IN_CH, CONV_6_FILTER_W, CONV_6_FILTER_H)
#define BUF_CONV8 \
    NN_CONV_S8_BUF_SIZE(CONV_8_IN_CH, CONV_8_FILTER_W, CONV_8_FILTER_H)

#define MAX_BUF_SIZE \
    (MAX5(BUF_CONV0, BUF_CONV2, BUF_CONV4, BUF_CONV6, BUF_CONV8))

#define ALIGNMENT 8

#endif
