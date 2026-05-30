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
#include "nn/functions.h"
#include "nn/support_functions.h"
#include "nn/buffer_utils.h"

#define BUF_CONV0                        \
    NN_CONV_S8_BUF_SIZE(CONV_0_IN_CH,    \
                        CONV_0_FILTER_H, \
                        CONV_0_FILTER_W, \
                        CONV_0_OUT_CH,   \
                        CONV_0_INPUT_H,  \
                        CONV_0_INPUT_W,  \
                        CONV_0_OUTPUT_H, \
                        CONV_0_OUTPUT_W)

#define BUF_CONV2                        \
    NN_CONV_S8_BUF_SIZE(CONV_2_IN_CH,    \
                        CONV_2_FILTER_H, \
                        CONV_2_FILTER_W, \
                        CONV_2_OUT_CH,   \
                        CONV_2_INPUT_H,  \
                        CONV_2_INPUT_W,  \
                        CONV_2_OUTPUT_H, \
                        CONV_2_OUTPUT_W)

#define BUF_CONV4                        \
    NN_CONV_S8_BUF_SIZE(CONV_4_IN_CH,    \
                        CONV_4_FILTER_H, \
                        CONV_4_FILTER_W, \
                        CONV_4_OUT_CH,   \
                        CONV_4_INPUT_H,  \
                        CONV_4_INPUT_W,  \
                        CONV_4_OUTPUT_H, \
                        CONV_4_OUTPUT_W)

#define BUF_CONV6                        \
    NN_CONV_S8_BUF_SIZE(CONV_6_IN_CH,    \
                        CONV_6_FILTER_H, \
                        CONV_6_FILTER_W, \
                        CONV_6_OUT_CH,   \
                        CONV_6_INPUT_H,  \
                        CONV_6_INPUT_W,  \
                        CONV_6_OUTPUT_H, \
                        CONV_6_OUTPUT_W)

#define BUF_CONV8                        \
    NN_CONV_S8_BUF_SIZE(CONV_8_IN_CH,    \
                        CONV_8_FILTER_H, \
                        CONV_8_FILTER_W, \
                        CONV_8_OUT_CH,   \
                        CONV_8_INPUT_H,  \
                        CONV_8_INPUT_W,  \
                        CONV_8_OUTPUT_H, \
                        CONV_8_OUTPUT_W)

#define MAX_BUF_SIZE \
    (MAX5(BUF_CONV0, BUF_CONV2, BUF_CONV4, BUF_CONV6, BUF_CONV8))

nn_context     ctx;
static uint8_t scratch_buffer[MAX_BUF_SIZE] __attribute__((aligned(16)));

void
th_nn_init(void)
{
    ctx.size = MAX_BUF_SIZE;
    ctx.buf  = scratch_buffer;
}
