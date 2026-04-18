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

#include "nn/math_types.h"
#include "nn/functions.h"
#include "rvv_support_guard.h"

#include <stddef.h>

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
    int32_t c = 0;

    while (c < CHANNELS)
    {
        size_t vl = __riscv_vsetvl_e8m4(CHANNELS - c);

        /* initializing accumulator in i16m8 */
        vint16m8_t vsum = __riscv_vmv_v_x_i16m8(0, vl);

        const q7_t *base = input_data + c;

        for (int32_t idx = 0; idx < COUNT; idx++)
        {
            /* loading in i8m4 */
            vint8m4_t v8 = __riscv_vle8_v_i8m4(base, vl);

            vsum = __riscv_vwadd_wv_i16m8(vsum, v8, vl);

            base += CHANNELS;
        }

        /* setting up numerator so that integer division truncates towards
         * nearest whole number rather than always towards 0 */
        vbool2_t   mask   = __riscv_vmsgt_vx_i16m8_b2(vsum, 0, vl);
        vint16m8_t vpos   = __riscv_vadd_vx_i16m8(vsum, HALF, vl);
        vint16m8_t vneg   = __riscv_vsub_vx_i16m8(vsum, HALF, vl);
        vint16m8_t vround = __riscv_vmerge_vvm_i16m8(vneg, vpos, mask, vl);

        vint16m8_t vavg = __riscv_vdiv_vx_i16m8(vround, COUNT, vl);

        vint8m4_t v8_out = __riscv_vncvt_x_x_w_i8m4(vavg, vl);

        __riscv_vse8_v_i8m4(output_data + c, v8_out, vl);

        c += vl;
    }

    return 0;
}
