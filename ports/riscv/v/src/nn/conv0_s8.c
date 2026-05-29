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

/* nn_conv0_s8: Input 49x10x1, Kernel 10x4->64, Stride 2x2, Pad 4x1, Output
 * 25x5x64 */

#include "functions.h"
#include "support_functions.h"
#include "ee_api.h"
#include "ee_nn.h"
#include "rvv_support_guard.h"

#include <stdint.h>

#define IH    49
#define IW    10
#define FH    10
#define FW    4
#define OH    25
#define OW    5
#define OC    64
#define SH    2
#define SW    2
#define PH    4
#define PW    1
#define PATCH (FH * FW) /* 40 */

#define Y_LO 2
#define Y_HI 22
#define X_LO 1
#define X_HI 4

static q15_t col_buf[2 * PATCH];

#define FILL_ROW(row_ptr, r, in_off, dst)                                      \
    do                                                                         \
    {                                                                          \
        size_t     _vl  = __riscv_vsetvl_e8m1(FW);                             \
        vint8m1_t  _v8  = __riscv_vle8_v_i8m1((row_ptr) + (r) * IW, _vl);      \
        vint16m2_t _v16 = __riscv_vsext_vf2_i16m2(_v8, _vl);                   \
        _v16            = __riscv_vadd_vx_i16m2(_v16, (int16_t)(in_off), _vl); \
        __riscv_vse16_v_i16m2((dst) + (r) * FW, _v16, _vl);                    \
    } while (0)

int32_t
nn_conv0_s8(const nn_conv_params              *conv_params,
            const nn_per_channel_quant_params *quant_params,
            const int8_t                      *input_data,
            const int8_t                      *filter_data,
            const int32_t                     *bias_data,
            int8_t                            *output_data)
{
    const int32_t  in_off  = conv_params->input_offset;
    const int32_t  out_off = conv_params->output_offset;
    const int32_t  act_min = conv_params->activation.min;
    const int32_t  act_max = conv_params->activation.max;
    const int32_t *mult    = quant_params->multiplier;
    const int32_t *shift   = quant_params->shift;

    int8_t *out      = output_data;
    int     px_count = 0;
    q15_t  *slot0    = col_buf;
    q15_t  *slot1    = col_buf + PATCH;

    for (int oy = 0; oy < OH; oy++)
    {
        const int32_t by = SH * oy - PH;
        const int     y_intr
            = ((uint32_t)(oy - Y_LO) <= (uint32_t)(Y_HI - Y_LO - 1));

        for (int ox = 0; ox < OW; ox++)
        {
            const int32_t bx = SW * ox - PW;
            const int     x_intr
                = ((uint32_t)(ox - X_LO) <= (uint32_t)(X_HI - X_LO - 1));
            q15_t *slot = (px_count == 0) ? slot0 : slot1;

            if (y_intr && x_intr)
            {
                const int8_t *row = input_data + by * IW + bx;
                FILL_ROW(row, 0, in_off, slot);
                FILL_ROW(row, 1, in_off, slot);
                FILL_ROW(row, 2, in_off, slot);
                FILL_ROW(row, 3, in_off, slot);
                FILL_ROW(row, 4, in_off, slot);
                FILL_ROW(row, 5, in_off, slot);
                FILL_ROW(row, 6, in_off, slot);
                FILL_ROW(row, 7, in_off, slot);
                FILL_ROW(row, 8, in_off, slot);
                FILL_ROW(row, 9, in_off, slot);
            }
            else
            {
                for (int ky = 0; ky < FH; ky++)
                {
                    const int32_t iy = by + ky;
                    for (int kx = 0; kx < FW; kx++)
                    {
                        const int32_t ix = bx + kx;
                        *slot++
                            = ((uint32_t)iy < (uint32_t)IH
                               && (uint32_t)ix < (uint32_t)IW)
                                  ? (q15_t)(input_data[iy * IW + ix] + in_off)
                                  : 0;
                    }
                }
                slot = (px_count == 0) ? slot0 : slot1;
            }

            px_count++;

            if (px_count == 2)
            {
                for (int co = 0; co < OC; co++)
                {
                    const int8_t *w   = filter_data + co * PATCH;
                    const q15_t  *p0  = slot0;
                    const q15_t  *p1  = slot1;
                    int           rem = PATCH;

                    /* zero-init accumulators at full PATCH width */
                    size_t     full_vl = __riscv_vsetvl_e8m1(PATCH);
                    vint32m4_t vacc0   = __riscv_vmv_v_x_i32m4(0, full_vl);
                    vint32m4_t vacc1   = __riscv_vmv_v_x_i32m4(0, full_vl);

                    while (rem > 0)
                    {
                        size_t     vl  = __riscv_vsetvl_e8m1(rem);
                        vint8m1_t  vw8 = __riscv_vle8_v_i8m1(w, vl);
                        vint16m2_t vw  = __riscv_vsext_vf2_i16m2(vw8, vl);
                        vint16m2_t vc0 = __riscv_vle16_v_i16m2(p0, vl);
                        vint16m2_t vc1 = __riscv_vle16_v_i16m2(p1, vl);
                        vint32m4_t vp0 = __riscv_vwmul_vv_i32m4(vw, vc0, vl);
                        vint32m4_t vp1 = __riscv_vwmul_vv_i32m4(vw, vc1, vl);
                        vacc0          = __riscv_vadd_vv_i32m4(vacc0, vp0, vl);
                        vacc1          = __riscv_vadd_vv_i32m4(vacc1, vp1, vl);
                        w += vl;
                        p0 += vl;
                        p1 += vl;
                        rem -= (int)vl;
                    }

                    /* one reduction per accumulator at end */
                    vint32m1_t vz = __riscv_vmv_s_x_i32m1(0, 1);
                    int32_t    s0 = (bias_data ? bias_data[co] : 0)
                                    + __riscv_vmv_x_s_i32m1_i32(
                                        __riscv_vredsum_vs_i32m4_i32m1(
                                            vacc0, vz, full_vl));
                    int32_t    s1 = (bias_data ? bias_data[co] : 0)
                                    + __riscv_vmv_x_s_i32m1_i32(
                                        __riscv_vredsum_vs_i32m4_i32m1(
                                            vacc1, vz, full_vl));

                    s0      = nn_requantize(s0, mult[co], shift[co]) + out_off;
                    s1      = nn_requantize(s1, mult[co], shift[co]) + out_off;
                    out[co] = (int8_t)MAX(act_min, MIN(act_max, s0));
                    out[OC + co] = (int8_t)MAX(act_min, MIN(act_max, s1));
                }
                out += 2 * OC;
                px_count = 0;
            }
        }
    }

    /* OH*OW=125 is odd; one pixel remains in slot0 */
    if (px_count == 1)
    {
        for (int co = 0; co < OC; co++)
        {
            const int8_t *w   = filter_data + co * PATCH;
            const q15_t  *p   = slot0;
            int           rem = PATCH;

            size_t     full_vl = __riscv_vsetvl_e8m2(PATCH);
            vint32m8_t vacc    = __riscv_vmv_v_x_i32m8(0, full_vl);

            while (rem > 0)
            {
                size_t     vl  = __riscv_vsetvl_e8m2(rem);
                vint8m2_t  vw8 = __riscv_vle8_v_i8m2(w, vl);
                vint16m4_t vw  = __riscv_vsext_vf2_i16m4(vw8, vl);
                vint16m4_t vc  = __riscv_vle16_v_i16m4(p, vl);
                vint32m8_t vp  = __riscv_vwmul_vv_i32m8(vw, vc, vl);
                vacc           = __riscv_vadd_vv_i32m8(vacc, vp, vl);
                w += vl;
                p += vl;
                rem -= (int)vl;
            }

            vint32m1_t vz = __riscv_vmv_s_x_i32m1(0, 1);
            int32_t s = (bias_data ? bias_data[co] : 0)
                        + __riscv_vmv_x_s_i32m1_i32(
                            __riscv_vredsum_vs_i32m8_i32m1(vacc, vz, full_vl));
            s         = nn_requantize(s, mult[co], shift[co]) + out_off;
            out[co]   = (int8_t)MAX(act_min, MIN(act_max, s));
        }
    }

    return 0;
}
