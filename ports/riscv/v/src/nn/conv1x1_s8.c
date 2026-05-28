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

/* nn_conv1x1_s8: 1x1 pointwise conv, 125x64 -> 125x64 */

#include "functions.h"
#include "support_functions.h"
#include "ee_api.h"
#include "ee_nn.h"
#include "rvv_support_guard.h"

#include <stdint.h>

#define SPATIAL 125
#define IC      64
#define OC      64

static q15_t col_buf[4 * IC];

int32_t
nn_conv1x1_s8(const nn_conv_params              *conv_params,
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

    const int8_t *in  = input_data;
    int8_t       *out = output_data;
    int           px  = 0;

    /* widen 4 input vectors to q15 in one pass, then GEMM against all OC rows
     */
    for (; px <= SPATIAL - 4; px += 4)
    {
        /* widen 4 x IC int8 inputs to q15; e8m2->i16m4 covers IC=64 in one shot
         * at VLEN>=256 */
        for (int n = 0; n < 4; n++)
        {
            const int8_t *src = in + (px + n) * IC;
            q15_t        *dst = col_buf + n * IC;
            int           rem = IC;
            while (rem > 0)
            {
                size_t     vl  = __riscv_vsetvl_e8m2(rem);
                vint8m2_t  v8  = __riscv_vle8_v_i8m2(src, vl);
                vint16m4_t v16 = __riscv_vsext_vf2_i16m4(v8, vl);
                v16 = __riscv_vadd_vx_i16m4(v16, (int16_t)in_off, vl);
                __riscv_vse16_v_i16m4(dst, v16, vl);
                src += vl;
                dst += vl;
                rem -= (int)vl;
            }
        }

        /*
         * GEMM: for each output channel, load weight row once and accumulate
         * four dot-products.  e8m1->i16m2->i32m4 keeps two concurrent
         * product vectors within the 32-register budget.
         */
        int8_t *o0 = out + (px + 0) * OC;
        int8_t *o1 = out + (px + 1) * OC;
        int8_t *o2 = out + (px + 2) * OC;
        int8_t *o3 = out + (px + 3) * OC;

        for (int co = 0; co < OC; co++)
        {
            const int8_t *w  = filter_data + co * IC;
            int32_t       b  = bias_data ? bias_data[co] : 0;
            int32_t       s0 = b, s1 = b, s2 = b, s3 = b;
            int           rem = IC;
            const q15_t  *p0  = col_buf + 0 * IC;
            const q15_t  *p1  = col_buf + 1 * IC;
            const q15_t  *p2  = col_buf + 2 * IC;
            const q15_t  *p3  = col_buf + 3 * IC;

            while (rem > 0)
            {
                size_t     vl  = __riscv_vsetvl_e8m1(rem);
                vint8m1_t  vw8 = __riscv_vle8_v_i8m1(w, vl);
                vint16m2_t vw  = __riscv_vsext_vf2_i16m2(vw8, vl);
                vint16m2_t vc0 = __riscv_vle16_v_i16m2(p0, vl);
                vint16m2_t vc1 = __riscv_vle16_v_i16m2(p1, vl);
                vint16m2_t vc2 = __riscv_vle16_v_i16m2(p2, vl);
                vint16m2_t vc3 = __riscv_vle16_v_i16m2(p3, vl);
                vint32m4_t vp0 = __riscv_vwmul_vv_i32m4(vw, vc0, vl);
                vint32m4_t vp1 = __riscv_vwmul_vv_i32m4(vw, vc1, vl);
                vint32m4_t vp2 = __riscv_vwmul_vv_i32m4(vw, vc2, vl);
                vint32m4_t vp3 = __riscv_vwmul_vv_i32m4(vw, vc3, vl);
                vint32m1_t vz  = __riscv_vmv_s_x_i32m1(0, 1);
                s0 += __riscv_vmv_x_s_i32m1_i32(
                    __riscv_vredsum_vs_i32m4_i32m1(vp0, vz, vl));
                s1 += __riscv_vmv_x_s_i32m1_i32(
                    __riscv_vredsum_vs_i32m4_i32m1(vp1, vz, vl));
                s2 += __riscv_vmv_x_s_i32m1_i32(
                    __riscv_vredsum_vs_i32m4_i32m1(vp2, vz, vl));
                s3 += __riscv_vmv_x_s_i32m1_i32(
                    __riscv_vredsum_vs_i32m4_i32m1(vp3, vz, vl));
                w += vl;
                p0 += vl;
                p1 += vl;
                p2 += vl;
                p3 += vl;
                rem -= (int)vl;
            }

            int32_t m = mult[co], sh = shift[co];
            s0     = nn_requantize(s0, m, sh) + out_off;
            s1     = nn_requantize(s1, m, sh) + out_off;
            s2     = nn_requantize(s2, m, sh) + out_off;
            s3     = nn_requantize(s3, m, sh) + out_off;
            o0[co] = (int8_t)MAX(act_min, MIN(act_max, s0));
            o1[co] = (int8_t)MAX(act_min, MIN(act_max, s1));
            o2[co] = (int8_t)MAX(act_min, MIN(act_max, s2));
            o3[co] = (int8_t)MAX(act_min, MIN(act_max, s3));
        }
    }

    /* SPATIAL=125: one pixel remains; e8m2->i16m4->i32m8 covers IC=64 in one
     * shot */
    if (px < SPATIAL)
    {
        const int8_t *src = in + px * IC;
        q15_t        *dst = col_buf;
        int           rem = IC;
        while (rem > 0)
        {
            size_t     vl  = __riscv_vsetvl_e8m2(rem);
            vint8m2_t  v8  = __riscv_vle8_v_i8m2(src, vl);
            vint16m4_t v16 = __riscv_vsext_vf2_i16m4(v8, vl);
            v16            = __riscv_vadd_vx_i16m4(v16, (int16_t)in_off, vl);
            __riscv_vse16_v_i16m4(dst, v16, vl);
            src += vl;
            dst += vl;
            rem -= (int)vl;
        }

        int8_t *o0 = out + px * OC;
        for (int co = 0; co < OC; co++)
        {
            const int8_t *w = filter_data + co * IC;
            int32_t       s = bias_data ? bias_data[co] : 0;
            rem             = IC;
            const q15_t *p  = col_buf;
            while (rem > 0)
            {
                size_t     vl  = __riscv_vsetvl_e8m2(rem);
                vint8m2_t  vw8 = __riscv_vle8_v_i8m2(w, vl);
                vint16m4_t vw  = __riscv_vsext_vf2_i16m4(vw8, vl);
                vint16m4_t vc  = __riscv_vle16_v_i16m4(p, vl);
                vint32m8_t vp  = __riscv_vwmul_vv_i32m8(vw, vc, vl);
                vint32m1_t vz  = __riscv_vmv_s_x_i32m1(0, 1);
                s += __riscv_vmv_x_s_i32m1_i32(
                    __riscv_vredsum_vs_i32m8_i32m1(vp, vz, vl));
                w += vl;
                p += vl;
                rem -= (int)vl;
            }
            s      = nn_requantize(s, mult[co], shift[co]) + out_off;
            o0[co] = (int8_t)MAX(act_min, MIN(act_max, s));
        }
    }

    return 0;
}
