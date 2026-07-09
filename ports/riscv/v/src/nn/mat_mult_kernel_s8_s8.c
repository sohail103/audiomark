/*
 * Copyright (C) 2010-2022 Arm Limited or its affiliates.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Modifications copyright (C) 2021-2023 Chair of Electronic Design Automation,
 * TUM
 * Modifications copyright (C) 2026 Sohail Raj Satapathy
 */

#include "rvv_support_guard.h"
#include "rvv_support_functions.h"

/* Same shape as nn_mat_mult_kernel_s8_s16: LMUL4 i32 accumulators x7 rows,
 * LMUL2 i16 widened weight vector, 7 scalar activations broadcast per k.
 * input_b rows are read straight from NHWC input_data — no im2col copy,
 * no offset add, since input_offset is already folded into bias. */
q7_t *
nn_mat_mult_kernel_s8_s8(const q7_t          *input_a,
                         const q7_t          *input_b,
                         const uint16_t       output_ch,
                         const int32_t       *out_shift,
                         const int32_t       *out_mult,
                         const int32_t        out_offset,
                         const int16_t        activation_min,
                         const int16_t        activation_max,
                         const uint16_t       num_col_a,
                         const int32_t *const output_bias,
                         q7_t                *out_0)
{
    q7_t *out_1 = out_0 + output_ch;
    q7_t *out_2 = out_1 + output_ch;
    q7_t *out_3 = out_2 + output_ch;
    q7_t *out_4 = out_3 + output_ch;
    q7_t *out_5 = out_4 + output_ch;
    q7_t *out_6 = out_5 + output_ch;

    const q7_t    *ip_b0      = input_b;
    const q7_t    *ip_b1      = ip_b0 + num_col_a;
    const q7_t    *ip_b2      = ip_b1 + num_col_a;
    const q7_t    *ip_b3      = ip_b2 + num_col_a;
    const q7_t    *ip_b4      = ip_b3 + num_col_a;
    const q7_t    *ip_b5      = ip_b4 + num_col_a;
    const q7_t    *ip_b6      = ip_b5 + num_col_a;
    const q7_t    *ip_a_strip = input_a;
    const int32_t *bias_ptr   = output_bias;
    const int32_t *mult_ptr   = out_mult;
    const int32_t *shift_ptr  = out_shift;
    size_t         remaining  = output_ch;

    while (remaining > 0)
    {
        size_t vl = __riscv_vsetvl_e32m4(remaining);

        vint32m4_t vbias  = __riscv_vle32_v_i32m4(bias_ptr, vl);
        vint32m4_t vacc_0 = vbias, vacc_1 = vbias, vacc_2 = vbias,
                   vacc_3 = vbias;
        vint32m4_t vacc_4 = vbias, vacc_5 = vbias, vacc_6 = vbias;
        bias_ptr += vl;

        for (uint16_t k = 0; k < num_col_a; k++)
        {
            vint8m1_t va_i8 = __riscv_vle8_v_i8m1(
                (const int8_t *)(ip_a_strip + (size_t)k * output_ch), vl);
            vint16m2_t va_i16 = __riscv_vsext_vf2_i16m2(va_i8, vl);

            int16_t b0 = (int16_t)ip_b0[k];
            int16_t b1 = (int16_t)ip_b1[k];
            int16_t b2 = (int16_t)ip_b2[k];
            int16_t b3 = (int16_t)ip_b3[k];
            int16_t b4 = (int16_t)ip_b4[k];
            int16_t b5 = (int16_t)ip_b5[k];
            int16_t b6 = (int16_t)ip_b6[k];

            vacc_0 = __riscv_vwmacc_vx_i32m4(vacc_0, b0, va_i16, vl);
            vacc_1 = __riscv_vwmacc_vx_i32m4(vacc_1, b1, va_i16, vl);
            vacc_2 = __riscv_vwmacc_vx_i32m4(vacc_2, b2, va_i16, vl);
            vacc_3 = __riscv_vwmacc_vx_i32m4(vacc_3, b3, va_i16, vl);
            vacc_4 = __riscv_vwmacc_vx_i32m4(vacc_4, b4, va_i16, vl);
            vacc_5 = __riscv_vwmacc_vx_i32m4(vacc_5, b5, va_i16, vl);
            vacc_6 = __riscv_vwmacc_vx_i32m4(vacc_6, b6, va_i16, vl);
        }

        vint32m4_t vmult   = __riscv_vle32_v_i32m4(mult_ptr, vl);
        vint32m4_t vshift  = __riscv_vle32_v_i32m4(shift_ptr, vl);
        vint32m4_t vleft_s = __riscv_vmax_vx_i32m4(vshift, 0, vl);
        vint32m4_t vright_s
            = __riscv_vmax_vx_i32m4(__riscv_vneg_v_i32m4(vshift, vl), 0, vl);
        vuint32m4_t vleft_u  = __riscv_vreinterpret_v_i32m4_u32m4(vleft_s);
        vuint32m4_t vright_u = __riscv_vreinterpret_v_i32m4_u32m4(vright_s);

        vacc_0 = __riscv_vsll_vv_i32m4(vacc_0, vleft_u, vl);
        vacc_1 = __riscv_vsll_vv_i32m4(vacc_1, vleft_u, vl);
        vacc_2 = __riscv_vsll_vv_i32m4(vacc_2, vleft_u, vl);
        vacc_3 = __riscv_vsll_vv_i32m4(vacc_3, vleft_u, vl);
        vacc_4 = __riscv_vsll_vv_i32m4(vacc_4, vleft_u, vl);
        vacc_5 = __riscv_vsll_vv_i32m4(vacc_5, vleft_u, vl);
        vacc_6 = __riscv_vsll_vv_i32m4(vacc_6, vleft_u, vl);

        vint32m4_t vr0
            = __riscv_vsmul_vv_i32m4(vacc_0, vmult, __RISCV_VXRM_RNU, vl);
        vint32m4_t vr1
            = __riscv_vsmul_vv_i32m4(vacc_1, vmult, __RISCV_VXRM_RNU, vl);
        vint32m4_t vr2
            = __riscv_vsmul_vv_i32m4(vacc_2, vmult, __RISCV_VXRM_RNU, vl);
        vint32m4_t vr3
            = __riscv_vsmul_vv_i32m4(vacc_3, vmult, __RISCV_VXRM_RNU, vl);
        vint32m4_t vr4
            = __riscv_vsmul_vv_i32m4(vacc_4, vmult, __RISCV_VXRM_RNU, vl);
        vint32m4_t vr5
            = __riscv_vsmul_vv_i32m4(vacc_5, vmult, __RISCV_VXRM_RNU, vl);
        vint32m4_t vr6
            = __riscv_vsmul_vv_i32m4(vacc_6, vmult, __RISCV_VXRM_RNU, vl);

        vr0 = __riscv_vssra_vv_i32m4(vr0, vright_u, __RISCV_VXRM_RNU, vl);
        vr1 = __riscv_vssra_vv_i32m4(vr1, vright_u, __RISCV_VXRM_RNU, vl);
        vr2 = __riscv_vssra_vv_i32m4(vr2, vright_u, __RISCV_VXRM_RNU, vl);
        vr3 = __riscv_vssra_vv_i32m4(vr3, vright_u, __RISCV_VXRM_RNU, vl);
        vr4 = __riscv_vssra_vv_i32m4(vr4, vright_u, __RISCV_VXRM_RNU, vl);
        vr5 = __riscv_vssra_vv_i32m4(vr5, vright_u, __RISCV_VXRM_RNU, vl);
        vr6 = __riscv_vssra_vv_i32m4(vr6, vright_u, __RISCV_VXRM_RNU, vl);

        vr0 = __riscv_vadd_vx_i32m4(vr0, out_offset, vl);
        vr1 = __riscv_vadd_vx_i32m4(vr1, out_offset, vl);
        vr2 = __riscv_vadd_vx_i32m4(vr2, out_offset, vl);
        vr3 = __riscv_vadd_vx_i32m4(vr3, out_offset, vl);
        vr4 = __riscv_vadd_vx_i32m4(vr4, out_offset, vl);
        vr5 = __riscv_vadd_vx_i32m4(vr5, out_offset, vl);
        vr6 = __riscv_vadd_vx_i32m4(vr6, out_offset, vl);

        vint16m2_t vo0 = __riscv_vnclip_wx_i16m2(vr0, 0, __RISCV_VXRM_RNU, vl);
        vint16m2_t vo1 = __riscv_vnclip_wx_i16m2(vr1, 0, __RISCV_VXRM_RNU, vl);
        vint16m2_t vo2 = __riscv_vnclip_wx_i16m2(vr2, 0, __RISCV_VXRM_RNU, vl);
        vint16m2_t vo3 = __riscv_vnclip_wx_i16m2(vr3, 0, __RISCV_VXRM_RNU, vl);
        vint16m2_t vo4 = __riscv_vnclip_wx_i16m2(vr4, 0, __RISCV_VXRM_RNU, vl);
        vint16m2_t vo5 = __riscv_vnclip_wx_i16m2(vr5, 0, __RISCV_VXRM_RNU, vl);
        vint16m2_t vo6 = __riscv_vnclip_wx_i16m2(vr6, 0, __RISCV_VXRM_RNU, vl);

        vo0 = __riscv_vmax_vx_i16m2(vo0, activation_min, vl);
        vo0 = __riscv_vmin_vx_i16m2(vo0, activation_max, vl);
        vo1 = __riscv_vmax_vx_i16m2(vo1, activation_min, vl);
        vo1 = __riscv_vmin_vx_i16m2(vo1, activation_max, vl);
        vo2 = __riscv_vmax_vx_i16m2(vo2, activation_min, vl);
        vo2 = __riscv_vmin_vx_i16m2(vo2, activation_max, vl);
        vo3 = __riscv_vmax_vx_i16m2(vo3, activation_min, vl);
        vo3 = __riscv_vmin_vx_i16m2(vo3, activation_max, vl);
        vo4 = __riscv_vmax_vx_i16m2(vo4, activation_min, vl);
        vo4 = __riscv_vmin_vx_i16m2(vo4, activation_max, vl);
        vo5 = __riscv_vmax_vx_i16m2(vo5, activation_min, vl);
        vo5 = __riscv_vmin_vx_i16m2(vo5, activation_max, vl);
        vo6 = __riscv_vmax_vx_i16m2(vo6, activation_min, vl);
        vo6 = __riscv_vmin_vx_i16m2(vo6, activation_max, vl);

        __riscv_vse8_v_i8m1(
            (int8_t *)out_0,
            __riscv_vnclip_wx_i8m1(vo0, 0, __RISCV_VXRM_RNU, vl),
            vl);
        __riscv_vse8_v_i8m1(
            (int8_t *)out_1,
            __riscv_vnclip_wx_i8m1(vo1, 0, __RISCV_VXRM_RNU, vl),
            vl);
        __riscv_vse8_v_i8m1(
            (int8_t *)out_2,
            __riscv_vnclip_wx_i8m1(vo2, 0, __RISCV_VXRM_RNU, vl),
            vl);
        __riscv_vse8_v_i8m1(
            (int8_t *)out_3,
            __riscv_vnclip_wx_i8m1(vo3, 0, __RISCV_VXRM_RNU, vl),
            vl);
        __riscv_vse8_v_i8m1(
            (int8_t *)out_4,
            __riscv_vnclip_wx_i8m1(vo4, 0, __RISCV_VXRM_RNU, vl),
            vl);
        __riscv_vse8_v_i8m1(
            (int8_t *)out_5,
            __riscv_vnclip_wx_i8m1(vo5, 0, __RISCV_VXRM_RNU, vl),
            vl);
        __riscv_vse8_v_i8m1(
            (int8_t *)out_6,
            __riscv_vnclip_wx_i8m1(vo6, 0, __RISCV_VXRM_RNU, vl),
            vl);

        out_0 += vl;
        out_1 += vl;
        out_2 += vl;
        out_3 += vl;
        out_4 += vl;
        out_5 += vl;
        out_6 += vl;
        mult_ptr += vl;
        shift_ptr += vl;
        ip_a_strip += vl;
        remaining -= vl;
    }

    return out_6; /* one past 7th row, ready for next call to continue writing
                   */
}
