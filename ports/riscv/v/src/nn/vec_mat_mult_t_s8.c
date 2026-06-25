/**
 * Copyright 2026 Robin John
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

#include "support_functions.h"
#include "rvv_support_guard.h"
#include <stdint.h>

/*
 * S8 basic fully-connected and matrix multiplication layer function
 */
int32_t
nn_vec_mat_mult_t_s8(const q7_t   *lhs,
                     const q7_t   *rhs,
                     const q31_t  *bias,
                     q7_t         *dst,
                     const int32_t lhs_offset,
                     const int32_t dst_offset,
                     const int32_t dst_multiplier,
                     const int32_t dst_shift,
                     const int32_t rhs_cols,
                     const int32_t rhs_rows,
                     const int32_t activation_min,
                     const int32_t activation_max,
                     const int32_t address_offset)
{

    size_t        outputs_left = rhs_rows;
    const int8_t *weight_ptr   = rhs;
    q7_t         *dst_ptr      = dst;

    /* Pre-calculate the shift offsets */
    const int32_t left_shift  = dst_shift > 0 ? dst_shift : 0;
    const int32_t right_shift = dst_shift < 0 ? -dst_shift : 0;

    while (outputs_left > 0)
    {
        size_t     vl = __riscv_vsetvl_e32m4(outputs_left);
        vint32m4_t v_acc;

        if (bias)
        {
            v_acc = __riscv_vle32_v_i32m4(bias, vl);
            bias += vl;
        }
        else
        {
            v_acc = __riscv_vmv_v_x_i32m4(0, vl);
        }

        const int8_t *current_weight_col = weight_ptr;
        const q7_t   *lhs_ptr            = lhs;
        int           i                  = 0;

        /* Unroll by 4 for better latency hiding */
        for (; i <= rhs_cols - 4; i += 4)
        {
            int32_t raw_in0 = (int32_t)lhs_ptr[0] + lhs_offset;
            int32_t raw_in1 = (int32_t)lhs_ptr[1] + lhs_offset;
            int32_t raw_in2 = (int32_t)lhs_ptr[2] + lhs_offset;
            int32_t raw_in3 = (int32_t)lhs_ptr[3] + lhs_offset;

            vint8m1x4_t v_weight_tuple
                = __riscv_vlsseg4e8_v_i8m1x4(current_weight_col, rhs_cols, vl);

            vint8m1_t  v_w8_0  = __riscv_vget_v_i8m1x4_i8m1(v_weight_tuple, 0);
            vint16m2_t v_w16_0 = __riscv_vwcvt_x_x_v_i16m2(v_w8_0, vl);
            v_acc
                = __riscv_vwmacc_vx_i32m4(v_acc, (int16_t)raw_in0, v_w16_0, vl);

            vint8m1_t  v_w8_1  = __riscv_vget_v_i8m1x4_i8m1(v_weight_tuple, 1);
            vint16m2_t v_w16_1 = __riscv_vwcvt_x_x_v_i16m2(v_w8_1, vl);
            v_acc
                = __riscv_vwmacc_vx_i32m4(v_acc, (int16_t)raw_in1, v_w16_1, vl);

            vint8m1_t  v_w8_2  = __riscv_vget_v_i8m1x4_i8m1(v_weight_tuple, 2);
            vint16m2_t v_w16_2 = __riscv_vwcvt_x_x_v_i16m2(v_w8_2, vl);
            v_acc
                = __riscv_vwmacc_vx_i32m4(v_acc, (int16_t)raw_in2, v_w16_2, vl);

            vint8m1_t  v_w8_3  = __riscv_vget_v_i8m1x4_i8m1(v_weight_tuple, 3);
            vint16m2_t v_w16_3 = __riscv_vwcvt_x_x_v_i16m2(v_w8_3, vl);
            v_acc
                = __riscv_vwmacc_vx_i32m4(v_acc, (int16_t)raw_in3, v_w16_3, vl);

            current_weight_col += 4;
            lhs_ptr += 4;
        }

        /* Tail Handling */
        for (; i < rhs_cols; i++)
        {
            int32_t raw_val = (int32_t)(*lhs_ptr++) + lhs_offset;

            vint8m1_t v_w8_tail
                = __riscv_vlse8_v_i8m1(current_weight_col, rhs_cols, vl);
            vint16m2_t v_w16_tail = __riscv_vwcvt_x_x_v_i16m2(v_w8_tail, vl);

            v_acc = __riscv_vwmacc_vx_i32m4(
                v_acc, (int16_t)raw_val, v_w16_tail, vl);
            current_weight_col += 1;
        }

        /* Requantize */
        v_acc = __riscv_vsll_vx_i32m4(v_acc, left_shift, vl);
        v_acc = __riscv_vsmul_vx_i32m4(
            v_acc, dst_multiplier, __RISCV_VXRM_RNU, vl);
        v_acc
            = __riscv_vssra_vx_i32m4(v_acc, right_shift, __RISCV_VXRM_RNU, vl);

        v_acc = __riscv_vadd_vx_i32m4(v_acc, dst_offset, vl);
        v_acc = __riscv_vmax_vx_i32m4(v_acc, activation_min, vl);
        v_acc = __riscv_vmin_vx_i32m4(v_acc, activation_max, vl);

        vint16m2_t v_narrow_16 = __riscv_vncvt_x_x_w_i16m2(v_acc, vl);
        vint8m1_t  v_narrow_8  = __riscv_vncvt_x_x_w_i8m1(v_narrow_16, vl);

        __riscv_vsse8_v_i8m1(dst_ptr, address_offset, v_narrow_8, vl);

        outputs_left -= vl;
        dst_ptr += vl * address_offset;
        weight_ptr += vl * rhs_cols;
    }
    return 0;
}
