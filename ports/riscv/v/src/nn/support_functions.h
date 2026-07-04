// Modifications copyright (C) 2024 Chair of Electronic Design Automation, TUM
/*
 * SPDX-FileCopyrightText: Copyright 2010-2024 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
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
 * Modifications copyright (C) 2026 Sohail Raj Satapathy
 */

#ifndef NN_SUPPORT_FUNCTIONS_H
#define NN_SUPPORT_FUNCTIONS_H

#include "rvv_support_guard.h"
#include "math_types.h"
#include "types.h"

#define LEFT_SHIFT(_shift)            (_shift > 0 ? _shift : 0)
#define MASK_IF_ZERO(x)               (x) == 0 ? ~0 : 0
#define MASK_IF_NON_ZERO(x)           (x) != 0 ? ~0 : 0
#define SELECT_USING_MASK(mask, a, b) ((mask) & (a)) ^ (~(mask) & (b))
#define MUL_POW2(a, b)                nn_mult_by_power_of_two((a), (b))
#define CLAMP(x, h, l)                MAX(MIN((x), (h)), (l))

#define MAX(a, b)           ((a) > (b) ? (a) : (b))
#define MAX3(a, b, c)       MAX(MAX(a, b), c)
#define MAX4(a, b, c, d)    MAX(MAX3(a, b, c), d)
#define MAX5(a, b, c, d, e) MAX(MAX4(a, b, c, d), e)

#define MIN(A, B) ((A) < (B) ? (A) : (B))

union nn_word
{
    int32_t word;
    /**< q31 type */
    int16_t half_words[2];
    /**< s16 type */
    int8_t bytes[4];
    /**< s8 type */
};

struct nn_double
{
    uint32_t low;
    int32_t  high;
};

union nn_long_long
{
    int64_t          long_long;
    struct nn_double word;
};

void nn_q7_to_q15_with_offset(const int8_t *src,
                              int16_t      *dst,
                              int32_t       block_size,
                              int16_t       offset);

int32_t nn_vec_mat_mult_t_s8(const int8_t  *lhs,
                             const int8_t  *rhs,
                             const int32_t *bias,
                             int8_t        *dst,
                             const int32_t  lhs_offset,
                             const int32_t  dst_offset,
                             const int32_t  dst_multiplier,
                             const int32_t  dst_shift,
                             const int32_t  rhs_cols,
                             const int32_t  rhs_rows,
                             const int32_t  activation_min,
                             const int32_t  activation_max,
                             const int32_t  address_offset);

q7_t *nn_mat_mult_kernel_s8_s16(const q7_t          *input_a,
                                const q15_t         *input_b,
                                const uint16_t       output_ch,
                                const int32_t       *out_shift,
                                const int32_t       *out_mult,
                                const int32_t        out_offset,
                                const int16_t        activation_min,
                                const int16_t        activation_max,
                                const uint16_t       num_col_a,
                                const int32_t *const output_bias,
                                q7_t                *out_0);

q7_t *nn_mat_mult_kernel_s8_s8(const q7_t          *input_a,
                               const q7_t          *input_b,
                               const uint16_t       output_ch,
                               const int32_t       *out_shift,
                               const int32_t       *out_mult,
                               const int32_t        out_offset,
                               const int16_t        activation_min,
                               const int16_t        activation_max,
                               const uint16_t       num_col_a,
                               const int32_t *const output_bias,
                               q7_t                *out_0);

q7_t *nn_mat_mult_core_1x1_s8(const q7_t          *input_a,
                              const q7_t          *act_row,
                              const uint16_t       output_ch,
                              const int32_t       *out_shift,
                              const int32_t       *out_mult,
                              const int32_t        out_offset,
                              const int16_t        activation_min,
                              const int16_t        activation_max,
                              const uint16_t       num_col_a,
                              const int32_t *const bias,
                              q7_t                *out);

#define RIGHT_SHIFT(_shift) (_shift > 0 ? 0 : -_shift)

/* Macros for shortening quantization functions' names and avoid long lines */
#define MUL_SAT(a, b)  nn_doubling_high_mult((a), (b))
#define DIV_POW2(a, b) nn_divide_by_power_of_two((a), (b))

static inline int32_t
nn_doubling_high_mult(const int32_t m1, const int32_t m2)
{
    int32_t result = 0;
    /* Rounding offset to add for a right shift of 31 */
    int64_t mult = 1 << 30;

    if ((m1 < 0) ^ (m2 < 0))
    {
        mult = 1 - mult;
    }

    /* Gets resolved as a SMLAL instruction */
    mult = mult + (int64_t)m1 * m2;

    /* Utilize all of the upper 32 bits. This is the doubling step as well. */
    result = (int32_t)(mult / (1ll << 31));

    if ((m1 == m2) && (m1 == (int32_t)NN_Q31_MIN))
    {
        result = NN_Q31_MAX;
    }
    return result;
}

static inline int32_t
nn_doubling_high_mult_no_sat(const int32_t m1, const int32_t m2)
{
    int32_t            result = 0;
    union nn_long_long mult;

    /* Rounding offset to add for a right shift of 31 */
    mult.word.low  = 1 << 30;
    mult.word.high = 0;

    /* Gets resolved as a SMLAL instruction */
    mult.long_long = mult.long_long + (int64_t)m1 * m2;

    /* Utilize all of the upper 32 bits. This is the doubling step as well. */
    result = (int32_t)(mult.long_long >> 31);

    return result;
}

static inline int32_t
nn_divide_by_power_of_two(const int32_t dividend, const int32_t exponent)
{
    int32_t       result         = 0;
    const int32_t remainder_mask = (1 << exponent) - 1;
    int32_t       remainder      = remainder_mask & dividend;

    /* Basic division */
    result = dividend >> exponent;

    /* Adjust 'result' for rounding (mid point away from zero) */
    int32_t threshold = remainder_mask >> 1;
    if (result < 0)
    {
        threshold++;
    }
    if (remainder > threshold)
    {
        result++;
    }

    return result;
}

static inline int32_t
nn_requantize(const int32_t val, const int32_t multiplier, const int32_t shift)
{
    return nn_divide_by_power_of_two(
        nn_doubling_high_mult_no_sat(val * (1 << LEFT_SHIFT(shift)),
                                     multiplier),
        RIGHT_SHIFT(shift));
}

static inline vint32m4_t
nn_requantize_vint32m4(const vint32m4_t val,
                       const q31_t      multiplier,
                       const q31_t      shift,
                       size_t           vl)
{
    vint32m4_t val_internal = __riscv_vsmul_vx_i32m4(
        val, multiplier * (1 << LEFT_SHIFT(shift)), __RISCV_VXRM_RNU, vl);

    const q31_t remainder_mask = (1 << RIGHT_SHIFT(shift)) - 1;
    vint32m4_t  remainder
        = __riscv_vand_vx_i32m4(val_internal, remainder_mask, vl);
    vint32m4_t threshold = __riscv_vmv_v_x_i32m4(remainder_mask >> 1, vl);

    val_internal = __riscv_vsra_vx_i32m4(val_internal, RIGHT_SHIFT(shift), vl);

    vbool8_t mask = __riscv_vmslt_vx_i32m4_b8(val_internal, 0, vl);
    threshold = __riscv_vadd_vx_i32m4_tum(mask, threshold, threshold, 1, vl);

    mask = __riscv_vmsgt_vv_i32m4_b8(remainder, threshold, vl);
    val_internal
        = __riscv_vadd_vx_i32m4_tum(mask, val_internal, val_internal, 1, vl);

    return val_internal;
}

#endif
