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

/* ----------------------------------------------------------------------
 * Project:      MURISCV NN Library
 * Title:        muriscv_nn_support_functions.h
 * Description:  Public header file of support functions for MURISCV NN Library
 *
 * $Date:        08 October 2024
 * $Revision:    V.22.4.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#ifndef MURISCV_NNSUPPORT_FUNCTIONS_H
#define MURISCV_NNSUPPORT_FUNCTIONS_H

#include "muriscv_nn_compiler.h"
#include "muriscv_nn_math_types.h"
#include "muriscv_nn_types.h"
#include "muriscv_nn_util.h"
#include "ee_api.h"

#define LEFT_SHIFT(_shift)            (_shift > 0 ? _shift : 0)
#define RIGHT_SHIFT(_shift)           (_shift > 0 ? 0 : -_shift)
#define MASK_IF_ZERO(x)               (x) == 0 ? ~0 : 0
#define MASK_IF_NON_ZERO(x)           (x) != 0 ? ~0 : 0
#define SELECT_USING_MASK(mask, a, b) ((mask) & (a)) ^ (~(mask) & (b))

#define MAX_RV(A, B) ((A) > (B) ? (A) : (B))
#define MIN_RV(A, B) ((A) < (B) ? (A) : (B))

#define MAX(A, B) MAX_RV(A, B)

#define MIN(A, B)      MIN_RV(A, B)
#define CLAMP(x, h, l) MAX(MIN((x), (h)), (l))
#define REDUCE_MULTIPLIER(_mult) \
    ((_mult < 0x7FFF0000) ? ((_mult + (1 << 15)) >> 16) : 0x7FFF)

// Number of channels processed in a block for DW Conv with Int8 weights(MVE)
// Requirement: Greater than 0 & less than 128
// This can be fine tuned to match number of input channels for best
// performance. A layer with lower number of channels than CH_IN_BLOCK_MVE will
// result in higher scratch buffer usage and a layer with higher number of
// channels than CH_IN_BLOCK_MVE will result in lower scratch buffer usage.
#define CH_IN_BLOCK_MVE (124)

#define PACK_S8x4_32x1(v0, v1, v2, v3) PACK_Q7x4_32x1(v0, v1, v2, v3)
// MURISCV_NN NEW CODE

#define PACK_Q7x4_32x1(v0, v1, v2, v3)               \
    ((((int32_t)(v0) << 0) & (int32_t)0x000000FF)    \
     | (((int32_t)(v1) << 8) & (int32_t)0x0000FF00)  \
     | (((int32_t)(v2) << 16) & (int32_t)0x00FF0000) \
     | (((int32_t)(v3) << 24) & (int32_t)0xFF000000))

#define PACK_Q15x2_32x1(v0, v1) \
    (((int32_t)v0 & (int32_t)0xFFFF) | ((int32_t)v1 << 16))

union muriscv_nn_word
{
    int32_t word;
    /**< q31 type */
    int16_t half_words[2];
    /**< s16 type */
    int8_t bytes[4];
    /**< s8 type */
};

struct muriscv_nn_double
{
    uint32_t low;
    int32_t  high;
};

union muriscv_nn_long_long
{
    int64_t                  long_long;
    struct muriscv_nn_double word;
};

void muriscv_nn_q7_to_q15_with_offset(const int8_t *src,
                                      int16_t      *dst,
                                      int32_t       block_size,
                                      int16_t       offset);

muriscv_nn_status muriscv_nn_mat_mult_nt_t_s8(const int8_t  *lhs,
                                              const int8_t  *rhs,
                                              const int32_t *bias,
                                              int8_t        *dst,
                                              const int32_t *dst_multipliers,
                                              const int32_t *dst_shifts,
                                              const int32_t  lhs_rows,
                                              const int32_t  rhs_rows,
                                              const int32_t  rhs_cols,
                                              const int32_t  lhs_offset,
                                              const int32_t  dst_offset,
                                              const int32_t  activation_min,
                                              const int32_t  activation_max,
                                              const int32_t  row_address_offset,
                                              const int32_t  lhs_cols_offset);

muriscv_nn_status muriscv_nn_vec_mat_mult_t_s8(const int8_t  *lhs,
                                               const int8_t  *rhs,
                                               const int32_t *kernel_sum,
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
                                               const int32_t  address_offset,
                                               const int32_t  rhs_offset);

static inline q31_t
muriscv_nn_read_q7x4(const q7_t *in_q7)
{
    q31_t val;
    val = (*((uint32_t *)(in_q7)));

    return val;
}

static inline void
muriscv_nn_memcpy(int8_t *dst, const int8_t *src, size_t block_size)
{
    th_memcpy(dst, src, block_size);
}

static inline void
muriscv_nn_memset(int8_t *dst, const int8_t val, size_t block_size)
{
    th_memset(dst, val, block_size);
}

__STATIC_FORCEINLINE void
muriscv_nn_memset_s8(int8_t *dst, const int8_t val, uint32_t block_size)
{
    th_memset(dst, val, block_size);
}

q7_t *muriscv_nn_mat_mult_kernel_s8_s16(const q7_t          *input_a,
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

void muriscv_nn_softmax_common_s8(const int8_t *input,
                                  const int32_t num_rows,
                                  const int32_t row_size,
                                  const int32_t mult,
                                  const int32_t shift,
                                  const int32_t diff_min,
                                  const bool    int16_output,
                                  void         *output);

#ifndef MURISCV_NN_TRUNCATE
#define NN_ROUND(out_shift) ((0x1 << out_shift) >> 1)
#else
#define NN_ROUND(out_shift) 0
#endif

// Macros for shortening quantization functions' names and avoid long lines
#define MUL_SAT(a, b)      muriscv_nn_doubling_high_mult((a), (b))
#define MUL_POW2(a, b)     muriscv_nn_mult_by_power_of_two((a), (b))
#define DIV_POW2(a, b)     muriscv_nn_divide_by_power_of_two((a), (b))
#define DIV_POW2_MVE(a, b) muriscv_nn_divide_by_power_of_two_mve((a), (b))
#define EXP_ON_NEG(x)      muriscv_nn_exp_on_negative_values((x))
#define ONE_OVER1(x)       muriscv_nn_one_over_one_plus_x_for_x_in_0_1((x))

__STATIC_FORCEINLINE int32_t
muriscv_nn_doubling_high_mult(const int32_t m1, const int32_t m2)
{
    int32_t result = 0;
    // Rounding offset to add for a right shift of 31
    int64_t mult = 1 << 30;

    if ((m1 < 0) ^ (m2 < 0))
    {
        mult = 1 - mult;
    }
    // Gets resolved as a SMLAL instruction
    mult = mult + (int64_t)m1 * m2;

    // Utilize all of the upper 32 bits. This is the doubling step
    // as well.
    result = (int32_t)(mult / (1ll << 31));

    if ((m1 == m2) && (m1 == (int32_t)NN_Q31_MIN))
    {
        result = NN_Q31_MAX;
    }
    return result;
}

__STATIC_FORCEINLINE int32_t
muriscv_nn_doubling_high_mult_no_sat(const int32_t m1, const int32_t m2)
{
    int32_t                    result = 0;
    union muriscv_nn_long_long mult;

    // Rounding offset to add for a right shift of 31
    mult.word.low  = 1 << 30;
    mult.word.high = 0;

    // Gets resolved as a SMLAL instruction
    mult.long_long = mult.long_long + (int64_t)m1 * m2;

    // Utilize all of the upper 32 bits. This is the doubling step
    // as well.
    result = (int32_t)(mult.long_long >> 31);

    return result;
}

__STATIC_FORCEINLINE int32_t
muriscv_nn_divide_by_power_of_two(const int32_t dividend,
                                  const int32_t exponent)
{
    int32_t       result         = 0;
    const int32_t remainder_mask = (1 << exponent) - 1;
    int32_t       remainder      = remainder_mask & dividend;

    // Basic division
    result = dividend >> exponent;

    // Adjust 'result' for rounding (mid point away from zero)
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

__STATIC_FORCEINLINE int32_t
muriscv_nn_requantize(const int32_t val,
                      const int32_t multiplier,
                      const int32_t shift)
{
#ifdef MURISCV_NN_USE_SINGLE_ROUNDING
    const int64_t total_shift = 31 - shift;
    const int64_t new_val     = val * (int64_t)multiplier;

    int32_t result = new_val >> (total_shift - 1);
    result         = (result + 1) >> 1;

    return result;
#else
    return muriscv_nn_divide_by_power_of_two(
        muriscv_nn_doubling_high_mult_no_sat(val * (1 << LEFT_SHIFT(shift)),
                                             multiplier),
        RIGHT_SHIFT(shift));
#endif
}

__STATIC_FORCEINLINE void
muriscv_nn_memcpy_s8(int8_t *dst, const int8_t *src, size_t block_size)
{
    th_memcpy(dst, src, block_size);
}

__STATIC_FORCEINLINE void
muriscv_nn_memcpy_q15(int16_t *dst, const int16_t *src, uint32_t block_size)
{
    th_memcpy(dst, src, block_size);
}

// @note The following functions are used only for softmax layer, scaled
// bits = 5 assumed

__STATIC_FORCEINLINE int32_t
muriscv_nn_exp_on_negative_values(int32_t val)
{
    int32_t mask  = 0;
    int32_t shift = 24;

    const int32_t val_mod_minus_quarter
        = (val & ((1 << shift) - 1)) - (1 << shift);
    const int32_t remainder = val_mod_minus_quarter - val;
    const int32_t x         = (val_mod_minus_quarter << 5) + (1 << 28);
    const int32_t x2        = MUL_SAT(x, x);

    int32_t result
        = 1895147668
          + MUL_SAT(1895147668,
                    x
                        + DIV_POW2(MUL_SAT(DIV_POW2(MUL_SAT(x2, x2), 2)
                                               + MUL_SAT(x2, x),
                                           715827883)
                                       + x2,
                                   1));

#define SELECT_IF_NON_ZERO(x)                                         \
    {                                                                 \
        mask   = MASK_IF_NON_ZERO(remainder & (1 << shift++));        \
        result = SELECT_USING_MASK(mask, MUL_SAT(result, x), result); \
    }

    SELECT_IF_NON_ZERO(1672461947)
    SELECT_IF_NON_ZERO(1302514674)
    SELECT_IF_NON_ZERO(790015084)
    SELECT_IF_NON_ZERO(290630308)
    SELECT_IF_NON_ZERO(39332535)
    SELECT_IF_NON_ZERO(720401)
    SELECT_IF_NON_ZERO(242)

#undef SELECT_IF_NON_ZERO

    mask = MASK_IF_ZERO(val);
    return SELECT_USING_MASK(mask, NN_Q31_MAX, result);
}

__STATIC_FORCEINLINE int32_t
muriscv_nn_mult_by_power_of_two(const int32_t val, const int32_t exp)
{
    const int32_t thresh = ((1 << (31 - exp)) - 1);
    int32_t       result = val << exp;
    result
        = SELECT_USING_MASK(MASK_IF_NON_ZERO(val > thresh), NN_Q31_MAX, result);
    result = SELECT_USING_MASK(
        MASK_IF_NON_ZERO(val < -thresh), NN_Q31_MIN, result);
    return result;
}

__STATIC_FORCEINLINE int32_t
muriscv_nn_one_over_one_plus_x_for_x_in_0_1(int32_t val)
{
    const int64_t sum = (int64_t)val + (int64_t)NN_Q31_MAX;
    const int32_t half_denominator
        = (int32_t)((sum + (sum >= 0 ? 1 : -1)) / 2L);
    int32_t x = 1515870810 + MUL_SAT(half_denominator, -1010580540);

    const int32_t shift = (1 << 29);
    x += MUL_POW2(MUL_SAT(x, shift - MUL_SAT(half_denominator, x)), 2);
    x += MUL_POW2(MUL_SAT(x, shift - MUL_SAT(half_denominator, x)), 2);
    x += MUL_POW2(MUL_SAT(x, shift - MUL_SAT(half_denominator, x)), 2);

    return MUL_POW2(x, 1);
}

#endif /* MURISCV_NNSUPPORT_FUNCTIONS_H */
