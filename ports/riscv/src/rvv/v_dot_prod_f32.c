/**
 * Copyright (C) 2024 SPEC Embedded Group
 * Copyright (C) 2022 EEMBC
 * Copyright (C) 2022 Arm Limited
 *
 * All EEMBC Benchmark Software are products of EEMBC and are provided under the
 * terms of the EEMBC Benchmark License Agreements. The EEMBC Benchmark Software
 * are proprietary intellectual properties of EEMBC and its Members and is
 * protected under all applicable laws, including all applicable copyright laws.
 *
 * If you received this EEMBC Benchmark Software without having a currently
 * effective EEMBC Benchmark License Agreement, you must discontinue use.
 */

#include "ee_audiomark.h"
#include "ee_api.h"
#include "riscv_audiomark.h"

#if defined(__riscv_vector) && defined(__riscv_zve32f)

#include <riscv_vector.h>

void
v_dot_prod_f32(ee_f32_t *p_a,
               ee_f32_t *p_b,
               uint32_t  len,
               ee_f32_t *p_result)
{
    if (!p_a || !p_b || !p_result || len == 0)
    {
        return;
    }

    size_t vl;
    vfloat32m4_t v_sum = __riscv_vfmv_v_f_f32m4(0.0f, __riscv_vsetvlmax_e32m4());

    for (size_t i = 0; i < len; i += vl)
    {
        vl = __riscv_vsetvl_e32m4(len - i);

        vfloat32m4_t va = __riscv_vle32_v_f32m4(p_a + i, vl);
        vfloat32m4_t vb = __riscv_vle32_v_f32m4(p_b + i, vl);

        v_sum = __riscv_vfmacc_vv_f32m4_tu(v_sum, va, vb, vl);
    }

    vfloat32m1_t v_zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    vfloat32m1_t v_reduced = __riscv_vfredusum_vs_f32m4_f32m1(v_sum, v_zero, 
                                                              __riscv_vsetvlmax_e32m4());

    *p_result = __riscv_vfmv_f_s_f32m1_f32(v_reduced);
}

#else

void
v_dot_prod_f32(ee_f32_t *p_a,
               ee_f32_t *p_b,
               uint32_t  len,
               ee_f32_t *p_result)
{
    s_riscv_dot_prod_f32(p_a, p_b, len, p_result);
}

#endif
