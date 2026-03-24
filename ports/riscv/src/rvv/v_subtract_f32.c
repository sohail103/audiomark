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
v_subtract_f32(ee_f32_t *p_a, ee_f32_t *p_b, ee_f32_t *p_c, uint32_t len)
{
    if (!p_a || !p_b || !p_c || len == 0)
    {
        return;
    }

    size_t vl;
    for (size_t i = 0; i < len; i += vl)
    {
        vl = __riscv_vsetvl_e32m4(len - i);

        vfloat32m4_t va = __riscv_vle32_v_f32m4(p_a + i, vl);
        vfloat32m4_t vb = __riscv_vle32_v_f32m4(p_b + i, vl);

        vfloat32m4_t vc = __riscv_vfsub_vv_f32m4(va, vb, vl);

        __riscv_vse32_v_f32m4(p_c + i, vc, vl);
    }
}

#else

void
v_subtract_f32(ee_f32_t *p_a, ee_f32_t *p_b, ee_f32_t *p_c, uint32_t len)
{
    s_riscv_subtract_f32(p_a, p_b, p_c, len);
}

#endif
