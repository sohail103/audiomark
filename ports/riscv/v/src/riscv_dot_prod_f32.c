/**
 * Copyright 2026 Ayush Dwivedi
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

#include "ee_api.h"
#include "ee_audiomark.h"
#include "rvv_support_guard.h"

void
th_dot_prod_f32(ee_f32_t *p_a, ee_f32_t *p_b, uint32_t len, ee_f32_t *p_result)
{
    if (!p_a || !p_b || !p_result || len == 0)
    {
        return;
    }

    size_t      vl = 0;
    vfloat32m4_t v_sum =
      __riscv_vfmv_v_f_f32m4(0.0f, __riscv_vsetvlmax_e32m4());

    for (size_t i = 0; i < len; i += vl)
    {
        vl = __riscv_vsetvl_e32m4(len - i);

        vfloat32m4_t v_a = __riscv_vle32_v_f32m4(p_a + i, vl);
        vfloat32m4_t v_b = __riscv_vle32_v_f32m4(p_b + i, vl);
        v_sum            = __riscv_vfmacc_vv_f32m4_tu(v_sum, v_a, v_b, vl);
    }

    vfloat32m1_t v_zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    vfloat32m1_t v_reduced =
      __riscv_vfredusum_vs_f32m4_f32m1(v_sum, v_zero, __riscv_vsetvlmax_e32m4());

    *p_result = __riscv_vfmv_f_s_f32m1_f32(v_reduced);
}
