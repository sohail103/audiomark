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
#include "rvv_support_guard.h"
#include "ee_audiomark.h"
#include "ee_api.h"

#include <math.h>

void
th_absmax_f32(const ee_f32_t *p_in,
              uint32_t        len,
              ee_f32_t       *p_max,
              uint32_t       *p_index)
{
    ee_f32_t max_val = fabsf(p_in[0]);
    uint32_t max_idx = 0;
    uint32_t offset  = 0;

    /* this function is only called with len = 16. on full V targets (vlen>=128)
     * this loop executes exactly once. the strip mining handles narrower
     * zve32f/zve64f vlen configurations and any other len. */
    while (len > 0)
    {
        size_t vl = __riscv_vsetvl_e32m4(len);

        vfloat32m4_t v    = __riscv_vle32_v_f32m4(p_in, vl);
        vfloat32m4_t vabs = __riscv_vfabs_v_f32m4(v, vl);

        vfloat32m1_t vmax_init = __riscv_vfmv_s_f_f32m1(max_val, 1);
        vfloat32m1_t vmax_red
            = __riscv_vfredmax_vs_f32m4_f32m1(vabs, vmax_init, vl);
        ee_f32_t chunk_max = __riscv_vfmv_f_s_f32m1_f32(vmax_red);

        /* only update max_val if chunk_max is strictly greater */
        if (chunk_max > max_val)
        {
            vbool8_t    veq  = __riscv_vmfeq_vf_f32m4_b8(vabs, chunk_max, vl);
            vuint32m4_t vidx = __riscv_vid_v_u32m4(vl);
            vuint32m1_t vidx_init = __riscv_vmv_s_x_u32m1((uint32_t)vl, 1);
            vuint32m1_t vidx_red
                = __riscv_vredminu_vs_u32m4_u32m1_m(veq, vidx, vidx_init, vl);
            uint32_t local_idx = __riscv_vmv_x_s_u32m1_u32(vidx_red);

            max_val = chunk_max;
            max_idx = offset + local_idx;
        }

        p_in += vl;
        offset += (uint32_t)vl;
        len -= (uint32_t)vl;
    }

    *p_max   = max_val;
    *p_index = max_idx;
}
