// Modifications copyright (C) 2023 Chair of Electronic Design Automation, TUM
/*
 * SPDX-FileCopyrightText: Copyright 2022-2023 Arm Limited and/or its affiliates
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

#include "rvv_support_guard.h"
#include "functions.h"
#include "support_functions.h"
#include "convolve_config.h"

/* Folds input_offset * sum_k(weight[k][oc]) into bias, once per call.
 * weights_rvv is k-major/oc-minor, so this is unit-stride per k. */
static void
nn_fold_input_offset_s8(int32_t       *corrected_bias,
                        const int32_t *bias_data,
                        const q7_t    *weights_rvv,
                        int32_t        input_ch,
                        int32_t        output_ch,
                        int32_t        input_offset)
{
    int32_t oc_off = 0;
    while (oc_off < output_ch)
    {
        size_t vl = __riscv_vsetvl_e32m4(output_ch - oc_off);

        vint32m4_t vacc = bias_data
                              ? __riscv_vle32_v_i32m4(bias_data + oc_off, vl)
                              : __riscv_vmv_v_x_i32m4(0, vl);
        vint32m4_t vsum = __riscv_vmv_v_x_i32m4(0, vl);

        for (int32_t k = 0; k < input_ch; k++)
        {
            /* unit stride: oc is the fast-varying dim in weights_rvv */
            vint8m1_t w8 = __riscv_vle8_v_i8m1(
                weights_rvv + (size_t)k * output_ch + oc_off, vl);
            vint32m4_t w32 = __riscv_vsext_vf4_i32m4(w8, vl);
            vsum           = __riscv_vadd_vv_i32m4(vsum, w32, vl);
        }

        vsum = __riscv_vmul_vx_i32m4(vsum, input_offset, vl);
        vacc = __riscv_vadd_vv_i32m4(vacc, vsum, vl);
        __riscv_vse32_v_i32m4(corrected_bias + oc_off, vacc, vl);

        oc_off += (int32_t)vl;
    }
}

int32_t
nn_conv1x1_s8(const nn_context                  *ctx,
              const nn_conv_params              *conv_params,
              const nn_per_channel_quant_params *quant_params,
              const nn_dims                     *input_dims,
              const q7_t                        *input_data,
              const nn_dims                     *filter_dims,
              const q7_t                        *filter_data,
              const nn_dims                     *bias_dims,
              const int32_t                     *bias_data,
              const nn_dims                     *output_dims,
              q7_t                              *output_data)
{
    const int32_t  col_len    = input_dims->w * input_dims->h * input_dims->n;
    const int32_t  output_ch  = output_dims->c;
    const int32_t  input_ch   = input_dims->c;
    const int32_t  out_offset = conv_params->output_offset;
    const int32_t  act_min    = conv_params->activation.min;
    const int32_t  act_max    = conv_params->activation.max;
    const int32_t *out_mult   = quant_params->multiplier;
    const int32_t *out_shift  = quant_params->shift;

    /* scratch reused from ctx->buf, same as conv0's im2col reuse; only
     * needs output_ch int32s and 1x1 has no gather buffer to fight with */
    int32_t *corrected_bias = (int32_t *)ctx->buf;

    nn_fold_input_offset_s8(corrected_bias,
                            bias_data,
                            filter_data,
                            input_ch,
                            output_ch,
                            conv_params->input_offset);

    int32_t i_items = 0;
    for (; i_items <= col_len - NN_KERNEL_COLS; i_items += NN_KERNEL_COLS)
    {
        output_data
            = nn_mat_mult_kernel_s8_s8(filter_data,
                                       input_data + (size_t)i_items * input_ch,
                                       output_ch,
                                       out_shift,
                                       out_mult,
                                       out_offset,
                                       act_min,
                                       act_max,
                                       input_ch,
                                       corrected_bias,
                                       output_data);
    }

    /* leftover spatial positions, still no reduction, just single-row */
    for (; i_items < col_len; i_items++)
    {
        output_data
            = nn_mat_mult_core_1x1_s8(filter_data,
                                      input_data + (size_t)i_items * input_ch,
                                      output_ch,
                                      out_shift,
                                      out_mult,
                                      out_offset,
                                      act_min,
                                      act_max,
                                      input_ch,
                                      corrected_bias,
                                      output_data);
    }

    return 0;
}
