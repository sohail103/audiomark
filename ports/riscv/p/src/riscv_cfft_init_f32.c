/*
 * Copyright 2026 Robin John
 * Copyright (C) 2010-2021 ARM Limited or its affiliates. All rights reserved.
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
 */


#include "ee_api.h"
#include "riscv_cfft_f32.h"

ee_status_t th_cfft_init_f32(riscv_cfft_instance_q31 *p_instance, int fftLength){
    ee_status_t status = EE_STATUS_OK;
    p_instance->fftLen = fftLength;
    switch(fftLength){
        case 64U:
            p_instance->pTwiddle = twiddleCoef_64_q31;
            p_instance->pBitRevTable = riscvBitRevIndexTable_fixed_64;
            p_instance->bitRevLength = RISCVBITREVINDEXTABLE_FIXED_64_TABLE_LENGTH;
            p_instance->rearranged_twiddle_stride1 = rearranged_twiddle_stride1_64_q31;
            p_instance->rearranged_twiddle_stride2 = rearranged_twiddle_stride2_64_q31;
            p_instance->rearranged_twiddle_stride3 = rearranged_twiddle_stride3_64_q31;
            p_instance->rearranged_twiddle_tab_stride1_arr = rearranged_twiddle_tab_stride1_arr_64_q31;
            p_instance->rearranged_twiddle_tab_stride2_arr = rearranged_twiddle_tab_stride2_arr_64_q31;
            p_instance->rearranged_twiddle_tab_stride3_arr = rearranged_twiddle_tab_stride3_arr_64_q31;
        break;
        case 128U:
            p_instance->pTwiddle = twiddleCoef_128_q31;
            p_instance->pBitRevTable = riscvBitRevIndexTable_fixed_128;
            p_instance->bitRevLength = RISCVBITREVINDEXTABLE_FIXED_128_TABLE_LENGTH;
            p_instance->rearranged_twiddle_stride1 = rearranged_twiddle_stride1_64_q31;
            p_instance->rearranged_twiddle_stride2 = rearranged_twiddle_stride2_64_q31;
            p_instance->rearranged_twiddle_stride3 = rearranged_twiddle_stride3_64_q31;
            p_instance->rearranged_twiddle_tab_stride1_arr = rearranged_twiddle_tab_stride1_arr_64_q31;
            p_instance->rearranged_twiddle_tab_stride2_arr = rearranged_twiddle_tab_stride2_arr_64_q31;
            p_instance->rearranged_twiddle_tab_stride3_arr = rearranged_twiddle_tab_stride3_arr_64_q31;
        break;
        case 512U:
            p_instance->pTwiddle = twiddleCoef_512_q31;
            p_instance->pBitRevTable = riscvBitRevIndexTable_fixed_512;
            p_instance->bitRevLength = RISCVBITREVINDEXTABLE_FIXED_512_TABLE_LENGTH;
            p_instance->rearranged_twiddle_stride1 = rearranged_twiddle_stride1_256_q31;
            p_instance->rearranged_twiddle_stride2 = rearranged_twiddle_stride2_256_q31;
            p_instance->rearranged_twiddle_stride3 = rearranged_twiddle_stride3_256_q31;
            p_instance->rearranged_twiddle_tab_stride1_arr = rearranged_twiddle_tab_stride1_arr_256_q31;
            p_instance->rearranged_twiddle_tab_stride2_arr = rearranged_twiddle_tab_stride2_arr_256_q31;
            p_instance->rearranged_twiddle_tab_stride3_arr = rearranged_twiddle_tab_stride3_arr_256_q31;
        break;
        default:
            status = EE_STATUS_ERROR;
        break;
    }

    return EE_STATUS_OK;
}
