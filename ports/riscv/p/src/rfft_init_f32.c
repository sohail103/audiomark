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

#include "rfft_f32.h"
#include "ee_api.h"
#include "th_types.h"
#include "rvp_support_guard.h"

ee_status_t
th_rfft_init_f32(riscv_rfft_fast_instance_q31 *S, int fftLenReal)
{
    ee_status_t status = EE_STATUS_OK;

    /*  Initialize the Real FFT length */
    S->fftLenRFFT = (uint16_t)fftLenReal;

    /* Unified Twiddle Tables */
    S->pTwiddleRFFT = (q31_t *)rfftFastTwiddleQ31_1024;

    /*  Initialization of coef modifier depending on the FFT length */
    switch (fftLenReal)
    {
        case 1024U:
            status          = th_cfft_init_f32(&(S->Sint), 512);
            break;
        case 512U:
            status          = th_cfft_init_f32(&(S->Sint), 256);
            break;
        case 128U:
            status          = th_cfft_init_f32(&(S->Sint), 64);
            break;
        default:
            status = EE_STATUS_ERROR;
            break;
    }

    return status;
}
