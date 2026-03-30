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

#include "riscv_rfft_f32.h"
#include "ee_api.h"

ee_status_t th_rfft_init_f32(riscv_rfft_fast_instance_q31 * S, int fftLenReal)
{
    ee_status_t status = EE_STATUS_OK;

    /*  Initialize the Real FFT length */
    S->fftLenRFFT = (uint16_t) fftLenReal;


    /*  Initialization of coef modifier depending on the FFT length */
    if (S->fftLenRFFT == 1024){

        S->pTwiddleRFFT = (q31_t *)rfftFastTwiddleQ31_1024;

        status = th_cfft_init_f32(&(S->Sint),512);

    }
    else if(S->fftLenRFFT == 128){

        S->pTwiddleRFFT = (q31_t *)rfftFastTwiddleQ31_128;

        status = th_cfft_init_f32(&(S->Sint),64);

    }
    else{
        status = EE_STATUS_ERROR;
    }

    return EE_STATUS_ERROR;
}
