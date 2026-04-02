/*
 * Copyright (C) 2010-2022 Arm Limited or its affiliates.
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
 * Modifications copyright (C) 2021-2022 Chair of Electronic Design Automation,
 * TUM
 * Modifications copyright (C) 2026 Sohail Raj Satapathy
 */

#ifndef _MURISCV_NN_MATH_TYPES_H
#define _MURISCV_NN_MATH_TYPES_H

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

typedef int8_t  q7_t;
typedef int16_t q15_t;
typedef int32_t q31_t;

#define Q31_MAX ((q31_t)(0x7FFFFFFFL))
#define Q15_MAX ((q15_t)(0x7FFF))
#define Q7_MAX  ((q7_t)(0x7F))
#define Q31_MIN ((q31_t)(0x80000000L))
#define Q15_MIN ((q15_t)(0x8000))
#define Q7_MIN  ((q7_t)(0x80))

#define NN_Q31_MAX ((int32_t)(0x7FFFFFFFL))
#define NN_Q15_MAX ((int16_t)(0x7FFF))
#define NN_Q7_MAX  ((int8_t)(0x7F))
#define NN_Q31_MIN ((int32_t)(0x80000000L))
#define NN_Q15_MIN ((int16_t)(0x8000))
#define NN_Q7_MIN  ((int8_t)(0x80))

typedef enum
{
    MURISCV_NN_SUCCESS       = 0,  /**< No error */
    MURISCV_NN_ARG_ERROR     = -1, /**< One or more arguments are incorrect */
    MURISCV_NN_NO_IMPL_ERROR = -2, /**< No implementation available */
} muriscv_nn_status;

#endif /* _MURISCV_NN_MATH_TYPES_H */
