/*
 * Copyright (C) 2019 Nuclei Limited.
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

/* Contains functions that emulate ARM instructions that RISC-V does not have.
 */

#ifndef _MURISCV_NN_COMPATIBLE_H
#define _MURISCV_NN_COMPATIBLE_H

#include <stdint.h>

static inline uint8_t
__CLZ(uint32_t data)
{
    uint8_t  ret  = 0;
    uint32_t temp = ~data;
    while (temp & 0x80000000)
    {
        temp <<= 1;
        ret++;
    }
    return ret;
}

#endif /* _MURISCV_NN_COMPATIBLE_H */
