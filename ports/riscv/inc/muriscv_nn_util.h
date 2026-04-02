/*
 * Copyright (C) 2021-2022 Chair of Electronic Design Automation, TUM.
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

#ifndef _MURISCV_NN_UTIL_H
#define _MURISCV_NN_UTIL_H

#include "muriscv_nn_math_types.h"
#include "muriscv_nn_types.h"

#include <stdio.h>

typedef enum
{
    RNU = 0,
    /**< round-to-nearest-up */
    RNE = 1,
    /**< round-to-nearest-even */
    RDN = 2,
    /**< round-down */
    ROD = 3,
    /**< round-to-odd */
} muriscv_nn_rounding_mode;

/**
 * Sets the RVV rounding mode.
 */
static inline void
muriscv_nn_set_rounding_mode(muriscv_nn_rounding_mode m)
{
    (void)m;
}

/**
 * Returns the number of clock cycles executed by the processor.
 * Overflows after 42.9 seconds on a 100 MIPS and with CPI = 1 (OVPsim).
 */
static inline uint32_t
rdcycle(void)
{
#if defined(__riscv) || defined(__riscv__)
    uint32_t cycles;
    __asm__ volatile("rdcycle %0" : "=r"(cycles));
    return cycles;
#else
    return 0;
#endif
}

/**
 * @brief Returns the full 64bit register cycle register, which holds the
 * number of clock cycles executed by the processor.
 */
static inline uint64_t
rdcycle64()
{
#if defined(__riscv) || defined(__riscv__)
#if __riscv_xlen == 32
    uint32_t cycles;
    uint32_t cyclesh1;
    uint32_t cyclesh2;

    /* Reads are not atomic. So ensure, that we are never reading
     * inconsistent values from the 64bit hardware register. */
    do
    {
        __asm__ volatile("rdcycleh %0" : "=r"(cyclesh1));
        __asm__ volatile("rdcycle %0" : "=r"(cycles));
        __asm__ volatile("rdcycleh %0" : "=r"(cyclesh2));
    } while (cyclesh1 != cyclesh2);

    return (((uint64_t)cyclesh1) << 32) | cycles;
#else
    uint64_t cycles;
    __asm__ volatile("rdcycle %0" : "=r"(cycles));
    return cycles;
#endif
#else
    return 0;
#endif
}

/* How many 'time cycles' (rdtime) per second (OVPsim and Spike). */
#define MURISCV_NN_RDTIME_PER_SECOND 1000000UL

/**
 * Returns wall-clock real time that has passed from an arbitrary
 * start time in the past.
 */
static inline uint32_t
rdtime(void)
{
#if defined(__riscv) || defined(__riscv__)
    uint32_t time;
    __asm__ volatile("rdtime %0" : "=r"(time));
    return time;
#else
    return 0;
#endif
}

/**
 * @brief Returns the number of instructions retired by the processor.
 */
static inline uint32_t
rdinstret(void)
{
#if defined(__riscv) || defined(__riscv__)
    uint32_t instret;
    __asm__ volatile("rdinstret %0" : "=r"(instret));
    return instret;
#else
    return 0;
#endif
}

#define MURSICV_NN_PROF_START uint32_t __prof = muriscv_nn_prof_start();
#define MURSICV_NN_PROF_STOP  muriscv_nn_prof_stop(__prof);

/**
 * Start profiling of a certain task.
 */
static inline uint32_t
muriscv_nn_prof_start(void)
{
    return rdcycle();
}

/**
 * Stop profiling of a certain task. Print and return resulting
 * cycles.
 */
static inline uint32_t
muriscv_nn_prof_stop(uint32_t cycles)
{
    cycles = rdcycle() - cycles;
    printf("##### muRISCV-NN profiler: %u cycles #####\n",
           (unsigned int)cycles);
    return cycles;
}

#endif /* _MURISCV_NN_UTIL_H */
