/**
 * Copyright (C) 2022 EEMBC
 *
 * All EEMBC Benchmark Software are products of EEMBC and are provided under the
 * terms of the EEMBC Benchmark License Agreements. The EEMBC Benchmark Software
 * are proprietary intellectual properties of EEMBC and its Members and is
 * protected under all applicable laws, including all applicable copyright laws.
 *
 * If you received this EEMBC Benchmark Software without having a currently
 * effective EEMBC Benchmark License Agreement, you must discontinue use.
 */

#ifndef __TH_TYPES_H
#define __TH_TYPES_H

#include "src/rvp_support_guard.h"

#include <stddef.h>
#include <string.h>

#define TH_FLOAT32_TYPE float

typedef int32_t   q31_t;
typedef int32x2_t q31x2_t;

/*
 * Instance structure for the fixed-point CFFT/CIFFT function.
 */
typedef struct
{
    uint16_t        fftLen;       /**< length of the FFT. */
    const q31_t    *pTwiddle;     /**< points to the Twiddle factor table. */
    const uint16_t *pBitRevTable; /**< points to the bit reversal table. */
    uint16_t        bitRevLength; /**< bit reversal table length. */
    const uint32_t
        *rearranged_twiddle_tab_stride1_arr; /**< Per stage reordered twiddle
                                                pointer (offset 1) */
    const uint32_t
        *rearranged_twiddle_tab_stride2_arr; /**< Per stage reordered twiddle
                                                pointer (offset 2) */
    const uint32_t
        *rearranged_twiddle_tab_stride3_arr; /**< Per stage reordered twiddle
                                                pointer (offset 3) */
    const q31_t
        *rearranged_twiddle_stride1; /**< reordered twiddle offset 1 storage */
    const q31_t
        *rearranged_twiddle_stride2; /**< reordered twiddle offset 2 storage */
    const q31_t *rearranged_twiddle_stride3;
} riscv_cfft_instance_q31;

/*
 * Instance structure for the floating-point RFFT/RIFFT function.
 */
typedef struct
{
    riscv_cfft_instance_q31 Sint;         /**< Internal CFFT structure. */
    uint16_t                fftLenRFFT;   /**< length of the real sequence */
    const q31_t            *pTwiddleRFFT; /**< Twiddle factors real stage  */
} riscv_rfft_fast_instance_q31;

/*
 *  struct for matrix type is not defined yet because audiomark
 *  expects a particular nomenclature for the MFCC functions and
 *  ee_matrix_f32_t is already defined according to that in
 *  src/ee_types.h. the ARM port defines custom structs for this
 *  and manages the conversions manually.
 */

#define TH_RFFT_INSTANCE_FLOAT32_TYPE riscv_rfft_fast_instance_q31

#define TH_CFFT_INSTANCE_FLOAT32_TYPE riscv_cfft_instance_q31

#endif /* __TH_TYPES_H */
