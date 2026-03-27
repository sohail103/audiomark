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

#ifndef RISCV_AUDIOMARK_H
#define RISCV_AUDIOMARK_H

#include "ee_audiomark.h"
#include "ee_api.h"

ee_status_t riscv_cfft_init_f32(ee_cfft_f32_t *p_instance, int fft_length);

void riscv_cfft_f32(ee_cfft_f32_t *p_instance,
                    ee_f32_t      *p_buf,
                    uint8_t        ifftFlag,
                    uint8_t        bitReverseFlagR);

ee_status_t riscv_rfft_init_f32(ee_rfft_f32_t *p_instance, int fft_length);

void riscv_rfft_f32(ee_rfft_f32_t *p_instance,
                    ee_f32_t      *p_in,
                    ee_f32_t      *p_out,
                    uint8_t        ifftFlag);

void riscv_absmax_f32(const ee_f32_t *p_in,
                      uint32_t        len,
                      ee_f32_t       *p_max,
                      uint32_t       *p_index);

void riscv_cmplx_mult_cmplx_f32(const ee_f32_t *p_a,
                                const ee_f32_t *p_b,
                                ee_f32_t       *p_c,
                                uint32_t        len);

void riscv_cmplx_conj_f32(const ee_f32_t *p_a, ee_f32_t *p_c, uint32_t len);

void riscv_cmplx_dot_prod_f32(const ee_f32_t *p_a,
                              const ee_f32_t *p_b,
                              uint32_t        len,
                              ee_f32_t       *p_r,
                              ee_f32_t       *p_i);

void riscv_int16_to_f32(const int16_t *p_src, ee_f32_t *p_dst, uint32_t len);

void riscv_f32_to_int16(const ee_f32_t *p_src, int16_t *p_dst, uint32_t len);

void riscv_add_f32(ee_f32_t *p_a, ee_f32_t *p_b, ee_f32_t *p_c, uint32_t len);

void riscv_subtract_f32(ee_f32_t *p_a,
                        ee_f32_t *p_b,
                        ee_f32_t *p_c,
                        uint32_t  len);

void riscv_dot_prod_f32(ee_f32_t *p_a,
                        ee_f32_t *p_b,
                        uint32_t  len,
                        ee_f32_t *p_result);

void riscv_multiply_f32(ee_f32_t *p_a,
                        ee_f32_t *p_b,
                        ee_f32_t *p_c,
                        uint32_t  len);

void riscv_cmplx_mag_f32(ee_f32_t *p_a, ee_f32_t *p_c, uint32_t len);

void riscv_offset_f32(ee_f32_t *p_a,
                      ee_f32_t  offset,
                      ee_f32_t *p_c,
                      uint32_t  len);

void riscv_vlog_f32(ee_f32_t *p_a, ee_f32_t *p_c, uint32_t len);

void riscv_mat_vec_mult_f32(ee_matrix_f32_t *p_a, ee_f32_t *p_b, ee_f32_t *p_c);

void riscv_nn_init(void);

ee_status_t riscv_nn_classify(void);

#endif
