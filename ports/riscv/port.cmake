include_directories(
    ${PORT_DIR}
)

# TODO: Add our FFT functions as an FFT backend in fftwrap.c
# for now, we enable but do not use smallfft since fftwrap.c
# throws an error without it.
add_definitions(-DUSE_SMALLFT)

set(PORT_SOURCE 

    ${PORT_DIR}/th_api.c

    ${PORT_DIR}/src/nn/avgpool_s8.c
    ${PORT_DIR}/src/nn/convolve_s8.c
    ${PORT_DIR}/src/nn/depthwise_conv_s8.c
    ${PORT_DIR}/src/nn/fully_connected_s8.c
    ${PORT_DIR}/src/nn/mat_mult_kernel_s8_s16.c
    ${PORT_DIR}/src/nn/q7_to_q15_with_offset.c
    ${PORT_DIR}/src/nn/softmax_s8.c
    ${PORT_DIR}/src/nn/vec_mat_mult_t_s8.c

    ${PORT_DIR}/src/riscv_absmax_f32.c
    ${PORT_DIR}/src/riscv_add_f32.c
    ${PORT_DIR}/src/riscv_cfft_f32.c
    ${PORT_DIR}/src/riscv_cfft_init_f32.c
    ${PORT_DIR}/src/riscv_cmplx_conj_f32.c
    ${PORT_DIR}/src/riscv_cmplx_dot_prod_f32.c
    ${PORT_DIR}/src/riscv_cmplx_mag_f32.c
    ${PORT_DIR}/src/riscv_cmplx_mult_cmplx_f32.c
    ${PORT_DIR}/src/riscv_dot_prod_f32.c
    ${PORT_DIR}/src/riscv_f32_to_int16.c
    ${PORT_DIR}/src/riscv_int16_to_f32.c
    ${PORT_DIR}/src/riscv_mat_vec_mult_f32.c
    ${PORT_DIR}/src/riscv_multiply_f32.c
    ${PORT_DIR}/src/riscv_nn_classify.c
    ${PORT_DIR}/src/riscv_nn_init.c
    ${PORT_DIR}/src/riscv_offset_f32.c
    ${PORT_DIR}/src/riscv_rfft_f32.c
    ${PORT_DIR}/src/riscv_rfft_init_f32.c
    ${PORT_DIR}/src/riscv_subtract_f32.c
    ${PORT_DIR}/src/riscv_vlog_f32.c
)
