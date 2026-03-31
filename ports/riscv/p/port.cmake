include_directories(
    ${PORT_DIR}
    ${PORT_DIR}/src
)

file(GLOB PORT_SOURCES
    ${PORT_DIR}/../th_api.c
    ${PORT_DIR}/src/*.c


  # NOTE: these sources are temporary added from scalar port as rvp port for
  # them is not yet implemented.
  ${PORT_DIR}/../src/riscv_cmplx_mult_cmplx_f32.c
  ${PORT_DIR}/../src/riscv_cmplx_conj_f32.c
  ${PORT_DIR}/../src/riscv_cmplx_dot_prod_f32.c
  ${PORT_DIR}/../src/riscv_dot_prod_f32.c
  ${PORT_DIR}/../src/riscv_cmplx_mag_f32.c
  ${PORT_DIR}/../src/riscv_vlog_f32.c
  ${PORT_DIR}/../src/riscv_mat_vec_mult_f32.c
  ${PORT_DIR}/../src/riscv_nn_init.c
  ${PORT_DIR}/../src/riscv_nn_classify.c
  ${PORT_DIR}/../src/riscv_add_f32.c
  ${PORT_DIR}/../src/riscv_subtract_f32.c
  ${PORT_DIR}/../src/riscv_offset_f32.c
  ${PORT_DIR}/../src/riscv_multiply_f32.c
  ${PORT_DIR}/../src/riscv_f32_to_int16.c
  ${PORT_DIR}/../src/riscv_int16_to_f32.c
  ${PORT_DIR}/../src/riscv_absmax_f32.c
)

# TODO: Add our FFT functions as an FFT backend in fftwrap.c
# for now, we enable but do not use smallfft since fftwrap.c
# throws an error without it.
add_definitions(-DUSE_SMALLFT)

set(PORT_SOURCE 
    ${PORT_SOURCES}
)
