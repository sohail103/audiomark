/* Copyright (C) 2006 Jean-Marc Valin */
/**
   @file filterbank.c
   @brief Converting between psd and filterbank
 */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:

   1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   3. The name of the author may not be used to endorse or promote products
   derived from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
   IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
   OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
   DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
   INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
   (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
   HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
   STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
   ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
   POSSIBILITY OF SUCH DAMAGE.
*/


#include <riscv_vector.h>

#if defined(FLOATING_POINT)

#ifdef OVERRIDE_FB_COMPUTE_BANK32
void filterbank_compute_bank32(FilterBank *bank, spx_word32_t *ps, spx_word32_t *mel)
{
    /* Known constant */
    const int NB_BANKS = 24;

    /* Buffer size: 24 banks * up to 32 vector lanes */
    /* VLEN = 512 gives 32 lanes at e32m2 */
    const int MAX_LANES = 32;
    float histoTmp[NB_BANKS * MAX_LANES];

    uint32_t  *pL = (uint32_t *) bank->bank_left;
    uint32_t  *pR = (uint32_t *) bank->bank_right;
    float *pfiltL = (float *) bank->filter_left;
    float *pfiltR = (float *) bank->filter_right;
    float *pPs = (float *) ps;

    size_t len = bank->len;
    size_t hw_vlmax = __riscv_vsetvlmax_e32m2();
    /* if hw_vlmax is greater than max supported lanes, clamp it */
    size_t active_lanes = (hw_vlmax > MAX_LANES) ? MAX_LANES : hw_vlmax;

    int clear_len = NB_BANKS * active_lanes;
    float *pClear = histoTmp;

    vfloat32m2_t vZero = __riscv_vfmv_v_f_f32m2(0.0f, hw_vlmax);

    while (clear_len > 0) {
        size_t vl = __riscv_vsetvl_e32m2(clear_len);
        __riscv_vse32_v_f32m2(pClear, vZero, vl);
        pClear += vl;
        clear_len -= vl;
    }

    size_t vl_lanes = __riscv_vsetvl_e32m2(active_lanes);
    vuint32m2_t vVid = __riscv_vid_v_u32m2(vl_lanes);
    vuint32m2_t vLaneOffsets = __riscv_vmul_vx_u32m2(vVid, NB_BANKS * sizeof(float), vl_lanes);

    while (len > 0) {
        /* Manually limit the requested length to our active_lanes cap */
        size_t req_len = (len < active_lanes) ? len : active_lanes;
        size_t vl = __riscv_vsetvl_e32m2(req_len);

        vfloat32m2_t vPsVec = __riscv_vle32_v_f32m2(pPs, vl);

        vuint32m2_t  vIdL = __riscv_vle32_v_u32m2(pL, vl);
        vuint32m2_t vOffL = __riscv_vsll_vx_u32m2(vIdL, 2, vl);
        vOffL = __riscv_vadd_vv_u32m2(vOffL, vLaneOffsets, vl);

        vfloat32m2_t vInL   = __riscv_vluxei32_v_f32m2(histoTmp, vOffL, vl);
        vfloat32m2_t vFiltL = __riscv_vle32_v_f32m2(pfiltL, vl);
        vInL = __riscv_vfmacc_vv_f32m2(vInL, vFiltL, vPsVec, vl);
        __riscv_vsuxei32_v_f32m2(histoTmp, vOffL, vInL, vl);

        vuint32m2_t vIdR  = __riscv_vle32_v_u32m2(pR, vl);
        vuint32m2_t vOffR = __riscv_vsll_vx_u32m2(vIdR, 2, vl);
        vOffR = __riscv_vadd_vv_u32m2(vOffR, vLaneOffsets, vl);

        vfloat32m2_t vInR = __riscv_vluxei32_v_f32m2(histoTmp, vOffR, vl);
        vfloat32m2_t vFiltR = __riscv_vle32_v_f32m2(pfiltR, vl);
        vInR = __riscv_vfmacc_vv_f32m2(vInR, vFiltR, vPsVec, vl);
        __riscv_vsuxei32_v_f32m2(histoTmp, vOffR, vInR, vl);

        /* Advance pointers */
        pL += vl;
        pR += vl;
        pfiltL += vl;
        pfiltR += vl;
        pPs += vl;
        len -= vl;
    }

    int banks_left = NB_BANKS;
    float *pMelOut = mel;
    float *pHistoBase = histoTmp;

    while (banks_left > 0) {
        size_t vl = __riscv_vsetvl_e32m2(banks_left);
        vfloat32m2_t vMel = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        float *pLane = pHistoBase;

        for (int lane = 0; lane < active_lanes; lane++) {
            vfloat32m2_t vLaneData = __riscv_vle32_v_f32m2(pLane, vl);
            vMel = __riscv_vfadd_vv_f32m2(vMel, vLaneData, vl);

            /* Jump exactly by one bank block */
            pLane += NB_BANKS; 
        }

        __riscv_vse32_v_f32m2(pMelOut, vMel, vl);

        pMelOut += vl;
        pHistoBase += vl;
        banks_left -= vl;
    }
}
#endif

#ifdef OVERRIDE_FB_COMPUTE_PSD16
void filterbank_compute_psd16(FilterBank * bank, spx_word16_t * mel, spx_word16_t * ps)
{
    uint32_t *pL  = (uint32_t *) bank->bank_left;
    uint32_t *pR  = (uint32_t *) bank->bank_right;
    float *pfiltL = (float *) bank->filter_left;
    float *pfiltR = (float *) bank->filter_right;

    float *pMel = (float *) mel;
    float *pPs  = (float *) ps;

    size_t len = bank->len;

    while (len > 0) {
        size_t vl = __riscv_vsetvl_e32m2(len); 

        vuint32m2_t vIdL     = __riscv_vle32_v_u32m2(pL, vl);
        vuint32m2_t vIdR     = __riscv_vle32_v_u32m2(pR, vl);

        vuint32m2_t vOffsetL = __riscv_vsll_vx_u32m2(vIdL, 2, vl);
        vuint32m2_t vOffsetR = __riscv_vsll_vx_u32m2(vIdR, 2, vl);

        vfloat32m2_t vMelL   = __riscv_vluxei32_v_f32m2(pMel, vOffsetL, vl);
        vfloat32m2_t vFiltL  = __riscv_vle32_v_f32m2(pfiltL, vl);

        vfloat32m2_t vTmp    = __riscv_vfmul_vv_f32m2(vMelL, vFiltL, vl);

        vfloat32m2_t vMelR   = __riscv_vluxei32_v_f32m2(pMel, vOffsetR, vl);
        vfloat32m2_t vFiltR  = __riscv_vle32_v_f32m2(pfiltR, vl);

        vTmp = __riscv_vfmacc_vv_f32m2(vTmp, vMelR, vFiltR, vl);

        __riscv_vse32_v_f32m2(pPs, vTmp, vl);

        pL     += vl;
        pR     += vl;
        pfiltL += vl;
        pfiltR += vl;
        pPs    += vl;
        len    -= vl;
    }
}
#endif

#else

/* FIXED_POINT not needed for EEMBC AudioMark */
#error "Fixed Point FilterBank Vector Optimization is not available"

#endif

