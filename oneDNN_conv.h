#ifndef ONEDNN_CONV_H
#define ONEDNN_CONV_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#include "mem_utils.h"
#include "example_utils.h"			// From OneDNN ("examples/example_utils.h" in the git repository)
#include "timing.h"

#define IND_TYPE uint64_t

// Layout of the convolution
typedef enum layout_in {YXC, XYC, CYX, XCY, CYXc} layout_in_t;
typedef enum layout_par {HWCF, WHCF, FCHW} layout_par_t;
typedef enum layout_out {YXF, XYF, FYX} layout_out_t;

#define DNN_LAYOUT_IN XYC
#define DNN_LAYOUT_PAR WHCF
#define DNN_LAYOUT_OUT XYF



// OneDNN implementation of the convolution computation
// Computation performed (b=1):
//		Output[ f, y, x] = K[f, c, h, w] * Input[ c, y+h, x+w]
void conv_onednn(
        papi_info_t info,
        long long* results,
        float * const __restrict__ Output,
        float * const __restrict__ Input,
        float * const __restrict__ K,
        const IND_TYPE W, const IND_TYPE H, const IND_TYPE C, const IND_TYPE F, const IND_TYPE X, const IND_TYPE Y,
        const IND_TYPE str_x, const IND_TYPE str_y);

// Layout to check correction.
//Only works for DNN_LAYOUT_IN = CYX | DNN_LAYOUT_PAR = FCHW | DNN_LAYOUT_OUT = FYX
void conv_naive_impl_FYX_CYX_FCHW(
        float * const __restrict__ Output,
        float const * const __restrict__ Input,
        float const * const __restrict__ K,
	const IND_TYPE W, const IND_TYPE H, const IND_TYPE C, const IND_TYPE F, const IND_TYPE X, const IND_TYPE Y,
	const IND_TYPE str_x, const IND_TYPE str_y);


#endif