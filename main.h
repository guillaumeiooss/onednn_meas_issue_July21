#ifndef MAIN_H
#define MAIN_H 

#include "oneDNN_conv.h"

// Number of repetition
#define NUM_REP (50)

// Allocation/free function used
#define ALLOC aligned_alloc
#define FREE free

// For peak performance computation
const int vec_size = 8; //16;			// AVX2
const int num_fma_port = 2;				// Dependent of the architecture


#endif