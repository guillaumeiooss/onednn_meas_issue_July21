#include <stdlib.h>
#include <assert.h>
#include <sys/mman.h>
#include "mem_utils.h"
#include <x86intrin.h>


void* ptr;
size_t nb_allocated_bytes ;
state_t state = NON_INIT;

#define FLAGS (MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB)
#define PROTECTION (PROT_READ | PROT_WRITE)

void mem_init() {
    assert (state == NON_INIT);
    void* alloc_ptr = mmap(0, INIT_LENGTH, PROTECTION, FLAGS, 0, 0);
    assert(alloc_ptr != MAP_FAILED);
    ptr = alloc_ptr;
    nb_allocated_bytes = 0;
    state = INIT;
}
void mem_close() {
    assert(state == INIT);
    munmap(ptr, 0);
    state = CLOSED;
}

#define BIG_SIZE 5000000
#define NUM_ITER 2
void flush_cache() {
    float tmp[8] = {0.};
    float res = 0;
    for (int i = 0; i< NUM_ITER; i++){
        float *dirty = (float *)malloc(BIG_SIZE * sizeof(float));
#pragma omp parallel for
        for (int dirt = 0; dirt < BIG_SIZE; dirt++){
            dirty[dirt] = dirt%100;
            tmp[dirt%8] += dirty[dirt];
        }
        for(int ii =0; ii<8;ii++){res+= tmp[ii];}
        free(dirty);

    }
	FILE* fd = fopen("/dev/null", "w");
	fprintf(fd, "%f\n", res);
	fclose(fd);
}

void flush_intrin(void* ptr, size_t size) {
	for (size_t i=0; i < size; i++) {
		_mm_clflush(ptr + i);
	}
}
