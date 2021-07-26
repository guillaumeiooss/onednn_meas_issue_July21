#ifndef MEM_UTILS
#define MEM_UTILS
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <sys/mman.h>

#define INIT_LENGTH (1 << 30)
typedef enum {NON_INIT, INIT, CLOSED} state_t;

void mem_init();
void mem_close();
void flush_cache(void);
void flush_intrin(void* ptr, size_t size);
#endif
