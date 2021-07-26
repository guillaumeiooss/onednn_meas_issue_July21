

=== Remarks for compiling/executing:

 * The file "example_utils.h" from the oneDNN git repository (location "examples/example_utils.h")
   is required and need to be added in this folder.

 * [PAPI] PAPI counters are used: don't forget to check the "perf_event_paranoid" to enable them
    ( sudo sh -c 'echo 1 >/proc/sys/kernel/perf_event_paranoid' )

 * [oneDNN sequential compilation] The oneDNN cmake command line is:
    cmake -DDNNL_CPU_RUNTIME=SEQ -DCMAKE_INSTALL_PREFIX=../oneDNNseq  ..  
  + add in our Makefile:
    - " -I/path/to/oneDNNseq/include " to the CFLAGS
    - " -L/path/to/oneDNNseq/lib " to the LDFLAGS
    - "/path/to/oneDNNseq/lib" to LD_LIBRARY_PATH with an export



File organisation:
  - timing.c/h : Timing utilities and Papi management (including Papi counter definition)
     => Both Papi and clock_gettime can be used.
  - mem_utils.c/.h : Memory allocation utilities
  - oneDNN_conv.c/.h : Use the oneDNN API to execute the kernel
  - main.c/.h : Benchmark management. In particular:
    - "main.h" control the number of repetitions, and needs to be parametrized in function of the architecture
            on which the benchmark is ran.
      (vec_size = 8 (AVX2) or 16 (AVX512))
      (num_fma_port is the number of fused multiply-add per vector unit on a single core)

    - The convolution sizes are defined at the beginning of the "main" function, in "main.c".
