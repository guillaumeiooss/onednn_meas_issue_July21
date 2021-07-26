
#include "main.h"


// Initialisation functions
void init_zero(float * tab, IND_TYPE size) {
	for (IND_TYPE i = 0; i < size; i++) {
		tab[i] = 0.;
	}
}

void init_rand_zero_one(float * tab, IND_TYPE size) {
	for (IND_TYPE i = 0; i < size; i++) {
                  tab[i] = (float)rand() / (float)RAND_MAX;
	}
}

// Utility function for sorting counter measurements
int compare_long(const void * i1, const void * i2) {
    long long l1, l2;
    l1 = *(long long *) i1;
    l2 = *(long long *) i2;
    return (l1 - l2);
}


/* Structures to contain the arguments of a convolution + the function that perform the computations */
typedef void (conv_kernel_t) (
        papi_info_t info,
        long long* results,
        float * __restrict__ output,
		float * __restrict__ input, float* __restrict__ params,
		IND_TYPE x_size, IND_TYPE w_size,
		IND_TYPE y_size, IND_TYPE h_size,
		IND_TYPE c_size, IND_TYPE f_size,
		IND_TYPE str_x, IND_TYPE str_y);

typedef struct conv_args {
    conv_kernel_t* kernel;
    float * output;
    float * input;
    float * params;
    IND_TYPE nx; IND_TYPE nw;
    IND_TYPE ny; IND_TYPE nh;
    IND_TYPE nc; IND_TYPE nf;
    IND_TYPE str_x; IND_TYPE str_y;
} conv_args_t;


conv_args_t build_conv_args(
		float * output, float * input, float * params,
		IND_TYPE nx, IND_TYPE nw,
		IND_TYPE ny, IND_TYPE nh,
		IND_TYPE nc, IND_TYPE nf,
		IND_TYPE str_x, IND_TYPE str_y,
		conv_kernel_t kernel) {

	conv_args_t cargs;
	
	cargs.output = output;
	cargs.input = input;
	cargs.params = params;
	
	cargs.nx = nx;
	cargs.nw = nw;
	cargs.ny = ny;
	cargs.nh = nh;
	cargs.nc = nc;
	cargs.nf = nf;
	cargs.str_x = str_x;
	cargs.str_y = str_y;
	
	cargs.kernel = kernel;
	
	return cargs;
}

void set_conv_kernel(conv_args_t *conv_args, conv_kernel_t kernel) {
	conv_args->kernel = kernel;
}


/* ======================= */

// Single execution of a convolution kernel
__attribute__((always_inline)) void exec_conv_rep(conv_args_t *conv_args, int num_rep, papi_info_t info, long long * results) {
    for (int r = 0; r < num_rep; r++) {
    	// init_kernel
    	init_zero(conv_args->output, conv_args->nx * conv_args->ny * conv_args->nf);
        
        conv_args->kernel(
                info, (long long *)&results[r * info.num_events],
                conv_args->output, conv_args->input, conv_args->params,
                conv_args->nx, conv_args->nw,
                conv_args->ny, conv_args->nh,
                conv_args->nc, conv_args->nf,
                conv_args->str_x, conv_args->str_y
				);
    }
}


// Multiple execution of a convolution kernel
void many_try(int64_t rep, conv_args_t * conv_args, long long* counter_results) {
    papi_info_t papi_info = build_papi_info();
    size_t num_events = papi_info.num_events;
    long long results [rep][num_events];
    exec_conv_rep(conv_args, rep, papi_info, (long long*)results);

    long long intermediate[rep];
    for (size_t i = 0; i < num_events; i++) {
        for (int r = 0; r < rep; r++) {
            intermediate[r] = results[r][i];
        }
        //fprintf(stderr, "UNSORTED INTERMEDIATE ");
        //for (int r = 0; r < rep  ; r++) {
        //    fprintf(stderr, "%lld,", intermediate[r]);
        //}
        //fprintf(stderr, "\n");
        qsort(intermediate, rep, sizeof(long long), compare_long);
        counter_results[i] = intermediate[rep / 2];
    }
}




// --- Main function (for testing) ---
int main(int argc, char **argv) {

	// Initialisations
	//omp_set_num_threads(32);
    init_papi();
    srand(time(NULL));

	// Problem sizes
	// TODO: do a real input with argv

    /* Yolo9000-13
	const int w_size = 1;
	const int h_size = 1;
	const int c_size = 512;
	const int f_size = 256;
	const int x_size = 34;
	const int y_size = 34;
	const int str_x = 1;	// Strides
	const int str_y = 1;
	//*/

	
	//* MobileNet-1
	const int w_size = 3;
	const int h_size = 3;
	const int c_size = 32;
	const int f_size = 32;
	const int x_size = 112;
	const int y_size = 112;
	const int str_x = 1;	// Strides
	const int str_y = 1;
	//*/


	const int b_size = 1;   // Fixed to 1
	

	// Array size calculation
	const int size_K = w_size * h_size * c_size * f_size;
	const int size_Input = b_size * ( (str_x*x_size) + w_size-1) * ( (str_y*y_size) + h_size-1) * c_size;
	const int size_Output = b_size * x_size * y_size * f_size;


	// Data preparation - allocate the input arrays
	float* m = (float*) ALLOC(64, (size_K + size_Input+ size_Output) * sizeof(float));
	if (m == NULL) {
        fprintf(stderr, "failed to allocate buffers\n");
        return -1;
    }
	float* Output = m;
	float* Input = Output + size_Output;
	float* K = Input + size_Input;	
	// Note: sizeof(float) = 32, but we impose vectorize on F => F is even
	//		(at the very least, through padding in the worst case) and everybody is aligned.

	// DEBUG
	//printf("Allocation done\n");

	// Initialisation
	init_zero(Output, size_Output);
    init_rand_zero_one(Input, size_Input);
    init_rand_zero_one(K, size_K);

    // DEBUG
	//printf("Initialisation done\n");

	conv_args_t conv_args = build_conv_args(Output, Input, K,
			x_size, w_size, y_size, h_size,
			c_size, f_size,
			str_x, str_y,
			NULL);
	set_conv_kernel(&conv_args, conv_onednn);


    // DEBUG
	//printf("Kernel set up\n");

	// Init measurement tools
    init_papi();
    long long conv_ctr_results[N_CTR];

    // Launch
    many_try(NUM_REP, &conv_args, conv_ctr_results);


    // Get the percentage of machine peak (sequential case)
    long long cyc = conv_ctr_results[CYCLES];
    float peak_perf = ( (float) (b_size * w_size * h_size * c_size * f_size * x_size * y_size))
    	/ (float) (vec_size * num_fma_port);

   	float peak_percent = peak_perf / ((float) cyc) * 100.0;

   	// Check
   	printf("Theoretical best performance - num cycle = %.2f \n", peak_perf);
   	printf("Cycle obtained = %lld \n", cyc);

    // Print result
    //print_counters(conv_ctr_results);
   	printf(" => Percentage of machine peak = %.4f \n", peak_percent);

    fflush(stdout);

    return 0;
}