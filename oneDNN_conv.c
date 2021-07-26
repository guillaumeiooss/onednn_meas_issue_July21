#include "oneDNN_conv.h"


// OneDNN implementation of the convolution computation, for b=1
void conv_onednn(
        papi_info_t info,
        long long* results,
        float * const __restrict__ Output,
        float * const __restrict__ Input,
        float * const __restrict__ K,
        const IND_TYPE X, const IND_TYPE W, const IND_TYPE Y, const IND_TYPE H, const IND_TYPE C, const IND_TYPE F,
        const IND_TYPE str_x, const IND_TYPE str_y ) {

	const IND_TYPE B = 1;

	// Note: https://oneapi-src.github.io/oneDNN/convolution_example_cpp.html

	// === ENGINE & STEAM ===

	// Create the engine (<=> device on which we execute the code)
	dnnl_engine_t engine_cpu;
	CHECK(dnnl_engine_create(&engine_cpu, dnnl_cpu, 0));

	// Create the stream (<=> execution context)
	dnnl_stream_t stream_cpu;
	CHECK(dnnl_stream_create(&stream_cpu, engine_cpu, dnnl_stream_default_flags));

	// === Convolution memory descriptors ===
	// Inputs
	dnnl_memory_desc_t src_Input_md;
	dnnl_dim_t dims_src_Input[4] = { B, C, str_y * Y+H-1, str_x * X + W-1 };
	CHECK(dnnl_memory_desc_init_by_tag(
	        &src_Input_md,
	        4,
	        dims_src_Input,
	        dnnl_f32,
	        dnnl_format_kind_any		/* This is important */
		));

	// Weights
	dnnl_memory_desc_t src_K_md;
	dnnl_dim_t dims_src_K[4] = { F, C, H , W };
	CHECK(dnnl_memory_desc_init_by_tag(
	        &src_K_md,
	        4,
	        dims_src_K,
	        dnnl_f32,
	        dnnl_format_kind_any		/* This is important */
		));

	// Output
	dnnl_memory_desc_t dst_Output_md;
	dnnl_dim_t dims_dst_Output[4] = { B, F, Y, X };
	CHECK(dnnl_memory_desc_init_by_tag(
	        &dst_Output_md,
	        4,
	        dims_dst_Output,
	        dnnl_f32,
	        dnnl_format_kind_any		/* This is important */
		));


	// === User memories ===

	// Global memories & association with input arrays
	// Doc: https://oneapi-src.github.io/oneDNN/group__dnnl__api__memory.html#ga77c4ac2c6c59730ade594b954c145f73
	//	+ Creating the memory objects & filling them (for the inputs)
	dnnl_memory_desc_t src_user_Input_md;
	dnnl_dim_t dims_src_user_Input[4] = { B, C, str_y * Y+H-1, str_x * X + W-1 };
	//dnnl_dim_t strides_src_user_Input_[4] = { (str_x * X+W-1)*(str_y *Y+H-1)*C, (str_y *Y+H-1)*(str_x * X+W-1), (X+W-1), 1 };
    dnnl_dim_t strides_src_user_Input[4];
    layout_in_t lay_in =DNN_LAYOUT_IN; 
    switch(lay_in) {
        case CYX:
            strides_src_user_Input[0] = (str_x*X+W-1)*(str_y*Y+H-1) * C;
            strides_src_user_Input[1] = (str_y*Y+H-1)*(str_x*X+W-1);
            strides_src_user_Input[2] = (str_x*X+W-1);
            strides_src_user_Input[3] = 1;
            break;
        case XCY:
            strides_src_user_Input[0] = (str_x * X+W-1)*(str_y *Y+H-1) * C;
            strides_src_user_Input[1] = (str_y *Y+H-1);
            strides_src_user_Input[2] = 1;
            strides_src_user_Input[3] = (str_y *Y+H-1) * C;
            break;
        case XYC:
            strides_src_user_Input[0] = (str_x * X+W-1)*(str_y *Y+H-1) * C;
            strides_src_user_Input[1] = 1;
            strides_src_user_Input[2] = C;
            strides_src_user_Input[3] = (str_y *Y+H-1) * C;
            break;
        case YXC:
            strides_src_user_Input[0] = (str_x * X+W-1)*(str_y *Y+H-1) * C;
            strides_src_user_Input[1] = 1;
            strides_src_user_Input[2] = C;
            strides_src_user_Input[3] = (str_y *Y+H-1) * C;
            break;
        case CYXc:
            perror("Not supported yet");
            break;
    }
	CHECK(dnnl_memory_desc_init_by_strides(
			&src_user_Input_md,
			4,
			dims_src_user_Input,
			dnnl_f32,
			strides_src_user_Input
		));


	dnnl_memory_t src_user_Input_mem;
	CHECK(dnnl_memory_create(
			&src_user_Input_mem,
			&src_user_Input_md,
			engine_cpu,
			DNNL_MEMORY_ALLOCATE
		));

	write_to_dnnl_memory(Input, src_user_Input_mem);

	dnnl_memory_desc_t src_user_K_md;
	dnnl_dim_t dims_src_user_K[4] = { F, C, H , W };
	dnnl_dim_t strides_src_user_K[4] ;
    layout_par_t lay_par = DNN_LAYOUT_PAR; 
    switch(lay_par) {
        case FCHW:
            strides_src_user_K[0] = H * W * C;
            strides_src_user_K[1] = H * W;
            strides_src_user_K[2] = W;
            strides_src_user_K[3] = 1;
            break;
        case WHCF:
            strides_src_user_K[0] = 1;
            strides_src_user_K[1] = F;
            strides_src_user_K[2] = F * C;
            strides_src_user_K[3] = F * C * H;
            break;
        case HWCF:
            strides_src_user_K[0] = 1;
            strides_src_user_K[1] = F;
            strides_src_user_K[2] = F * C * W;
            strides_src_user_K[3] = F * C;
            break;
    }
	CHECK(dnnl_memory_desc_init_by_strides(
			&src_user_K_md,						// Memory descriptor (to init)
			4,									// Number of dims
			dims_src_user_K,					// Logical dimensions (sizes of the tensor)
			dnnl_f32,							// Data type
			strides_src_user_K					// Strides (<=> linearisation function)			// tag: OC IC H W ? dnnl_oihw 
		));

	dnnl_memory_t src_user_K_mem;
	CHECK(dnnl_memory_create(
			&src_user_K_mem,					// Memory (initialized)
			&src_user_K_md,						// Memory descriptor (defined above)
			engine_cpu,							// On which device
			DNNL_MEMORY_ALLOCATE
		));
	write_to_dnnl_memory(K, src_user_K_mem);




	

	dnnl_memory_desc_t dst_user_Output_md;
	dnnl_dim_t dims_dst_user_Output[4] = { B, F, Y, X };
	dnnl_dim_t strides_dst_user_Output[4] ;
    layout_out_t lay_out = DNN_LAYOUT_OUT;
    switch(lay_out) {
        case YXF:
            strides_dst_user_Output[0] = F * X * Y;
            strides_dst_user_Output[1] = 1;
            strides_dst_user_Output[2] = F * X;
            strides_dst_user_Output[3] = F;
            break;
        case XYF:
            strides_dst_user_Output[0] = F * X * Y;
            strides_dst_user_Output[1] = 1;
            strides_dst_user_Output[2] = F ;
            strides_dst_user_Output[3] = F * Y;
            break;
        case FYX:
            strides_dst_user_Output[0] = F * X * Y;
            strides_dst_user_Output[1] = Y * X;
            strides_dst_user_Output[2] = X;
            strides_dst_user_Output[3] = 1;
            break;
    }
	CHECK(dnnl_memory_desc_init_by_strides(
			&dst_user_Output_md,
			4,
			dims_dst_user_Output,
			dnnl_f32,
			strides_dst_user_Output
		));
	dnnl_memory_t dst_user_Output_mem;
	CHECK(dnnl_memory_create(
			&dst_user_Output_mem,
			&dst_user_Output_md,
			engine_cpu,
			DNNL_MEMORY_ALLOCATE
		));



	// === Convolution kernel ===

	// Source for formula: https://oneapi-src.github.io/oneDNN/dev_guide_convolution.html
	/* Extracted for the "dnnl.h"
	dnnl_status_t DNNL_API dnnl_convolution_forward_desc_init(
		dnnl_convolution_desc_t *conv_desc,
		dnnl_prop_kind_t prop_kind,
		dnnl_alg_kind_t alg_kind,
		const dnnl_memory_desc_t *src_desc,
		const dnnl_memory_desc_t *weights_desc,
		const dnnl_memory_desc_t *bias_desc,
		const dnnl_memory_desc_t *dst_desc,
		const dnnl_dims_t strides,
		const dnnl_dims_t padding_l,
		const dnnl_dims_t padding_r);
	*/
	dnnl_dim_t strides_conv[2] = {str_x, str_y};
	dnnl_dim_t padding_top_left_conv[2] = {0, 0};
	dnnl_dim_t padding_bot_right_conv[2] = {0, 0};


	dnnl_convolution_desc_t convolution_descriptor;
	CHECK(dnnl_convolution_forward_desc_init(
			&convolution_descriptor,		// Convolution descriptor
			dnnl_forward_training,			// Propagation kind  (forward)
			dnnl_convolution_direct,		// Kind of algorithm (direct,  alternative is "Winograd")

			&src_Input_md,					// Data
			&src_K_md,
			NULL,							// Disable the "bias" factor
			&dst_Output_md,

			strides_conv,					// Strides
			padding_top_left_conv,			// Padding top & left
			padding_bot_right_conv			// Padding bot & right
		));

	dnnl_primitive_desc_t convolution_pd;
	CHECK(dnnl_primitive_desc_create(
			&convolution_pd,
			&convolution_descriptor,
			NULL,							// Primitive attribute (can be NULL)
			engine_cpu,
			NULL							// Hint for backward propagation
		));

	dnnl_primitive_t convolution_p;
	CHECK(dnnl_primitive_create(
			&convolution_p,
			convolution_pd
		));
	

	// === Reorder - Link between user and conv memories===

	// Reorder the memories
	// https://oneapi-src.github.io/oneDNN/v2/group__dnnl__api__reorder.html
	//		=> This is needed to leave oneDNN the freedom of changing the input format,
	//		so that it can select the correct implementation

	// Input reorder
	dnnl_memory_t src_Input_mem;
    const dnnl_memory_desc_t *src_Input_conv_md = dnnl_primitive_desc_query_md(convolution_pd, dnnl_query_src_md, 0);
	CHECK(dnnl_memory_create(
			&src_Input_mem,
			src_Input_conv_md,
			engine_cpu,
			DNNL_MEMORY_ALLOCATE
		));
	dnnl_primitive_desc_t pd_reorder_Input;
	CHECK(dnnl_reorder_primitive_desc_create(
			&pd_reorder_Input,
			&src_user_Input_md,
			engine_cpu,
			src_Input_conv_md,//&src_Input_md,
			engine_cpu,
			NULL			// Not sure what this does ("Primitive attributes")
		));
	dnnl_primitive_t reorder_Input_p;
	CHECK(dnnl_primitive_create(
			&reorder_Input_p,
			pd_reorder_Input
		));
	dnnl_exec_arg_t reorder_Input_p_args[2] =
		{
			{DNNL_ARG_FROM, src_user_Input_mem},
            {DNNL_ARG_TO, src_Input_mem},
		};


	// Kernel reorder
	dnnl_memory_t src_K_mem;
    const dnnl_memory_desc_t *src_K_conv_md = dnnl_primitive_desc_query_md(convolution_pd, dnnl_query_weights_md, 0);
	CHECK(dnnl_memory_create(
			&src_K_mem,
			src_K_conv_md,
			engine_cpu,
			DNNL_MEMORY_ALLOCATE
		));
	dnnl_primitive_desc_t pd_reorder_K;
	CHECK(dnnl_reorder_primitive_desc_create(
			&pd_reorder_K,
			&src_user_K_md,
			engine_cpu,
			src_K_conv_md,//&src_K_md,
			engine_cpu,
			NULL			// Not sure what this does ("Primitive attributes")
		));
	dnnl_primitive_t reorder_K_p;
	CHECK(dnnl_primitive_create(
			&reorder_K_p,
			pd_reorder_K
		));
	dnnl_exec_arg_t reorder_K_p_args[2] =
		{
			{DNNL_ARG_FROM, src_user_K_mem},
            {DNNL_ARG_TO, src_K_mem},
		};


	// Output reorder
	dnnl_memory_t dst_Output_mem;
    const dnnl_memory_desc_t *dst_Output_conv_md = dnnl_primitive_desc_query_md(convolution_pd, dnnl_query_dst_md, 0);
	CHECK(dnnl_memory_create(
			&dst_Output_mem,
			dst_Output_conv_md,
			engine_cpu,
			DNNL_MEMORY_ALLOCATE
		));

	dnnl_primitive_desc_t pd_reorder_Output;
	CHECK(dnnl_reorder_primitive_desc_create(
			&pd_reorder_Output,
			dst_Output_conv_md,//&dst_Output_md,
			engine_cpu,
			&dst_user_Output_md,
			engine_cpu,
			NULL			// Not sure what this does ("Primitive attributes")
		));
	dnnl_primitive_t reorder_Output_p;
	CHECK(dnnl_primitive_create(
			&reorder_Output_p,
			pd_reorder_Output
		));
	dnnl_exec_arg_t reorder_Output_p_args[2] =
		{
			{DNNL_ARG_FROM, dst_Output_mem},
            {DNNL_ARG_TO, dst_user_Output_mem},
		};


	// Arguments to be used during execution
	dnnl_exec_arg_t conv_p_args[3] =
		{
            {DNNL_ARG_SRC, src_Input_mem},	// Source tag and memory obj
			{DNNL_ARG_WEIGHTS, src_K_mem}, 	// Weights tag and memory obj
            {DNNL_ARG_DST, dst_Output_mem},	// Destination tag and memory obj
		};

	



	// === Create primitive ===

    int num_jit_pre_execution = 10;
	// Few initial iterations, for the JIT (it is useful?)
	for (int i=0; i<num_jit_pre_execution; i++) {
		CHECK(dnnl_primitive_execute(
			reorder_Input_p,		// Primitive to execute
			stream_cpu,				//
			2,						// Number of elements in args
			reorder_Input_p_args	// Arguments (data I/O)
		));
		CHECK(dnnl_primitive_execute(
				reorder_K_p,			// Primitive to execute
				stream_cpu,				//
				2,						// Number of elements in args
				reorder_K_p_args		// Arguments (data I/O)
			));
		
		CHECK(dnnl_primitive_execute(
				convolution_p,			// Primitive to execute
				stream_cpu,				//
				3,						// Number of elements in args
				conv_p_args				// Arguments (data I/O)
			));

		CHECK(dnnl_primitive_execute(
				reorder_Output_p,			// Primitive to execute
				stream_cpu,				//
				2,						// Number of elements in args
				reorder_Output_p_args		// Arguments (data I/O)
			));
	}

	// === EXECUTION ===
	/* Extracted for the "dnnl.h"
	dnnl_status_t DNNL_API dnnl_primitive_execute(const_dnnl_primitive_t primitive,
        dnnl_stream_t stream, int nargs, const dnnl_exec_arg_t *args);
    */


    // Trying to flush cache before exec (if cold cache)
    flush_cache();
    //flush_intrin(Input, size_input * sizeof(float) );
    //flush_intrin(Output, size_output * sizeof(float));
    //flush_intrin(K, size_params * sizeof(float));

    
    //long start = get_ms();
    //fprintf(stderr, "LAUNCHING\n");
    // RECORD COUNTERS
    record_events(info);

	CHECK(dnnl_primitive_execute(
			reorder_Input_p,		// Primitive to execute
			stream_cpu,				//
			2,						// Number of elements in args
			reorder_Input_p_args	// Arguments (data I/O)
		));
	CHECK(dnnl_primitive_execute(
			reorder_K_p,			// Primitive to execute
			stream_cpu,				//
			2,						// Number of elements in args
			reorder_K_p_args		// Arguments (data I/O)
		));

    // record_events(info);
	CHECK(dnnl_primitive_execute(
			convolution_p,			// Primitive to execute
			stream_cpu,				//
			3,						// Number of elements in args
			conv_p_args				// Arguments (data I/O)
		));
    //retrieve_results(info, results);

	CHECK(dnnl_primitive_execute(
			reorder_Output_p,		// Primitive to execute
			stream_cpu,				//
			2,						// Number of elements in args
			reorder_Output_p_args	// Arguments (data I/O)
		));

    // WRITE COUNTER RESULTS
    retrieve_results(info, results);

    //time_t stop = time(NULL) - start;
    //long stop = get_ms() - start;
    //fprintf(stderr, "%ld,", stop );
	

	// === RETRIEVING OUTPUT ===
	read_from_dnnl_memory(Output, dst_user_Output_mem);


	// Cleaning up
	//dnnl_primitive_desc_destroy(convolution_pd);
	//dnnl_primitive_destroy(convolution_p);
	dnnl_memory_destroy(src_K_mem);
	dnnl_memory_destroy(src_Input_mem);
	dnnl_memory_destroy(dst_Output_mem);
	//dnnl_stream_destroy(stream_cpu);
	//dnnl_engine_destroy(engine_cpu);

	return;
}


// Layout to check correction.
//Only works for DNN_LAYOUT_IN = CYX | DNN_LAYOUT_PAR = FCHW | DNN_LAYOUT_OUT = FYX
void conv_naive_impl_FYX_CYX_FCHW(
	float * const __restrict__ Output,
	float const * const __restrict__ Input,
	float const * const __restrict__ K,
	const IND_TYPE W, const IND_TYPE H, const IND_TYPE C, const IND_TYPE F, const IND_TYPE X, const IND_TYPE Y,
	const IND_TYPE str_x, const IND_TYPE str_y) {

	//const IND_TYPE B = 1;

	for (IND_TYPE f=0; f<F; f++)
		for (IND_TYPE y=0; y<Y; y++)
			for (IND_TYPE x=0; x<X; x++) {
				Output[f*Y*X + y*Y + x] = 0.0;

				for (IND_TYPE c=0; c<C; c++)
					for (IND_TYPE h=0; h<H; h++)
						for (IND_TYPE w=0; w<W; w++) {
							Output[f*Y*X + y*Y + x] +=
								Input[c*(str_y*Y+H-1)*(str_x*X+W-1) + (str_y*y+h)*(str_x*X+W-1) + (x* str_x) +w]
									* K[f * C*H*W + c * H*W + h*W + w];
						}
			}
	return;
}
