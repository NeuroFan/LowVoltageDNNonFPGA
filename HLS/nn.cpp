#include <hls_math.h>
#include "nn.h"

nn_t l0_coef_multiplication[8][784] = {
	#include "data/l0_coef_multiplication.dat"
};

nn_t l1_coef_addition[8] = {
	#include "data/l1_coef_addition.dat"
};

nn_t l3_coef_multiplication[16][8] = {
	#include "data/l3_coef_multiplication.dat"
};

nn_t l4_coef_addition[16] = {
	#include "data/l4_coef_addition.dat"
};

nn_t l6_coef_multiplication[12][16] = {
	#include "data/l6_coef_multiplication.dat"
};

nn_t l7_coef_addition[12] = {
	#include "data/l7_coef_addition.dat"
};

nn_t l9_coef_multiplication[8][12] = {
	#include "data/l9_coef_multiplication.dat"
};

nn_t l10_coef_addition[8] = {
	#include "data/l10_coef_addition.dat"
};

nn_t l12_coef_multiplication[40][8] = {
	#include "data/l12_coef_multiplication.dat"
};

nn_t l13_coef_addition[40] = {
	#include "data/l13_coef_addition.dat"
};

void l0_resource_multiplication(nn_t outputs[LAYER_0_OUTPUTS_OUTER][LAYER_0_OUTPUTS_INNER], nn_t inputs[LAYER_0_INPUTS])
{
#pragma HLS PIPELINE II=128
#pragma HLS ALLOCATION instances=Mul limit=49 operation
	for(int i=0; i<LAYER_0_OUTPUTS_OUTER; i++){
	#pragma HLS UNROLL
		for(int j=0; j<LAYER_0_OUTPUTS_INNER; j++){
			outputs[i][j] = inputs[j] * l0_coef_multiplication[i][j];
		}
	}
}

void l1_resource_addition(nn_t outputs[LAYER_1_OUTPUTS], nn_t inputs[LAYER_1_INPUTS_OUTER][LAYER_1_INPUTS_INNER])
{
#pragma HLS PIPELINE II=128
#pragma HLS ALLOCATION instances=add limit=47 operation
	for(int i=0; i<LAYER_1_INPUTS_OUTER; i++){
	#pragma HLS UNROLL
		outputs[i] = l1_coef_addition[i];
		for(int j=0; j<LAYER_1_INPUTS_INNER; j++){
			outputs[i] += inputs[i][j];
		}
	}
}

void l2_resource_activation_tansig(nn_t outputs[LAYER_2_OUTPUTS], nn_t inputs[LAYER_2_INPUTS])
{
#pragma HLS PIPELINE II=128
	for(int i=0; i<LAYER_2_OUTPUTS; i++){
	#pragma HLS UNROLL
		outputs[i] = tanh(inputs[i]);
	}
}

void l3_resource_multiplication(nn_t outputs[LAYER_3_OUTPUTS_OUTER][LAYER_3_OUTPUTS_INNER], nn_t inputs[LAYER_3_INPUTS])
{
#pragma HLS PIPELINE II=128
#pragma HLS ALLOCATION instances=Mul limit=1 operation
	for(int i=0; i<LAYER_3_OUTPUTS_OUTER; i++){
	#pragma HLS UNROLL
		for(int j=0; j<LAYER_3_OUTPUTS_INNER; j++){
			outputs[i][j] = inputs[j] * l3_coef_multiplication[i][j];
		}
	}
}

void l4_resource_addition(nn_t outputs[LAYER_4_OUTPUTS], nn_t inputs[LAYER_4_INPUTS_OUTER][LAYER_4_INPUTS_INNER])
{
#pragma HLS PIPELINE II=128
#pragma HLS ALLOCATION instances=add limit=1 operation
	for(int i=0; i<LAYER_4_INPUTS_OUTER; i++){
	#pragma HLS UNROLL
		outputs[i] = l4_coef_addition[i];
		for(int j=0; j<LAYER_4_INPUTS_INNER; j++){
			outputs[i] += inputs[i][j];
		}
	}
}

void l5_resource_activation_tansig(nn_t outputs[LAYER_5_OUTPUTS], nn_t inputs[LAYER_5_INPUTS])
{
#pragma HLS PIPELINE II=128
	for(int i=0; i<LAYER_5_OUTPUTS; i++){
	#pragma HLS UNROLL
		outputs[i] = tanh(inputs[i]);
	}
}

void l6_resource_multiplication(nn_t outputs[LAYER_6_OUTPUTS_OUTER][LAYER_6_OUTPUTS_INNER], nn_t inputs[LAYER_6_INPUTS])
{
#pragma HLS PIPELINE II=128
#pragma HLS ALLOCATION instances=Mul limit=2 operation
	for(int i=0; i<LAYER_6_OUTPUTS_OUTER; i++){
	#pragma HLS UNROLL
		for(int j=0; j<LAYER_6_OUTPUTS_INNER; j++){
			outputs[i][j] = inputs[j] * l6_coef_multiplication[i][j];
		}
	}
}

void l7_resource_addition(nn_t outputs[LAYER_7_OUTPUTS], nn_t inputs[LAYER_7_INPUTS_OUTER][LAYER_7_INPUTS_INNER])
{
#pragma HLS PIPELINE II=128
#pragma HLS ALLOCATION instances=add limit=2 operation
	for(int i=0; i<LAYER_7_INPUTS_OUTER; i++){
	#pragma HLS UNROLL
		outputs[i] = l7_coef_addition[i];
		for(int j=0; j<LAYER_7_INPUTS_INNER; j++){
			outputs[i] += inputs[i][j];
		}
	}
}

void l8_resource_activation_tansig(nn_t outputs[LAYER_8_OUTPUTS], nn_t inputs[LAYER_8_INPUTS])
{
#pragma HLS PIPELINE II=128
	for(int i=0; i<LAYER_8_OUTPUTS; i++){
	#pragma HLS UNROLL
		outputs[i] = tanh(inputs[i]);
	}
}

void l9_resource_multiplication(nn_t outputs[LAYER_9_OUTPUTS_OUTER][LAYER_9_OUTPUTS_INNER], nn_t inputs[LAYER_9_INPUTS])
{
#pragma HLS PIPELINE II=128
#pragma HLS ALLOCATION instances=Mul limit=1 operation
	for(int i=0; i<LAYER_9_OUTPUTS_OUTER; i++){
	#pragma HLS UNROLL
		for(int j=0; j<LAYER_9_OUTPUTS_INNER; j++){
			outputs[i][j] = inputs[j] * l9_coef_multiplication[i][j];
		}
	}
}

void l10_resource_addition(nn_t outputs[LAYER_10_OUTPUTS], nn_t inputs[LAYER_10_INPUTS_OUTER][LAYER_10_INPUTS_INNER])
{
#pragma HLS PIPELINE II=128
#pragma HLS ALLOCATION instances=add limit=1 operation
	for(int i=0; i<LAYER_10_INPUTS_OUTER; i++){
	#pragma HLS UNROLL
		outputs[i] = l10_coef_addition[i];
		for(int j=0; j<LAYER_10_INPUTS_INNER; j++){
			outputs[i] += inputs[i][j];
		}
	}
}

void l11_resource_activation_tansig(nn_t outputs[LAYER_11_OUTPUTS], nn_t inputs[LAYER_11_INPUTS])
{
#pragma HLS PIPELINE II=128
	for(int i=0; i<LAYER_11_OUTPUTS; i++){
	#pragma HLS UNROLL
		outputs[i] = tanh(inputs[i]);
	}
}

void l12_resource_multiplication(nn_t outputs[LAYER_12_OUTPUTS_OUTER][LAYER_12_OUTPUTS_INNER], nn_t inputs[LAYER_12_INPUTS])
{
#pragma HLS PIPELINE II=128
#pragma HLS ALLOCATION instances=Mul limit=3 operation
	for(int i=0; i<LAYER_12_OUTPUTS_OUTER; i++){
	#pragma HLS UNROLL
		for(int j=0; j<LAYER_12_OUTPUTS_INNER; j++){
			outputs[i][j] = inputs[j] * l12_coef_multiplication[i][j];
		}
	}
}

void l13_resource_addition(nn_t outputs[LAYER_13_OUTPUTS], nn_t inputs[LAYER_13_INPUTS_OUTER][LAYER_13_INPUTS_INNER])
{
#pragma HLS PIPELINE II=128
#pragma HLS ALLOCATION instances=add limit=3 operation
	for(int i=0; i<LAYER_13_INPUTS_OUTER; i++){
	#pragma HLS UNROLL
		outputs[i] = l13_coef_addition[i];
		for(int j=0; j<LAYER_13_INPUTS_INNER; j++){
			outputs[i] += inputs[i][j];
		}
	}
}

void l14_resource_activation_tansig(nn_t outputs[LAYER_14_OUTPUTS], nn_t inputs[LAYER_14_INPUTS])
{
#pragma HLS PIPELINE II=128
	for(int i=0; i<LAYER_14_OUTPUTS; i++){
	#pragma HLS UNROLL
		outputs[i] = tanh(inputs[i]);
	}
}




void nn_top(nn_t outputs[NN_OUTPUT_COUNT], nn_t inputs[NN_INPUT_COUNT])
{
#pragma HLS INTERFACE s_axilite port=outputs
#pragma HLS INTERFACE s_axilite port=inputs
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS PIPELINE II=128
#pragma HLS ARRAY_PARTITION variable=inputs complete dim=1
#pragma HLS ARRAY_PARTITION variable=outputs complete dim=1
	nn_t output_0[LAYER_0_OUTPUTS_OUTER][LAYER_0_OUTPUTS_INNER];
#pragma HLS ARRAY_PARTITION variable=output_0 complete dim=0
	nn_t output_1[LAYER_1_OUTPUTS];
#pragma HLS ARRAY_PARTITION variable=output_1 complete dim=0
	nn_t output_2[LAYER_2_OUTPUTS];
#pragma HLS ARRAY_PARTITION variable=output_2 complete dim=0
	nn_t output_3[LAYER_3_OUTPUTS_OUTER][LAYER_3_OUTPUTS_INNER];
#pragma HLS ARRAY_PARTITION variable=output_3 complete dim=0
	nn_t output_4[LAYER_4_OUTPUTS];
#pragma HLS ARRAY_PARTITION variable=output_4 complete dim=0
	nn_t output_5[LAYER_5_OUTPUTS];
#pragma HLS ARRAY_PARTITION variable=output_5 complete dim=0
	nn_t output_6[LAYER_6_OUTPUTS_OUTER][LAYER_6_OUTPUTS_INNER];
#pragma HLS ARRAY_PARTITION variable=output_6 complete dim=0
	nn_t output_7[LAYER_7_OUTPUTS];
#pragma HLS ARRAY_PARTITION variable=output_7 complete dim=0
	nn_t output_8[LAYER_8_OUTPUTS];
#pragma HLS ARRAY_PARTITION variable=output_8 complete dim=0
	nn_t output_9[LAYER_9_OUTPUTS_OUTER][LAYER_9_OUTPUTS_INNER];
#pragma HLS ARRAY_PARTITION variable=output_9 complete dim=0
	nn_t output_10[LAYER_10_OUTPUTS];
#pragma HLS ARRAY_PARTITION variable=output_10 complete dim=0
	nn_t output_11[LAYER_11_OUTPUTS];
#pragma HLS ARRAY_PARTITION variable=output_11 complete dim=0
	nn_t output_12[LAYER_12_OUTPUTS_OUTER][LAYER_12_OUTPUTS_INNER];
#pragma HLS ARRAY_PARTITION variable=output_12 complete dim=0
	nn_t output_13[LAYER_13_OUTPUTS];
#pragma HLS ARRAY_PARTITION variable=output_13 complete dim=0

	l0_resource_multiplication(output_0, inputs);

	l1_resource_addition(output_1, output_0);

	l2_resource_activation_tansig(output_2, output_1);

	l3_resource_multiplication(output_3, output_2);

	l4_resource_addition(output_4, output_3);

	l5_resource_activation_tansig(output_5, output_4);

	l6_resource_multiplication(output_6, output_5);

	l7_resource_addition(output_7, output_6);

	l8_resource_activation_tansig(output_8, output_7);

	l9_resource_multiplication(output_9, output_8);

	l10_resource_addition(output_10, output_9);

	l11_resource_activation_tansig(output_11, output_10);

	l12_resource_multiplication(output_12, output_11);

	l13_resource_addition(output_13, output_12);

	l14_resource_activation_tansig(outputs, output_13);
}
