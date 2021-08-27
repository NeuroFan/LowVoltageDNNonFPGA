import json
import os
import math
import numpy as np
from pprint import pprint
# from c_nn_layer import *
# from c_nn_normalization import *

FNAME_CPP_HEADER       = 'nn.h'
FNAME_CPP_SOURCE       = 'nn.cpp'
FNAME_CPP_TESTBENCH    = 'main.cpp'
DIR_DATA               = 'data'
DEFAULT_PATH_OUTPUT    = 'output'
DEFAULT_PATH_DATA      = DEFAULT_PATH_OUTPUT + "/" + DIR_DATA
DEFAULT_PATH_TESTBENCH = DEFAULT_PATH_OUTPUT + '/testbench'
DEFAULT_WIDTH_NETWORK          = 32
DEFAULT_WIDTH_NETWORK_WHOLE    = 16
DEFAULT_WIDTH_LUT_INPUT        = 12
DEFAULT_WIDTH_LUT_INPUT_WHOLE  = 3
DEFAULT_WIDTH_LUT_OUTPUT       = 32
DEFAULT_WIDTH_LUT_OUTPUT_WHOLE = 8
DEFAULT_RESOURCE_COUNT = 1

DEFAULT_INTERFACE = "s_axilite"
DEFAULT_DTYPE     = "float"

RESOURCE_EXECUTION_MULTIPLICATION        = 1.0
RESOURCE_EXECUTION_ADDITION              = 1.0
RESOURCE_EXECUTION_ACTIVATION_TANSIG     = 58.0
RESOURCE_EXECUTION_ACTIVATION_TANSIG_LUT = 1.0
RESOURCE_EXECUTION_ACTIVATION_LINEAR     = 1.0


class nn:
	def __init__(self, fname):
		self.json = json.load(open(fname))
		self.json_test  = None
		self.test_count = 0
		self.layers = {}
		self.current_layer = 0
		self.width_network          = DEFAULT_WIDTH_NETWORK
		self.width_network_whole    = DEFAULT_WIDTH_NETWORK_WHOLE
		self.width_lut_input        = DEFAULT_WIDTH_LUT_INPUT
		self.width_lut_input_whole  = DEFAULT_WIDTH_LUT_INPUT_WHOLE
		self.width_lut_output       = DEFAULT_WIDTH_LUT_OUTPUT
		self.width_lut_output_whole = DEFAULT_WIDTH_LUT_OUTPUT_WHOLE
		self.path_output            = DEFAULT_PATH_OUTPUT
		self.path_data              = DEFAULT_PATH_DATA
		self.path_testbench         = DEFAULT_PATH_TESTBENCH
		self.interface              = DEFAULT_INTERFACE
		self.dtype                  = DEFAULT_DTYPE
		self.dtype_LUT_in           = 'ap_fixed<' + str(DEFAULT_WIDTH_LUT_INPUT) + ',' + str(DEFAULT_WIDTH_LUT_INPUT_WHOLE) + '>'
		self.dtype_LUT_out          = 'ap_fixed<' + str(DEFAULT_WIDTH_LUT_OUTPUT) + ',' + str(DEFAULT_WIDTH_LUT_OUTPUT_WHOLE) + '>'

	def set_interface(self, interface):
		if (interface != "s_axilite") and (interface != "s_axis"):
			raise ValueError('Interface type is not supported')
		self.interface = interface

	def set_dtype(self, dtype):
		if dtype == "float":
			self.dtype = dtype
		elif dtype == "double":
			self.dtype = dtype
		elif dtype == "fixed":
			self.dtype = 'ap_fixed<' + str(self.width_network) + ',' + str(self.width_network_whole) + '>'
		else:
			raise ValueError('Data type is not supported')

	def set_dtype_fixed(self,width,width_whole):
		self.width_network       = width
		self.width_network_whole = width_whole 
		self.dtype               = 'ap_fixed<' + str(width) + ',' + str(width_whole) + '>'

	def set_dtype_fixed_LUT_input(self,width,width_whole):
		self.width_lut_input       = width
		self.width_lut_input_whole = width_whole
		self.dtype_LUT_in          = 'ap_fixed<' + str(width) + ',' + str(width_whole) + '>'

	def set_dtype_fixed_LUT_output(self,width,width_whole):
		self.width_lut_output       = width
		self.width_lut_output_whole = width_whole
		self.dtype_LUT_out          = 'ap_fixed<' + str(width) + ',' + str(width_whole) + '>'

	def set_path_output(self, path):
		self.path_output = path
		self.path_data   = path + "/" + DIR_DATA

	def set_path_testbench(self, path):
		self.path_testbench = path

	def set_test_file(self, fname):
		if fname != None:
			self.json_test  = json.load(open(fname))
			self.test_count =  len(self.json_test["inputs"])

	def parse_configuration(self):
		# get layers
		self.__parse_configuration_layers()

		# get execution time
		self.__parse_configuration_set_execution_time()

	def show_configuration(self):
		print("DATA TYPE (Network): " + self.dtype)
		print("| NUMBER |                     LAYER TYPE | INPUTS | OUTPUTS | RESOURCES | EXECUTION |")
		execution_total = 0
		for idx in range(0, self.current_layer):
			layer = self.layers[str(idx)]
			print("| %6.1d | %30s | %6s | %7s | %9d | %9d |" 
				%(idx, 
				layer['type'],
				layer['inputs'],
				layer['outputs'],
				layer['resources'],
				layer['execution']))
			execution_total = execution_total + layer['execution']

		print("Total execution cycles (delay): " + str(execution_total))

	def update_configuration_max_execution(self, max_execution):
		self.max_execution = max_execution
		for idx in self.layers:
			execution_target = max_execution
			layer = self.layers[idx]

			# check physically maximum execution
			if self.__get_min_execution_time(layer) > execution_target:
				execution_target = self.__get_min_execution_time(layer)
				print("NOTE: requested execution time exceeds maximum possible(" + layer['type'] + ")")


			while layer['execution'] > execution_target:
				layer['resources'] = layer['resources'] + 1
				self.__parse_configuration_set_execution_time()
		
		# self.show_configuration()

	def generate_testbench(self):
		# generate input data file
		self.__gen_data_file_array_2D(self.path_testbench + "/testbench_inputs.dat", self.json_test["inputs"])

		# generate output data file
		self.__gen_data_file_array_2D(self.path_testbench + "/testbench_outputs.dat", self.json_test["outputs"])

		# generate target data file
		self.__gen_data_file_array_2D(self.path_testbench + "/testbench_targets.dat", self.json_test["targets"])

		# generate testbench file
		fname = self.path_testbench + "/" + FNAME_CPP_TESTBENCH
		print(fname)
		fd = open(fname, "w")

		# includes
		fd.write("#include <iostream>\n")
		fd.write("#include <ap_fixed.h>\n")
		fd.write("#include \"nn.h\"\n\n")

		# defines
		fd.write("#define TEST_COUNT " + str(self.test_count) + "\n\n")


		# globally declared input test data
		fd.write("nn_t test_inputs[TEST_COUNT][NN_INPUT_COUNT] = {\n")
		fd.write("\t#include \"testbench_inputs.dat\"\n")
		fd.write("};\n\n")

		# globally declared output test data
		fd.write("nn_t test_outputs[TEST_COUNT][NN_OUTPUT_COUNT] = {\n")
		fd.write("\t#include \"testbench_outputs.dat\"\n")
		fd.write("};\n\n")

		# globally declared target test data
		fd.write("nn_t test_targets[TEST_COUNT][NN_OUTPUT_COUNT] = {\n")
		fd.write("\t#include \"testbench_targets.dat\"\n")
		fd.write("};\n\n")

		fd.write("int main(void)\n")
		fd.write("{\n")
		fd.write("\tnn_t outputs[NN_OUTPUT_COUNT];\n")
		fd.write("\tdouble err_outputs[TEST_COUNT][NN_OUTPUT_COUNT];\n")
		fd.write("\tdouble err_outputs_relative[TEST_COUNT][NN_OUTPUT_COUNT];\n")
		fd.write("\tdouble err_targets[TEST_COUNT][NN_OUTPUT_COUNT];\n")
		fd.write("\tdouble err_targets_relative[TEST_COUNT][NN_OUTPUT_COUNT];\n")
		fd.write("\tdouble err_outputs_relative_sum = 0.0;\n")
		fd.write("\tdouble err_targets_relative_sum = 0.0;\n")
		fd.write("\tdouble err_outputs_absolute_sum = 0.0;\n")
		fd.write("\tdouble err_targets_absolute_sum = 0.0;\n")
		fd.write("\tdouble err_outputs_absolute_sum_squared = 0.0;\n")
		fd.write("\tdouble err_targets_absolute_sum_squared = 0.0;\n")
		fd.write("\tdouble err_outputs_std_deviation = 0.0;\n")
		fd.write("\tdouble err_targets_std_deviation = 0.0;\n")
		fd.write("\tdouble err_outputs_absolute_mean = 0.0;\n")
		fd.write("\tdouble err_targets_absolute_mean = 0.0;\n")
		fd.write("\tdouble err_outputs_mean = 0.0;\n")
		fd.write("\tdouble err_targets_mean = 0.0;\n")
		fd.write("\tdouble err_outputs_absolute_max = 0.0;\n")
		fd.write("\tdouble err_targets_absolute_max = 0.0;\n")
		fd.write("\tdouble err_outputs_relative_percentage = 0.0;\n")
		fd.write("\tdouble err_targets_relative_percentage = 0.0;\n")
		fd.write("\tdouble err_outputs_relative_percentage_max = 0.0;\n")
		fd.write("\tdouble err_targets_relative_percentage_max = 0.0;\n")
		fd.write("\n")
		fd.write("\tstd::cout << \"Performing tests\" << std::endl;\n")
		fd.write("\tfor(int test=0; test<TEST_COUNT; test++){\n")
		fd.write("\t\tnn_top(outputs,test_inputs[test]);\n")
		fd.write("\t\tfor(int output=0; output<NN_OUTPUT_COUNT; output++){\n")
		fd.write("\t\t\tif(test_outputs[test][output] != 0){\n")
		fd.write("\t\t\t\terr_outputs[test][output] = (double)(test_outputs[test][output] - outputs[output]);\n")
		fd.write("\t\t\t\terr_outputs_relative[test][output] = ((double)err_outputs[test][output])/(double)test_outputs[test][output];\n")
		fd.write("\t\t\t}\n")
		fd.write("\t\t\tif(test_targets[test][output] != 0){\n")
		fd.write("\t\t\t\terr_targets[test][output] = (double)(test_targets[test][output] - outputs[output]);\n")
		fd.write("\t\t\t\terr_targets_relative[test][output] = ((double)err_targets[test][output])/(double)test_targets[test][output];\n")
		fd.write("\t\t\t}\n")
		fd.write("\t\t}\n")
		fd.write("\t}\n")
		fd.write("\n")
		fd.write("\tstd::cout << \"Calculating absolute and relative error mean values\" << std::endl;\n")
		fd.write("\tfor(int test=0; test<TEST_COUNT; test++){\n")
		fd.write("\t\tfor(int output=0; output<NN_OUTPUT_COUNT; output++){\n")
		fd.write("\t\t\terr_outputs_absolute_sum += abs(err_outputs[test][output]);\n")
		fd.write("\t\t\terr_targets_absolute_sum += abs(err_targets[test][output]);\n")
		fd.write("\t\t\terr_outputs_relative_sum += abs(err_outputs_relative[test][output]);\n")
		fd.write("\t\t\terr_targets_relative_sum += abs(err_targets_relative[test][output]);\n")
		fd.write("\t\t}\n")
		fd.write("\t}\n")
		fd.write("\terr_outputs_absolute_mean = err_outputs_absolute_sum/(TEST_COUNT*NN_OUTPUT_COUNT);\n")
		fd.write("\terr_targets_absolute_mean = err_targets_absolute_sum/(TEST_COUNT*NN_OUTPUT_COUNT);\n")
		fd.write("\terr_outputs_relative_percentage = err_outputs_relative_sum/(TEST_COUNT*NN_OUTPUT_COUNT);\n")
		fd.write("\terr_targets_relative_percentage = err_targets_relative_sum/(TEST_COUNT*NN_OUTPUT_COUNT);\n")
		fd.write("\n")
		fd.write("\tstd::cout << \"Finding absolute and relative error maximum values\" << std::endl;\n")
		fd.write("\tfor(int test=0; test<TEST_COUNT; test++){\n")
		fd.write("\t\tfor(int output=0; output<NN_OUTPUT_COUNT; output++){\n")
		fd.write("\t\t\tif(abs(err_outputs[test][output]) > err_outputs_absolute_max){\n")
		fd.write("\t\t\t\terr_outputs_absolute_max = abs(err_outputs[test][output]);\n")
		fd.write("\t\t\t}\n")
		fd.write("\t\t\tif(abs(err_targets[test][output]) > err_targets_absolute_max){\n")
		fd.write("\t\t\t\terr_targets_absolute_max = abs(err_targets[test][output]);\n")
		fd.write("\t\t\t\t}\n")
		fd.write("\t\t\tif(abs(err_outputs_relative[test][output]) > err_outputs_relative_percentage_max){\n")
		fd.write("\t\t\t\terr_outputs_relative_percentage_max = abs(err_outputs_relative[test][output]);\n")
		fd.write("\t\t\t}\n")
		fd.write("\t\t\tif(abs(err_targets_relative[test][output]) > err_targets_relative_percentage_max){\n")
		fd.write("\t\t\t\terr_targets_relative_percentage_max = abs(err_targets_relative[test][output]);\n")
		fd.write("\t\t\t}\n")
		fd.write("\t\t}\n")
		fd.write("\t}\n")
		fd.write("\n")
		fd.write("\tstd::cout << \"Calculating error mean value\" << std::endl;\n")
		fd.write("\tfor(int test=0; test<TEST_COUNT; test++){\n")
		fd.write("\t\tfor(int output=0; output<NN_OUTPUT_COUNT; output++){\n")
		fd.write("\t\t\terr_outputs_mean += err_outputs[test][output];\n")
		fd.write("\t\t\terr_targets_mean += err_targets[test][output];\n")
		fd.write("\t\t}\n")
		fd.write("\t}\n")
		fd.write("\terr_outputs_mean = err_outputs_mean/(TEST_COUNT*NN_OUTPUT_COUNT);\n")
		fd.write("\terr_targets_mean = err_targets_mean/(TEST_COUNT*NN_OUTPUT_COUNT);\n")
		fd.write("\n")
		fd.write("\tstd::cout << \"Calculating standard deviation\" << std::endl;\n")
		fd.write("\tfor(int test=0; test<TEST_COUNT; test++){\n")
		fd.write("\t\tfor(int output=0; output<NN_OUTPUT_COUNT; output++){\n")
		fd.write("\t\t\terr_outputs_absolute_sum_squared += (err_outputs[test][output] - err_outputs_mean)*(err_outputs[test][output] - err_outputs_mean);\n")
		fd.write("\t\t\terr_targets_absolute_sum_squared += (err_targets[test][output] - err_targets_mean)*(err_targets[test][output] - err_targets_mean);\n")
		fd.write("\t\t}\n")
		fd.write("\t}\n")
		fd.write("\terr_outputs_std_deviation = sqrt(err_outputs_absolute_sum_squared/(TEST_COUNT*NN_OUTPUT_COUNT-1));\n")
		fd.write("\terr_targets_std_deviation = sqrt(err_targets_absolute_sum_squared/(TEST_COUNT*NN_OUTPUT_COUNT-1));\n")
		fd.write("\n")
		fd.write("\n")
		fd.write("\tstd::cout << \"Outputs\" << std::endl;\n")
		fd.write("\tstd::cout << \"Absolute Error Mean: \" << err_outputs_absolute_mean<< \" \" << 100.0*err_outputs_relative_percentage << \"%\" << std::endl;\n")
		fd.write("\tstd::cout << \"Absolute Error Max: \" << err_outputs_absolute_max<< \" \" << 100.0*err_outputs_relative_percentage_max << \"%\" << std::endl;\n")
		fd.write("\tstd::cout << \"Error Mean: \" << err_outputs_mean << std::endl;\n")
		fd.write("\tstd::cout << \"Error Mean Standard Deviation: \" << err_outputs_std_deviation << std::endl;\n")
		fd.write("\tstd::cout << \"Targets\" << std::endl;\n")
		fd.write("\tstd::cout << \"Absolute Error Mean: \" << err_targets_absolute_mean<< \" \" << 100.0*err_targets_relative_percentage << \"%\" << std::endl;\n")
		fd.write("\tstd::cout << \"Absolute Error Max: \" << err_targets_absolute_max<< \" \" << 100.0*err_targets_relative_percentage_max << \"%\" << std::endl;\n")
		fd.write("\tstd::cout << \"Error Mean: \" << err_targets_mean << std::endl;\n")
		fd.write("\tstd::cout << \"Error Mean Standard Deviation: \" << err_targets_std_deviation << std::endl;\n")
		fd.write("\n")
		fd.write("\treturn 0;\n")
		fd.write("}\n")
		
		fd.close()


	def generate_implementation(self):
		# generate paths
		if not os.path.exists(self.path_output):
			os.makedirs(self.path_output)

		if not os.path.exists(self.path_data):
			os.makedirs(self.path_data)

		self.__generate_data_nn()

		self.__generate_header_nn()

		self.__generate_source_nn()

	def __generate_data_nn(self):
		for idx in range(0,self.current_layer):
			layer = self.layers[str(idx)]
			if layer['type'] == "normalization_input_offset":
				self.__gen_coef_file_array_1D(idx, layer)
			elif layer['type'] == "normalization_input_gain":
				self.__gen_coef_file_array_1D(idx, layer)
			elif layer['type'] == "normalization_input_min":
				self.__gen_coef_file_value(idx, layer)
			elif layer['type'] == "normalization_output_offset":
				self.__gen_coef_file_array_1D(idx, layer)
			elif layer['type'] == "normalization_output_gain":
				self.__gen_coef_file_array_1D(idx, layer)
			elif layer['type'] == "normalization_output_min":
				self.__gen_coef_file_value(idx, layer)
			elif layer['type'] == "multiplication":
				self.__gen_coef_file_array_2D(idx, layer)
			elif layer['type'] == "addition":
				self.__gen_coef_file_array_1D(idx, layer)
			elif layer['type'] == "activation_tansig":
				continue
			elif layer['type'] == "activation_tansig_lut":
				continue
			elif layer['type'] == "activation_linear":
				self.__gen_coef_file_array_1D(idx, layer)
			else:
				raise ValueError('This layer type is not supported')

	def __gen_coef_file_value(self, idx, layer):
		fpath = self.path_data + "/l" + str(idx) + "_coef_" + layer['type'] + ".dat"
		print("generating \"" + fpath + "\"")

		fd = open(fpath, "w")
		fd.write(str(layer['coef']))
		fd.write("\n")
		fd.close()

	def __gen_coef_file_array_1D(self, idx, layer):
		fpath = self.path_data + "/l" + str(idx) + "_coef_" + layer['type'] + ".dat"
		print("generating \"" + fpath + "\"")
		
		fd = open(fpath, "w")
		for i in layer['coef']:
			fd.write(str(i) + ",")
		fd.write("\n")
		fd.close()

	def __gen_coef_file_array_2D(self, idx, layer):
		fpath = self.path_data + "/l" + str(idx) + "_coef_" + layer['type'] + ".dat"
		print("generating \"" + fpath + "\"")

		fd = open(fpath, "w")
		for array in layer['coef']:
			fd.write("{")
			for i in array:
				fd.write(str(i) + ",")
			fd.write("},\n")
		fd.write("\n")
		fd.close()

	def __gen_data_file_array_2D(self, fpath, data):
		print("generating \"" + fpath + "\"")
		
		fd = open(fpath, "w")
		for array in data:
			fd.write("{")
			for i in array:
				fd.write(str(i) + ",")
			fd.write("},\n")
		fd.write("\n")
		fd.close()

	def __gen_coef_instantation(self, fd, idx, layer):
		if layer['type'] == "normalization_input_offset":
			self.__gen_coef_instantation_array_1D(fd, idx, layer)
		elif layer['type'] == "normalization_input_gain":
			self.__gen_coef_instantation_array_1D(fd, idx, layer)
		elif layer['type'] == "normalization_input_min":
			self.__gen_coef_instantation_value(fd, idx, layer)
		elif layer['type'] == "normalization_output_offset":
			self.__gen_coef_instantation_array_1D(fd, idx, layer)
		elif layer['type'] == "normalization_output_gain":
			self.__gen_coef_instantation_array_1D(fd, idx, layer)
		elif layer['type'] == "normalization_output_min":
			self.__gen_coef_instantation_value(fd, idx, layer)
		elif layer['type'] == "multiplication":
			self.__gen_coef_instantation_array_2D(fd, idx, layer)
		elif layer['type'] == "addition":
			self.__gen_coef_instantation_array_1D(fd, idx, layer)
		elif layer['type'] == "activation_tansig":
			return
		elif layer['type'] == "activation_tansig_lut":
			return
		elif layer['type'] == "activation_linear":
			self.__gen_coef_instantation_array_1D(fd, idx, layer)
		else:
			raise ValueError('This layer type is not supported')

	def __gen_coef_instantation_value(self, fd, idx, layer):
		vname = "l" + str(idx) + "_coef_" + layer['type']
		ipath = DIR_DATA + "/l" + str(idx) + "_coef_" + layer['type'] + ".dat"
		fd.write("nn_t " + vname + " = \n")
		fd.write("\t#include \"" + ipath + "\"\n")
		fd.write(";\n\n")

	def __gen_coef_instantation_array_1D(self, fd, idx, layer):
		vname = "l" + str(idx) + "_coef_" + layer['type']
		ipath = DIR_DATA + "/l" + str(idx) + "_coef_" + layer['type'] + ".dat"
		fd.write("nn_t " + vname + "[" + str(len(layer['coef'])) + "] = {\n")
		fd.write("\t#include \"" + ipath + "\"\n")
		fd.write("};\n\n")

	def __gen_coef_instantation_array_2D(self, fd, idx, layer):
		vname = "l" + str(idx) + "_coef_" + layer['type']
		ipath = DIR_DATA + "/l" + str(idx) + "_coef_" + layer['type'] + ".dat"
		fd.write("nn_t " + vname + "[" + str(len(layer['coef'])) + "]" + "[" + str(len(layer['coef'][0])) + "] = {\n")
		fd.write("\t#include \"" + ipath + "\"\n")
		fd.write("};\n\n")

	def __generate_header_nn(self):
		print("generating \"" + self.path_output+"/"+FNAME_CPP_HEADER + "\"")

		fd = open(self.path_output+"/"+FNAME_CPP_HEADER, "w")
		fd.write('#ifndef _' + FNAME_CPP_HEADER.replace("/", "_").replace(".", "_").upper() + '_\n')
		fd.write('#define _' + FNAME_CPP_HEADER.replace("/", "_").replace(".", "_").upper() + '_\n\n')
		
		# lay out neural network constants
		for idx in range(0, self.current_layer):
			fd.write("#define LAYER_" + str(idx) + "_INPUTS  \t\t" + str(self.__get_inout_count(self.layers[str(idx)]['inputs'])) + "\n")
			fd.write("#define LAYER_" + str(idx) + "_OUTPUTS \t\t" + str(self.__get_inout_count(self.layers[str(idx)]['outputs'])) + "\n")

		fd.write("\n")

		# lay out additional neural network constants
		for idx in range(0, self.current_layer):
			if self.layers[str(idx)]['type'] == "multiplication":
				split_inputs  = self.layers[str(idx)]['inputs'].split('x')
				split_outputs = self.layers[str(idx)]['outputs'].split('x')
				fd.write("#define LAYER_" + str(idx) + "_OUTPUTS_OUTER  \t" + str(split_outputs[0]) + "\n")
				fd.write("#define LAYER_" + str(idx) + "_OUTPUTS_INNER  \t" + str(split_outputs[1]) + "\n")
			if self.layers[str(idx)]['type'] == "addition":
				split_inputs  = self.layers[str(idx)]['inputs'].split('x')
				fd.write("#define LAYER_" + str(idx) + "_INPUTS_OUTER  \t" + str(split_inputs[0]) + "\n")
				fd.write("#define LAYER_" + str(idx) + "_INPUTS_INNER  \t" + str(split_inputs[1]) + "\n")
		fd.write("\n")


		# lay out neural network inout constants
		fd.write("#define NN_INPUT_COUNT  LAYER_0_INPUTS\n")
		fd.write("#define NN_OUTPUT_COUNT LAYER_" + str(self.current_layer-1) + "_OUTPUTS\n\n")

		# lay out neural network LUT constants and macros
		for idx in range(0, self.current_layer):
			if self.layers[str(idx)]['type'] == "activation_tansig_lut":
				fd.write("#define WIDTH_LUT_INPUT "        + str(self.width_lut_input) + "\n")
				fd.write("#define WIDTH_LUT_INPUT_WHOLE "  + str(self.width_lut_input_whole) + "\n")
				fd.write("#define WIDTH_LUT_OUTPUT "       + str(self.width_lut_output) + "\n")
				fd.write("#define WIDTH_LUT_OUTPUT_WHOLE " + str(self.width_lut_output_whole) + "\n")
				fd.write("#define WIDTH_LUT_INPUT_FRAC        (WIDTH_LUT_INPUT-WIDTH_LUT_INPUT_WHOLE)\n")
				fd.write("#define WIDTH_LUT_OUTPUT_FRAC       (WIDTH_LUT_OUTPUT-WIDTH_LUT_OUTPUT_WHOLE)\n")
				fd.write("#define WIDTH_LUT_INPUT_CAST        (WIDTH_LUT_INPUT+WIDTH_LUT_INPUT_FRAC)\n")
				fd.write("#define WIDTH_LUT_INPUT_CAST_WHOLE  (WIDTH_LUT_INPUT_WHOLE+WIDTH_LUT_INPUT_FRAC)\n\n")

				fd.write("#define AP_FIXED_MAX_VALUE(width,width_whole) ((ap_fixed<width,width_whole>)( pow(2,width_whole-1) - pow(2,width_whole-width)))\n")
				fd.write("#define AP_FIXED_MIN_VALUE(width,width_whole) ((ap_fixed<width,width_whole>)(-pow(2,width_whole-1)))\n")
				break

		# type definition
		fd.write("\ntypedef " + self.dtype + " nn_t;\n")

		# ltype definitions for LUT
		for idx in range(0, self.current_layer):
			if self.layers[str(idx)]['type'] == "activation_tansig_lut":
				fd.write("typedef ap_fixed<WIDTH_LUT_INPUT,WIDTH_LUT_INPUT_WHOLE>            lut_in_t;\n")
				fd.write("typedef ap_ufixed<WIDTH_LUT_INPUT_CAST,WIDTH_LUT_INPUT_CAST_WHOLE> lut_in_cast_t;\n")
				fd.write("typedef ap_fixed<WIDTH_LUT_OUTPUT,WIDTH_LUT_OUTPUT_WHOLE>          lut_out_t;\n")
				break

		# function declaration
		fd.write("\nvoid nn_top(nn_t outputs[NN_OUTPUT_COUNT], nn_t inputs[NN_INPUT_COUNT]);\n")

		fd.write("\n#endif\n")
		fd.close()

	def __generate_source_nn(self):
		print("generating \"" + self.path_output+"/"+FNAME_CPP_SOURCE + "\"")
		
		fd = open(self.path_output+"/"+FNAME_CPP_SOURCE, "w")

		# includes
		fd.write("#include <hls_math.h>\n")
		fd.write("#include \"nn.h\"\n\n")

		# generate constants
		for idx in range(0, self.current_layer):
			if self.__type_should_have_coefficients(self.layers[str(idx)]["type"]):
				self.__gen_coef_instantation(fd, str(idx), self.layers[str(idx)])

		# generate support functions
		for idx in range(0, self.current_layer):
			if self.layers[str(idx)]['type'] == "activation_tansig_lut":
				self.__gen_tansig_lut_support(fd)
				break

		# generate resources
		for idx in range(0, self.current_layer):
			self.__gen_resource(fd, str(idx), self.layers[str(idx)])

		# main hw acceleration function
		fd.write("\n\n\nvoid nn_top(nn_t outputs[NN_OUTPUT_COUNT], nn_t inputs[NN_INPUT_COUNT])\n")
		fd.write("{\n")

		# generate pragmas
		self.generate_pragmas_interface(fd)
		if self.interface == "s_axilite":
			self.generate_pragmas_pipeline(fd)
		elif self.interface == "s_axis":
			fd.write("#pragma HLS PIPELINE II=" + str(self.max_execution) + " enable_flush\n")
		fd.write("#pragma HLS ARRAY_PARTITION variable=inputs complete dim=1\n")
		fd.write("#pragma HLS ARRAY_PARTITION variable=outputs complete dim=1\n")

		# generate output containers
		for idx in range(0, self.current_layer-1):
			vname = "output_" + str(idx)
			dname = "LAYER_" + str(idx) + "_OUTPUTS"
			if self.layers[str(idx)]['type'] == "multiplication":
				fd.write("\tnn_t " + vname + "[" + dname + "_OUTER][" + dname + "_INNER];\n")
			else:
				fd.write("\tnn_t " + vname + "[" + dname + "];\n")
			fd.write("#pragma HLS ARRAY_PARTITION variable=" + vname + " complete dim=0\n")

		# generate resource calls
		for idx in range(0, self.current_layer):
			if idx == 0:
				vname_in  = "inputs"
			else:
				vname_in  = "output_" + str(idx-1)

			if idx == self.current_layer-1:
				vname_out = "outputs"
			else:
				vname_out = "output_" + str(idx)
				
			rname = "l" + str(idx) + "_resource_" + self.layers[str(idx)]['type']
			fd.write("\n\t" + rname + "(" + vname_out + ", " + vname_in + ");\n")

		fd.write("}\n")

	def __gen_tansig_lut_support(self, fd):
		# saturate
		fd.write("lut_in_t tanh_saturate(nn_t input)\n")
		fd.write("{\n")
		fd.write("\tlut_in_t lut_input;\n")
		fd.write("\tif(input > AP_FIXED_MAX_VALUE(WIDTH_LUT_INPUT,WIDTH_LUT_INPUT_WHOLE)){\n")
		fd.write("\t\treturn AP_FIXED_MAX_VALUE(WIDTH_LUT_INPUT,WIDTH_LUT_INPUT_WHOLE);\n")
		fd.write("\t}else if (input < AP_FIXED_MIN_VALUE(WIDTH_LUT_INPUT,WIDTH_LUT_INPUT_WHOLE)){\n")
		fd.write("\t\treturn AP_FIXED_MIN_VALUE(WIDTH_LUT_INPUT,WIDTH_LUT_INPUT_WHOLE);\n")
		fd.write("\t}else{\n")
		fd.write("\t\treturn (lut_in_t)input;\n")
		fd.write("\t}\n")
		fd.write("}\n\n")

		# LUT abstraction
		fname_lut_data = "lut_tanh_" + str(self.width_lut_output) + "_" + str(self.width_lut_output_whole) + "__" + str(self.width_lut_input) + "_" + str(self.width_lut_input_whole) + ".dat"
		fd.write("lut_out_t tanh(lut_in_t input)\n")
		fd.write("{\n")
		fd.write("\tlut_out_t lut_tanh[] = {\n")
		fd.write("\t\t#include \"data/" + fname_lut_data + "\"\n")
		fd.write("\t};\n")
		fd.write("\t#pragma HLS RESOURCE variable=lut_tanh core=ROM_1P_BRAM\n")
		fd.write("\tunsigned address = (unsigned)((lut_in_cast_t)input << WIDTH_LUT_INPUT_FRAC);\n")
		fd.write("\treturn lut_tanh[address];\n")
		fd.write("}\n\n")

	def __gen_resource(self, fd, idx, layer):
		if layer["type"] == "normalization_input_offset":
			self.__gen_resource_normalization_input_offset(fd, idx, layer)
		elif layer["type"] == "normalization_input_gain":
			self.__gen_resource_normalization_input_gain(fd, idx, layer)
		elif layer["type"] == "normalization_input_min":
			self.__gen_resource_normalization_input_min(fd, idx, layer)
		elif layer["type"] == "normalization_output_offset":
			self.__gen_resource_normalization_output_offset(fd, idx, layer)
		elif layer["type"] == "normalization_output_gain":
			self.__gen_resource_normalization_output_gain(fd, idx, layer)
		elif layer["type"] == "normalization_output_min":
			self.__gen_resource_normalization_output_min(fd, idx, layer)
		elif layer["type"] == "multiplication":
			self.__gen_resource_multiplication(fd, idx, layer)
		elif layer["type"] == "addition":
			self.__gen_resource_addition(fd, idx, layer)
		elif layer["type"] == "activation_tansig":
			self.__gen_resource_activation_tansig(fd, idx, layer)
		elif layer["type"] == "activation_tansig_lut":
			self.__gen_resource_activation_tansig_lut(fd, idx, layer)
		elif layer["type"] == "activation_linear":
			self.__gen_resource_activation_linear(fd, idx, layer)
		else:
			raise ValueError('This layer type is not supported')

	def __gen_resource_normalization_input_offset(self, fd, idx, layer):
		rname  = "l" + str(idx) + "_resource_normalization_input_offset"
		dname0 = "LAYER_" + str(idx) + "_OUTPUTS"
		dname1 = "LAYER_" + str(idx) + "_INPUTS"
		cname  = "l" + str(idx) + "_coef_normalization_input_offset"
		fd.write("void " + rname + "(nn_t outputs[" + dname0 + "], nn_t inputs[" + dname1 + "])\n")
		fd.write("{\n")
		self.generate_pragmas_pipeline(fd)
		fd.write("#pragma HLS ALLOCATION instances=add limit=" + str(layer['resources']) + " operation\n")
		fd.write("\tfor(int i=0; i<" + dname0 + "; i++){\n")
		fd.write("\t#pragma HLS UNROLL\n")
		fd.write("\t\toutputs[i] = inputs[i] - " + cname + "[i];\n")
		fd.write("\t}\n")
		fd.write("}\n\n")

	def __gen_resource_normalization_input_gain(self, fd, idx, layer):
		rname  = "l" + str(idx) + "_resource_normalization_input_gain"
		dname0 = "LAYER_" + str(idx) + "_OUTPUTS"
		dname1 = "LAYER_" + str(idx) + "_INPUTS"
		cname  = "l" + str(idx) + "_coef_normalization_input_gain"
		fd.write("void " + rname + "(nn_t outputs[" + dname0 + "], nn_t inputs[" + dname1 + "])\n")
		fd.write("{\n")
		self.generate_pragmas_pipeline(fd)
		fd.write("#pragma HLS ALLOCATION instances=mul limit=" + str(layer['resources']) + " operation\n")
		fd.write("\tfor(int i=0; i<" + dname0 + "; i++){\n")
		fd.write("\t#pragma HLS UNROLL\n")
		fd.write("\t\toutputs[i] = inputs[i] * " + cname + "[i];\n")
		fd.write("\t}\n")
		fd.write("}\n\n")

	def __gen_resource_normalization_input_min(self, fd, idx, layer):
		rname  = "l" + str(idx) + "_resource_normalization_input_min"
		dname0 = "LAYER_" + str(idx) + "_OUTPUTS"
		dname1 = "LAYER_" + str(idx) + "_INPUTS"
		cname  = "l" + str(idx) + "_coef_normalization_input_min"
		fd.write("void " + rname + "(nn_t outputs[" + dname0 + "], nn_t inputs[" + dname1 + "])\n")
		fd.write("{\n")
		self.generate_pragmas_pipeline(fd)
		fd.write("#pragma HLS ALLOCATION instances=add limit=" + str(layer['resources']) + " operation\n")
		fd.write("\tfor(int i=0; i<" + dname0 + "; i++){\n")
		fd.write("\t#pragma HLS UNROLL\n")
		fd.write("\t\toutputs[i] = inputs[i] + " + cname + ";\n")
		fd.write("\t}\n")
		fd.write("}\n\n")

	def __gen_resource_normalization_output_offset(self, fd, idx, layer):
		rname  = "l" + str(idx) + "_resource_normalization_output_offset"
		dname0 = "LAYER_" + str(idx) + "_OUTPUTS"
		dname1 = "LAYER_" + str(idx) + "_INPUTS"
		cname  = "l" + str(idx) + "_coef_normalization_output_offset"
		fd.write("void " + rname + "(nn_t outputs[" + dname0 + "], nn_t inputs[" + dname1 + "])\n")
		fd.write("{\n")
		self.generate_pragmas_pipeline(fd)
		fd.write("#pragma HLS ALLOCATION instances=add limit=" + str(layer['resources']) + " operation\n")
		fd.write("\tfor(int i=0; i<" + dname0 + "; i++){\n")
		fd.write("\t#pragma HLS UNROLL\n")
		fd.write("\t\toutputs[i] = inputs[i] - " + cname + "[i];\n")
		fd.write("\t}\n")
		fd.write("}\n\n")

	def __gen_resource_normalization_output_gain(self, fd, idx, layer):
		rname  = "l" + str(idx) + "_resource_normalization_output_gain"
		dname0 = "LAYER_" + str(idx) + "_OUTPUTS"
		dname1 = "LAYER_" + str(idx) + "_INPUTS"
		cname  = "l" + str(idx) + "_coef_normalization_output_gain"
		fd.write("void " + rname + "(nn_t outputs[" + dname0 + "], nn_t inputs[" + dname1 + "])\n")
		fd.write("{\n")
		self.generate_pragmas_pipeline(fd)
		fd.write("#pragma HLS ALLOCATION instances=mul limit=" + str(layer['resources']) + " operation\n")
		fd.write("\tfor(int i=0; i<" + dname0 + "; i++){\n")
		fd.write("\t#pragma HLS UNROLL\n")
		fd.write("\t\toutputs[i] = inputs[i] * " + cname + "[i];\n")
		fd.write("\t}\n")
		fd.write("}\n\n")

	def __gen_resource_normalization_output_min(self, fd, idx, layer):
		rname  = "l" + str(idx) + "_resource_normalization_output_min"
		dname0 = "LAYER_" + str(idx) + "_OUTPUTS"
		dname1 = "LAYER_" + str(idx) + "_INPUTS"
		cname  = "l" + str(idx) + "_coef_normalization_output_min"
		fd.write("void " + rname + "(nn_t outputs[" + dname0 + "], nn_t inputs[" + dname1 + "])\n")
		fd.write("{\n")
		self.generate_pragmas_pipeline(fd)
		fd.write("#pragma HLS ALLOCATION instances=add limit=" + str(layer['resources']) + " operation\n")
		fd.write("\tfor(int i=0; i<" + dname0 + "; i++){\n")
		fd.write("\t#pragma HLS UNROLL\n")
		fd.write("\t\toutputs[i] = inputs[i] - " + cname + ";\n")
		fd.write("\t}\n")
		fd.write("}\n\n")

	def __gen_resource_multiplication(self, fd, idx, layer):
		rname  = "l" + str(idx) + "_resource_multiplication"
		dname0 = "LAYER_" + str(idx) + "_OUTPUTS_OUTER"
		dname1 = "LAYER_" + str(idx) + "_OUTPUTS_INNER"
		dname2 = "LAYER_" + str(idx) + "_INPUTS"
		cname  = "l" + str(idx) + "_coef_multiplication"
		fd.write("void " + rname + "(nn_t outputs[" + dname0 + "][" + dname1 + "], nn_t inputs[" + dname2 + "])\n")
		fd.write("{\n")
		self.generate_pragmas_pipeline(fd)
		fd.write("#pragma HLS ALLOCATION instances=Mul limit=" + str(layer['resources']) + " operation\n")
		fd.write("\tfor(int i=0; i<" + dname0 + "; i++){\n")
		fd.write("\t#pragma HLS UNROLL\n")
		fd.write("\t\tfor(int j=0; j<" + dname1 + "; j++){\n" )
		fd.write("\t\t\toutputs[i][j] = inputs[j] * " + cname + "[i][j];\n")
		fd.write("\t\t}\n")
		fd.write("\t}\n")
		fd.write("}\n\n")

	def __gen_resource_addition(self, fd, idx, layer):
		rname  = "l" + str(idx) + "_resource_addition"
		dname0 = "LAYER_" + str(idx) + "_INPUTS_OUTER"
		dname1 = "LAYER_" + str(idx) + "_INPUTS_INNER"
		dname2 = "LAYER_" + str(idx) + "_OUTPUTS"
		cname  = "l" + str(idx) + "_coef_addition"
		fd.write("void " + rname + "(nn_t outputs[" + dname2 + "], nn_t inputs[" + dname0 + "][" + dname1 + "])\n")
		fd.write("{\n")
		self.generate_pragmas_pipeline(fd)
		fd.write("#pragma HLS ALLOCATION instances=add limit=" + str(layer['resources']) + " operation\n")
		fd.write("\tfor(int i=0; i<" + dname0 + "; i++){\n")
		fd.write("\t#pragma HLS UNROLL\n")
		fd.write("\t\toutputs[i] = " + cname + "[i];\n" )
		fd.write("\t\tfor(int j=0; j<" + dname1 + "; j++){\n" )
		fd.write("\t\t\toutputs[i] += inputs[i][j];\n")
		fd.write("\t\t}\n")
		fd.write("\t}\n")
		fd.write("}\n\n")

	def __gen_resource_activation_tansig(self, fd, idx, layer):
		rname  = "l" + str(idx) + "_resource_activation_tansig"
		dname0 = "LAYER_" + str(idx) + "_OUTPUTS"
		dname1 = "LAYER_" + str(idx) + "_INPUTS"
		cname  = "l" + str(idx) + "_coef_activation_tansig"
		fd.write("void " + rname + "(nn_t outputs[" + dname0 + "], nn_t inputs[" + dname1 + "])\n")
		fd.write("{\n")
		self.generate_pragmas_pipeline(fd)
		fd.write("\tfor(int i=0; i<" + dname0 + "; i++){\n")
		fd.write("\t#pragma HLS UNROLL\n")
		fd.write("\t\toutputs[i] = tanh(inputs[i]);\n")
		fd.write("\t}\n")
		fd.write("}\n\n")

	def __gen_resource_activation_tansig_lut(self, fd, idx, layer):
		rname  = "l" + str(idx) + "_resource_activation_tansig_lut"
		dname0 = "LAYER_" + str(idx) + "_OUTPUTS"
		dname1 = "LAYER_" + str(idx) + "_INPUTS"
		fd.write("void " + rname + "(nn_t outputs[" + dname0 + "], nn_t inputs[" + dname1 + "])\n")
		fd.write("{\n")
		self.generate_pragmas_pipeline(fd)
		fd.write("#pragma HLS ALLOCATION instances=tanh limit=" + str(layer['resources']) + " function\n")
		fd.write("#pragma HLS ALLOCATION instances=tanh_saturate limit=" + str(layer['resources']) + " function\n")
		fd.write("\tfor(int i=0; i<" + dname0 + "; i++){\n")
		fd.write("\t#pragma HLS UNROLL\n")
		fd.write("\t\tlut_in_t input_saturated = tanh_saturate(inputs[i]);\n")
		fd.write("\t\toutputs[i] = tanh(input_saturated);\n")
		fd.write("\t}\n")
		fd.write("}\n\n")

	def __gen_resource_activation_linear(self, fd, idx, layer):
		rname  = "l" + str(idx) + "_resource_activation_linear"
		dname0 = "LAYER_" + str(idx) + "_OUTPUTS"
		dname1 = "LAYER_" + str(idx) + "_INPUTS"
		cname  = "l" + str(idx) + "_coef_activation_linear"
		fd.write("void " + rname + "(nn_t outputs[" + dname0 + "], nn_t inputs[" + dname1 + "])\n")
		fd.write("{\n")
		self.generate_pragmas_pipeline(fd)
		fd.write("#pragma HLS ALLOCATION instances=Mul limit=" + str(layer['resources']) + " operation\n")
		fd.write("\tfor(int i=0; i<" + dname0 + "; i++){\n")
		fd.write("\t#pragma HLS UNROLL\n")
		fd.write("\t\toutputs[i] = inputs[i] * " + cname + "[i];\n")
		fd.write("\t}\n")
		fd.write("}\n\n")

	def __parse_configuration_layers(self):
		for layer in self.json:
			self.layers[str(self.current_layer)] = {
			'type'     : layer['type'], 
			'inputs'   : layer['inputs'], 
			'outputs'  : layer['outputs'], 
			'resources': DEFAULT_RESOURCE_COUNT,
			'coef'     : layer['coefficients'],
			'execution': 0}

			self.__increment_current_layer()

	def __increment_current_layer(self):
		self.current_layer = self.current_layer + 1

	def __get_inout_count(self, input_string):
		if input_string.find('x') == -1:
			return int(input_string)
		else:
			return int(input_string.split('x')[0])*int(input_string.split('x')[1])

	def __type_should_have_coefficients(self, type):
		if type == "normalization_input_offset":
			return True
		if type == "normalization_input_gain":
			return True
		if type == "normalization_input_min":
			return True
		if type == "normalization_output_offset":
			return True
		if type == "normalization_output_gain":
			return True
		if type == "normalization_output_min":
			return True
		if type == "multiplication":
			return True
		if type == "addition":
			return True
		if type == "activation_linear":
			return True
		
		return False


	def __get_min_execution_time(self, layer):
		if layer['type'] == "normalization_input_offset":
			return 1
		elif layer['type'] == "normalization_input_gain":
			return 1
		elif layer['type'] == "normalization_input_min":
			return 1
		elif layer['type'] == "normalization_output_offset":
			return 1
		elif layer['type'] == "normalization_output_gain":
			return 1
		elif layer['type'] == "normalization_output_min":
			return 1
		elif layer['type'] == "multiplication":
			return 1
		elif layer['type'] == "addition":
			return math.ceil(math.log(int(layer['inputs'].split('x')[1]) + 1,2))
		elif layer['type'] == "activation_tansig":
			return 1
		elif layer['type'] == "activation_tansig_lut":
			return 1
		elif layer['type'] == "activation_linear":
			return 1
		else:
			raise ValueError('This layer type is not supported')


	def __parse_configuration_set_execution_time(self):
		for idx in self.layers:
			layer = self.layers[idx]
			input_count  = self.__get_inout_count(layer['inputs'])
			output_count = self.__get_inout_count(layer['outputs'])
			if layer['type'] == "normalization_input_offset":
				layer['execution'] = self.__parse_configuration_get_execution_time_offset(input_count, output_count, layer['resources'])
			elif layer['type'] == "normalization_input_gain":
				layer['execution'] = self.__parse_configuration_get_execution_time_mulitplication(input_count, output_count, layer['resources'])
			elif layer['type'] == "normalization_input_min":
				layer['execution'] = self.__parse_configuration_get_execution_time_offset(input_count, output_count, layer['resources'])
			elif layer['type'] == "normalization_output_offset":
				layer['execution'] = self.__parse_configuration_get_execution_time_offset(input_count, output_count, layer['resources'])
			elif layer['type'] == "normalization_output_gain":
				layer['execution'] = self.__parse_configuration_get_execution_time_mulitplication(input_count, output_count, layer['resources'])
			elif layer['type'] == "normalization_output_min":
				layer['execution'] = self.__parse_configuration_get_execution_time_offset(input_count, output_count, layer['resources'])
			elif layer['type'] == "multiplication":
				layer['execution'] = self.__parse_configuration_get_execution_time_mulitplication(input_count, output_count, layer['resources'])
			elif layer['type'] == "addition":
				layer['execution'] = self.__parse_configuration_get_execution_time_addition(layer['inputs'], layer['resources'])
			elif layer['type'] == "activation_tansig":
				layer['execution'] = self.__parse_configuration_get_execution_time_activation_tansig(input_count, output_count, layer['resources'])
			elif layer['type'] == "activation_tansig_lut":
				layer['execution'] = self.__parse_configuration_get_execution_time_activation_tansig_lut(input_count, output_count, layer['resources'])
			elif layer['type'] == "activation_linear":
				layer['execution'] = self.__parse_configuration_get_execution_time_activation_linear(input_count, output_count, layer['resources'])
			else:
				raise ValueError('This layer type is not supported')

	def __parse_configuration_get_execution_time_addition(self, inputs, resources):
		inputs_outer = int(inputs.split('x')[1])
		inputs_inner = int(inputs.split('x')[0])

		input_array = np.repeat(inputs_inner+1, inputs_outer)
		return self.__parse_configuration_execution_time_addition_recursive(input_array, resources, 0)


	def __parse_configuration_execution_time_addition_recursive(self, inputs_array, resources_max, resources_carry):
		resources_used = resources_carry
		for i in range(0, len(inputs_array)):
			if inputs_array[i] == 1:
				continue

			# update resources, input/output count
			resources_used = resources_used + inputs_array[i]/2
			inputs_array[i] = inputs_array[i] - inputs_array[i]/2

		# calculate delay
		delay = resources_used / resources_max
		resources_carry = resources_used % resources_max
		
		# check if recursive call
		for i in inputs_array:
			if i != 1:
				return delay + self.__parse_configuration_execution_time_addition_recursive(inputs_array, resources_max, resources_carry)

		# no recursion, return
		if resources_carry == 0:
			return delay
		else:
			return delay + 1


	def __parse_configuration_get_execution_time_mulitplication(self, inputs, outputs, resources):
		return RESOURCE_EXECUTION_MULTIPLICATION *outputs / resources

	def __parse_configuration_get_execution_time_offset(self, inputs, outputs, resources):
		return RESOURCE_EXECUTION_ADDITION * inputs / resources

	def __parse_configuration_get_execution_time_activation_tansig(self, inputs, outputs, resources):
		return RESOURCE_EXECUTION_ACTIVATION_TANSIG * inputs / resources

	def __parse_configuration_get_execution_time_activation_tansig_lut(self, inputs, outputs, resources):
		return RESOURCE_EXECUTION_ACTIVATION_TANSIG_LUT * inputs / resources

	def __parse_configuration_get_execution_time_activation_linear(self, inputs, outputs, resources):
		return RESOURCE_EXECUTION_ACTIVATION_LINEAR *outputs / resources

	def generate_pragmas_interface(self, fd):
		if self.interface == "s_axilite":
			fd.write("#pragma HLS INTERFACE s_axilite port=outputs\n")
			fd.write("#pragma HLS INTERFACE s_axilite port=inputs\n")
			fd.write("#pragma HLS INTERFACE s_axilite port=return\n")
		elif self.interface == "s_axis":
			fd.write("#pragma HLS INTERFACE axis register both port=outputs\n")
			fd.write("#pragma HLS INTERFACE axis register both port=input\n")
			fd.write("#pragma HLS INTERFACE ap_ctrl_none port=return\n")
		else:
			raise ValueError('Interface type is not supported')

	def generate_pragmas_pipeline(self, fd):
		fd.write("#pragma HLS PIPELINE II=" + str(self.max_execution) + "\n")
