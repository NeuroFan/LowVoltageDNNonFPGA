#!/usr/bin/python3
import os
import sys
import tty
import termios
import argparse
from c_nn import *

# PATH_NN_OBJECT = '../topology_example.json'
PATH_NN_OBJECT = '../topology_8_16_12_8_4.json'
PATH_TEST_DATA = '../topology_example_test.json'


############### DEFAULT MODE ###############
def mode_default(pathOut, pathNN, pathTest, ii, dtype, interface):
    # load neural network object
    obj = nn(pathNN)
    
    # parse configuration
    obj.parse_configuration()
    
    # show starting configuration
    obj.show_configuration()

    # update configuration
    obj.update_configuration_max_execution(int(ii))

    # update configuration
    obj.set_dtype(dtype)

    # select interface
    obj.set_interface(interface)
    
    # set test data object
    obj.set_test_file(pathTest)
    
    # set relevant paths
    obj.set_path_output(pathOut)
    obj.set_path_testbench(pathOut+"/tb")

    # show final configuration
    obj.show_configuration()

    # generate sources for the output nn
    obj.generate_implementation()


############### INTERACTIVE MODE ###############
MODE_TOP       = 0
MODE_TOPOLOGY  = 1
MODE_DATA_TYPE = 2
MODE_INTERFACE = 3
MODE_OUTPUT    = 4
MODE_IINTERVAL = 5
MODE_TEST      = 6
MODE_EXIT      = 7

mode_current = MODE_TOP
def mode_interactiveClearScreen():
    os.system('cls' if os.name == 'nt' else 'clear')

def mode_interactiveReadCharRaw():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setraw(sys.stdin.fileno())
    ch = sys.stdin.read(1)
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def mode_interactivePrintActions():
    print("Type in character for the respective action")
    print("q - quit")
    print("t - set topology file path")
    print("d - select data type")
    print("i - select interface")
    print("o - set output directory")
    print("n - set iteration interval")
    print("s - set test input directory")
    print("g - generate")

def mode_interactiveStateTopology(obj):
    print("Please provide path to the topology file...")
    obj.load_config(sys.stdin.readline().rstrip())
    obj.parse_configuration()

def mode_interactiveStateDataType(obj):
    print("Please select data type: double, float or fixed...")
    dtype = sys.stdin.readline().rstrip()

    if dtype == "fixed":
        print("Please specify highest bit...")
        high = sys.stdin.readline().rstrip()
        print("Please specify lowest bit...")
        low = sys.stdin.readline().rstrip()
        dtype = "ap_fixed<"+high+","+low+">"
        obj.set_dtype(dtype)
    else:
        obj.set_dtype(dtype)

def mode_interactiveStateInterface(obj):
    ch = 0
    while(ch != "1" and ch != 2):
        print("Please select interface...")
        print("1 - s_axilite")
        print("2 - s_axis")
        ch = mode_interactiveReadCharRaw()
        if ch == 'q':
            quit()

    if ch == 1:
        obj.set_interface("s_axilite")
        
    if ch == 2:
        obj.set_interface("s_axis")

def mode_interactiveStateOutput(obj):
    print("TODO: Not yet implemented")

def mode_interactiveStateIterationInterval(obj):
    print("Please provide iteration interval...")
    ii = sys.stdin.readline().rstrip()
    obj.update_configuration_max_execution(int(ii))

def mode_interactiveStateTest(obj):
    print("TODO: Not yet implemented")

def mode_interactiveGenerate(obj):
    print("Generating implementation...")
    obj.generate_implementation()
    print("Done")
    quit()


def mode_interactive(pathOut, pathNN, pathTest, ii, dtype, interface):
    # create nn object and parse initial configuration
    obj = nn(pathNN)
    obj.parse_configuration()

    # set default configuration
    obj.update_configuration_max_execution(int(ii))
    obj.set_dtype(dtype)
    obj.set_interface(interface)
    obj.set_test_file(pathTest)
    obj.set_path_output(pathOut)
    obj.set_path_testbench(pathOut+"/tb")

    while(1):
        # print topology and help
        mode_interactiveClearScreen()
        obj.show_configuration()
        mode_interactivePrintActions()

        # read a single character in raw mode and reset terminal
        ch = mode_interactiveReadCharRaw()

        if ch == 'q':
            quit()
        elif ch == 't':
            mode_interactiveStateTopology(obj)
        elif ch == 'd':
            mode_interactiveStateDataType(obj)
        elif ch == 'i':
            mode_interactiveStateInterface(obj)
        elif ch == 'o':
            mode_interactiveStateOutput(obj)
        elif ch == 'n':
            mode_interactiveStateIterationInterval(obj)
        elif ch == 's':
            mode_interactiveStateTest(obj)
        elif ch == 'g':
            mode_interactiveGenerate(obj)
        else:
            mode_interactiveClearScreen()
            obj.show_configuration()
            mode_interactivePrintActions()



############### ARGUMENT PARSER  ###############
parser = argparse.ArgumentParser(
  description="Noisy sinusoid generation/testing helper script")

parser.add_argument(
  '--topology', dest='topology',
  default="topology.json",
  help='path to the JSON description of the topology')

parser.add_argument(
  '--dtype', dest='dtype',
  default="float",
  help='base data type for the neural network')

parser.add_argument(
  '--interface', dest='interface',
  default="s_axilite",
  choices=["s_axis","s_axilite"],
  help='interface to the neural network, either memory mapped or streaming')

parser.add_argument(
  '--output', dest='output',
  default="out",
  help='output path for the generated topology')

parser.add_argument(
  '--mode', dest='mode',
  default="default",
  choices=["default","interactive"],
  help='mode in which to explore neural network topology')

parser.add_argument(
  '--ii', dest='ii',
  default="128",
  help='target iteration interval')

parser.add_argument(
  '--test', dest='test',
  default=None,
  help='path to the JSON file with test vectors')

############### ACQUIRE ARGUMENTS ###############
args = parser.parse_args()
arg_topology  = args.topology
arg_dtype     = args.dtype
arg_interface = args.interface
arg_output    = args.output
arg_mode      = args.mode
arg_ii        = args.ii
arg_test      = args.test

print("Running script with the following arguments")
print("- topology  = " + args.topology)
print("- dtype     = " + args.dtype)
print("- interface = " + args.interface)
print("- output    = " + args.output)
print("- mode      = " + args.mode)

if arg_mode == "interactive":
    mode_interactive(
        arg_output,      # path output (generated)
        arg_topology,    # path nn input topology
        arg_test,        # path nn test vector
        arg_ii,          # initiation interval
        arg_dtype,       # network's base data type
        arg_interface)   # network's interface
else:
    mode_default(
        arg_output,      # path output (generated)
        arg_topology,    # path nn input topology
        arg_test,        # path nn test vector
        arg_ii,          # initiation interval
        arg_dtype,       # network's base data type
        arg_interface)   # network's interface
