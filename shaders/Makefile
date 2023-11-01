
CL_FLAGS = -cl-fast-relaxed-math -cl-single-precision-constant 

FLAGS = $(CL_FLAGS)

all:
	clspv --cl-std=CL2.0 --spv-version=1.3 --inline-entry-points fun.cl -o fun.spv
