NVCC = nvcc

CUDA_INCLUDEPATH = /usr/local/cuda/include

# OpenCV 4
OPENCV_CFLAGS = $(shell pkg-config --cflags opencv4)
OPENCV_LIBS   = $(shell pkg-config --libs opencv4)

# CUDA 11.8 + RTX 2060
NVCC_OPTS = -O3 -arch=sm_75 -Xcompiler -Wall -Xcompiler -Wextra -m64
GCC_OPTS  = -O3 -Wall -Wextra -m64


student: main.o student_func.o compare.o reference_calc.o Makefile
	$(NVCC) -o HW1 main.o student_func.o compare.o reference_calc.o \
	$(OPENCV_LIBS) $(NVCC_OPTS)


main.o: main.cpp timer.h utils.h reference_calc.cpp compare.cpp HW1.cpp
	g++ -c main.cpp $(GCC_OPTS) \
	-I$(CUDA_INCLUDEPATH) $(OPENCV_CFLAGS)


student_func.o: student_func.cu utils.h
	nvcc -c student_func.cu $(NVCC_OPTS) \
	-I$(CUDA_INCLUDEPATH)


compare.o: compare.cpp compare.h
	g++ -c compare.cpp $(GCC_OPTS) \
	-I$(CUDA_INCLUDEPATH) $(OPENCV_CFLAGS)


reference_calc.o: reference_calc.cpp reference_calc.h
	g++ -c reference_calc.cpp $(GCC_OPTS) \
	-I$(CUDA_INCLUDEPATH) $(OPENCV_CFLAGS)


clean:
	rm -f *.o HW1 *.png