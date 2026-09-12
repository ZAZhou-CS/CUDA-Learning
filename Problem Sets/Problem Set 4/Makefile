NVCC = nvcc
CXX  = g++

CUDA_INCLUDEPATH = /usr/local/cuda/include

# OpenCV 4
OPENCV_CFLAGS = $(shell pkg-config --cflags opencv4)
OPENCV_LIBS   = $(shell pkg-config --libs opencv4)

# RTX 2060 = Compute Capability 7.5
NVCC_OPTS = -O3 -arch=sm_75 -Xcompiler -Wall -Xcompiler -Wextra -m64
GCC_OPTS  = -O3 -Wall -Wextra -m64


HW4: main.o student_func.o HW4.o compare.o reference_calc.o loadSaveImage.o
	$(NVCC) -o $@ $^ $(OPENCV_LIBS)
	
HW4.o: HW4.cu utils.h loadSaveImage.h
	$(NVCC) -c HW4.cu $(NVCC_OPTS) \
	-I$(CUDA_INCLUDEPATH)

main.o: main.cpp timer.h utils.h reference_calc.h compare.h
	$(CXX) -c main.cpp $(GCC_OPTS) \
	-I$(CUDA_INCLUDEPATH) $(OPENCV_CFLAGS)


student_func.o: student_func.cu utils.h
	$(NVCC) -c student_func.cu $(NVCC_OPTS) \
	-I$(CUDA_INCLUDEPATH)


compare.o: compare.cpp compare.h
	$(CXX) -c compare.cpp $(GCC_OPTS) \
	-I$(CUDA_INCLUDEPATH) $(OPENCV_CFLAGS)


reference_calc.o: reference_calc.cpp reference_calc.h
	$(CXX) -c reference_calc.cpp $(GCC_OPTS) \
	-I$(CUDA_INCLUDEPATH) $(OPENCV_CFLAGS)


loadSaveImage.o: loadSaveImage.cpp loadSaveImage.h
	$(CXX) -c loadSaveImage.cpp $(GCC_OPTS) \
	-I$(CUDA_INCLUDEPATH) $(OPENCV_CFLAGS)


clean:
	rm -f *.o HW4 *.png