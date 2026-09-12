NVCC = nvcc

CUDA_INCLUDEPATH = /usr/local/cuda/include

# OpenCV 4
OPENCV_CFLAGS = $(shell pkg-config --cflags opencv4)
OPENCV_LIBS   = $(shell pkg-config --libs opencv4)

NVCC_OPTS = -O3 -arch=sm_75 -Xcompiler -Wall -Xcompiler -Wextra -m64
GCC_OPTS  = -O3 -Wall -Wextra -m64


student: main.o HW3.o student_func.o compare.o reference_calc.o loadSaveImage.o Makefile
	$(NVCC) -o HW3 main.o HW3.o student_func.o compare.o reference_calc.o loadSaveImage.o \
	$(OPENCV_LIBS) $(NVCC_OPTS)

HW3.o: HW3.cu utils.h loadSaveImage.h
	$(NVCC) -c HW3.cu $(NVCC_OPTS)
	
main.o: main.cpp timer.h utils.h reference_calc.h compare.h
	g++ -c main.cpp $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)


loadSaveImage.o: loadSaveImage.cpp loadSaveImage.h
	g++ -c loadSaveImage.cpp $(OPENCV_CFLAGS) $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)


compare.o: compare.cpp compare.h
	g++ -c compare.cpp $(OPENCV_CFLAGS) $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)


reference_calc.o: reference_calc.cpp reference_calc.h
	g++ -c reference_calc.cpp $(OPENCV_CFLAGS) $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)


student_func.o: student_func.cu utils.h
	$(NVCC) -c student_func.cu $(NVCC_OPTS)


clean:
	rm -f *.o HW3
	find . -type f -name '*.exr' | grep -v memorial | xargs rm -f