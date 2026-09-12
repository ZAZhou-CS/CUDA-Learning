NVCC = nvcc
CXX  = g++

CUDA_ARCH = sm_75

NVCC_OPTS = -O3 -arch=$(CUDA_ARCH) -Xcompiler -Wall -Xcompiler -Wextra -m64
CXX_OPTS  = -O3 -Wall -Wextra -m64

TARGET = HW5

all: $(TARGET)

$(TARGET): main.o reference_calc.o student.o
	$(NVCC) -o $(TARGET) main.o reference_calc.o student.o $(NVCC_OPTS)

main.o: main.cu
	$(NVCC) -c main.cu $(NVCC_OPTS)

student.o: student.cu
	$(NVCC) -c student.cu $(NVCC_OPTS)

reference_calc.o: reference_calc.cpp reference_calc.h
	$(CXX) -c reference_calc.cpp $(CXX_OPTS)

clean:
	rm -f *.o $(TARGET)
