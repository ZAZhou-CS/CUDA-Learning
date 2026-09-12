# =========================
# Compiler
# =========================
NVCC = nvcc
CXX  = g++

# =========================
# CUDA
# RTX 2060 -> Compute Capability 7.5
# =========================
CUDA_ARCH = sm_75

NVCC_FLAGS = -O3 -arch=$(CUDA_ARCH) -Xcompiler -Wall -Xcompiler -Wextra -m64
CXX_FLAGS  = -O3 -Wall -Wextra -m64

# =========================
# OpenCV
# =========================
OPENCV_INCLUDEPATH = /usr/include/opencv4
OPENCV_LIBPATH     = /usr/lib/x86_64-linux-gnu

OPENCV_LIBS = -lopencv_core \
              -lopencv_imgproc \
              -lopencv_highgui \
              -lopencv_imgcodecs

# =========================
# Target
# =========================
TARGET = HW6

OBJS = main.o \
       student_func.o \
       HW6.o \
       loadSaveImage.o \
       compare.o \
       reference_calc.o

# =========================
# Default target
# =========================
all: $(TARGET)

# =========================
# Link
# =========================
$(TARGET): $(OBJS)
	$(NVCC) -o $@ $(OBJS) \
		-L$(OPENCV_LIBPATH) $(OPENCV_LIBS)

# =========================
# main.cpp
# =========================
main.o: main.cpp timer.h utils.h
	$(CXX) -c main.cpp $(CXX_FLAGS) \
		-I$(OPENCV_INCLUDEPATH) \
		-I/usr/local/cuda/include

# =========================
# HW6.cu
# =========================
HW6.o: HW6.cu loadSaveImage.h utils.h
	$(NVCC) -c HW6.cu $(NVCC_FLAGS) \
		-I$(OPENCV_INCLUDEPATH)

# =========================
# student_func.cu
# =========================
student_func.o: student_func.cu utils.h
	$(NVCC) -c student_func.cu $(NVCC_FLAGS) \
		-I$(OPENCV_INCLUDEPATH)

# =========================
# loadSaveImage.cpp
# =========================
loadSaveImage.o: loadSaveImage.cpp loadSaveImage.h
	$(CXX) -c loadSaveImage.cpp $(CXX_FLAGS) \
		-I$(OPENCV_INCLUDEPATH) \
		-I/usr/local/cuda/include

# =========================
# compare.cpp
# =========================
compare.o: compare.cpp compare.h
	$(CXX) -c compare.cpp $(CXX_FLAGS) \
		-I$(OPENCV_INCLUDEPATH) \
		-I/usr/local/cuda/include

# =========================
# reference_calc.cpp
# =========================
reference_calc.o: reference_calc.cpp reference_calc.h
	$(CXX) -c reference_calc.cpp $(CXX_FLAGS) \
		-I$(OPENCV_INCLUDEPATH) \
		-I/usr/local/cuda/include

# =========================
# Clean
# =========================
clean:
	rm -f *.o $(TARGET)
	rm -f *.png