# we believe that glog is in system path
CUDA=/usr/local/cuda-6.5
CXX=g++
# uncomment these line to use the library built by the TAs
# also, you have to set LD_LIBRARY_PATH
#GOOGLE_INC=-Iexternal/include
#GOOGLE_LIB=-Lexternal/lib/linux64
CFLAGS=-Wall -c -O2 -std=c++11 -I$(CUDA)/include $(GOOGLE_INC) -Wno-narrowing -DNDEBUG
LFLAGS=-L$(CUDA)/lib64 -lOpenCL $(GOOGLE_LIB) -lglog -lgflags
EXE=3dmm15s_hw3

all: main.o pgm.o bilateral.o cl_helper.o
	$(CXX) $? $(LFLAGS) -o $(EXE)

%.o: %.cpp %.h global.h
	$(CXX) $(CFLAGS) $< -o $@

main.o: main.cpp global.h
	$(CXX) $(CFLAGS) $< -o $@

clean:
	rm *.o $(EXE)
