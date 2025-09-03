CXXFLAGS = -std=c++11 -O0 -g -I../include -Wformat=2
NVCC ?= nvcc
NVCCFLAGS = -std=c++11 -O0 -g -I../include -Xcompiler -Wformat=2
MPICXX ?= mpicxx

.Phony: all
all: printf.exe dynamic_bitset.exe group_by_hostname.exe block_reduce_sample.exe

.Phony: clean
clean:
	$(RM) printf.exe dynamic_bitset.exe group_by_hostname.exe

%.exe: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

group_by_hostname.exe: group_by_hostname.cpp
	$(MPICXX) $(CXXFLAGS) $< -o $@

block_reduce_sample.exe: block_reduce_sample.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@