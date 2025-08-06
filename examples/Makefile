CXXFLAGS = -std=c++11 -O0 -g -I../include -Wformat=2

.Phony: all
all: printf.exe dynamic_bitset.exe

.Phony: clean
clean:
	$(RM) printf.exe dynamic_bitset.exe

%.exe: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@
