CXXFLAGS = -std=c++11 -O0 -g -I../include -Wformat=2

.Phony: all
all: printf.exe dynamic_bitset.exe group_by_hostname.exe

.Phony: clean
clean:
	$(RM) printf.exe dynamic_bitset.exe group_by_hostname.exe

%.exe: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

group_by_hostname.exe: group_by_hostname.cpp
	mpicxx $(CXXFLAGS) $< -o $@