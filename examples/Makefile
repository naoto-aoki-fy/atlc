CXXFLAGS = -O0 -g -I../include -Wformat=2

.Phony: all
all: printf.exe

.Phony: clean
clean:
	$(RM) printf.exe

%.exe: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@
