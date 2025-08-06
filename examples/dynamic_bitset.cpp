#include <iostream>
#include <atlc/dynamic_bitset.hpp>

int main() {
    atlc::dynamic_bitset bs(10);   // 10 bits, all 0
    bs.set(3);
    bs.set(7);
    bs.flip(7);             // now only bit 3 is 1
    std::cout << bs.to_string() << " count=" << bs.count() << std::endl;
}
