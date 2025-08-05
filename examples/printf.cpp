#include <cstdio>
#include <cinttypes>
#include <atlc/reorder.h>

#define ATLC_VAR_EQ(fmt, var) #var "=%" fmt, (var)

int main() {
    int intvar = -1234;
    unsigned int uintvar = 1234;
    float floatvar = 1.234;
    double doublevar = 1.23456789;
    printf(ATLC_REORDER("variables: " ATLC_VAR_EQ("d", intvar), " " ATLC_VAR_EQ("u", uintvar), " " ATLC_VAR_EQ("f", floatvar), " " ATLC_VAR_EQ("lf", doublevar), "\n"));
    return 0;
}
