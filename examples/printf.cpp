#include <cstdio>
#include <cinttypes>
#include <atlc/reorder.h>

#define STR_VAR_INT(var) #var "=%" PRId64, (int64_t)var
#define STR_VAR_UINT(var) #var "=%" PRIu64, (uint64_t)var
#define STR_VAR_FLOAT(var) #var "=%g", (double)var

int main() {
    int intvar = -1234;
    int uintvar = 1234;
    float floatvar = 1.234;
    double doublevar = 1.23456789;
    printf(ATLC_REORDER("variables: " STR_VAR_INT(intvar), " " STR_VAR_UINT(uintvar), " " STR_VAR_FLOAT(floatvar), " " STR_VAR_FLOAT(doublevar), "\n"));
    return 0;
}