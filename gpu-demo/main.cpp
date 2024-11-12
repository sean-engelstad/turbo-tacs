
#include "FakeAssembler.h"
#include <cstdio>

int main(void) {
    int nelems = 80000;
    FakeAssembler assembler(nelems);
    printf("made a FakeAssembler object\n");
    return 0;
}