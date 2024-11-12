
#include "FakeAssembler.h"
#include <cstdio>

int main(void) {
    int nelems = 80000;
    FakeAssembler assembler(nelems);
    printf("made a FakeAssembler object\n");

    double *resid, **jacobian;
    printf("assemble Jacobian\n");
    assembler.assembleJacobian(resid, jacobian);
    return 0;
}