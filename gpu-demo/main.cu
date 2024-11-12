
#include "FakeAssembler.h"
#include <cstdio>

int main(void) {
    int nelems = 80000;
    printf("making a FakeAssembler object\n");
    FakeAssembler *assembler = new FakeAssembler(nelems);

    double *residual = nullptr;
    double **jacobian = nullptr;
    assembler->assembleJacobian(residual, jacobian);

    return 0;
}