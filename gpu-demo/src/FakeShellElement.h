#pragma once

#ifdef __CUDACC__

#include "FakeShellElement_kernels.cuh"

#endif // end of CUDACC

class FakeShellElement {
public:
    FakeShellElement() {};
    static const int VARS_PER_NODE = 6;

private:
    int x;
};

