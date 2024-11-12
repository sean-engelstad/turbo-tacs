#pragma once

#ifdef __CUDACC__

#include "FakeShellElement_kernels.cuh"

#endif // end of CUDACC

#include <chrono>
#include <cmath>
#include <iostream>
using namespace std;
using namespace std::chrono;


class FakeShellElement {
public:
    FakeShellElement() {};
    static const int VARS_PER_NODE = 6;

    void getElementJacobian_CPU(int num_elements, int num_nodes, double *vars, double *kelem) {
        int NDOF = 24;
        const int NLOOP = 1000;
        // auto start = high_resolution_clock::now();

        // fake kelem calculations to give representative runtime
        double x = 0.8;
        for (int i = 0; i < NLOOP; i++) {
            x = sin(x) * cos(x) + sqrtf(x) + log(x + 1.0f);
        }
        // printf("x = %.8f\n", x);

        // now compute a fake kelem
        for (int idof = 0; idof < 24; idof++) {
            for (int jdof = 0; jdof < 24; jdof++) {
                kelem[NDOF * idof + jdof] = 1.0; // could be random val here, who cares
            }
        }

        // auto stop = high_resolution_clock::now();
        // auto duration = duration_cast<microseconds>(stop - start);
        // cout << "duration, " << duration.count() << std::endl;

    }

private:
    int x;
};

