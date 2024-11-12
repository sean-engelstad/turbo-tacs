#pragma once

#ifdef __CUDACC__

    __device__ void getElementJacobian(int num_elements, int num_nodes, double *vars, double *kelem) {
        int NDOF = 24;
        const int NLOOP = 100000;

        // fake kelem calculations to give representative runtime
        double x = 1.234;
        for (int i = 0; i < NLOOP; i++) {
            x = sin(x) * cos(x) + sqrtf(x) + log(x + 1.0f);
        }

        // now compute a fake kelem
        for (int idof = 0; idof < 24; idof++) {
            for (int jdof = 0; jdof < 24; jdof++) {
                kelem[NDOF * idof + jdof] = 1.0; // could be random val here, who cares
            }
        }

    }

#endif // end of __CUDACC__