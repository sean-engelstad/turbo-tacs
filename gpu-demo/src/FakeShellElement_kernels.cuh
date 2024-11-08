#pragma once

#ifdef __CUDACC__

    __device__ void getElementJacobian(int num_elements, int num_nodes, double *vars, double *kelem) {
        int VARS_PER_NODE = 6;

        kelem[VARS_PER_NODE] = 1.0;
    }

#endif // end of __CUDACC__