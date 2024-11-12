#pragma once
#include "FakeShellElement_kernels.cuh"

// kernels can't be included in a class, so in separate file

template <int vars_per_node>
__global__ void jacobian_kernel(int num_elements, int num_nodes, double *vars, FakeShellElement *global_elems, double **all_kelems) {

    // get local element ID
    //  blockIdx.x * blockDim.x + 
    int local_element = threadIdx.x;
    bool active_thread = local_element < num_elements;

    if (!active_thread) { return; }

    // setup shared memory for each block (which elements on the block)
    const int elements_per_block = 1024;
    const int nodes_per_elem = 4;
    const int ndof = vars_per_node * nodes_per_elem;

    // double l_vars[elements_per_block][vars_per_node];
    // FakeShellElement l_elems[elements_per_block];
    // double l_kelems[elements_per_block][vars_per_node*vars_per_node];

    __shared__ double l_vars[elements_per_block][ndof];
    __shared__ FakeShellElement l_elems[elements_per_block];
    __shared__ double l_kelems[elements_per_block][ndof*ndof];

    // copy device data onto shared data
    int block_elem_offset = blockIdx.x * blockDim.x;
    for (int ielem = 0; ielem < elements_per_block; ielem++) {

        int global_elem_idx = block_elem_offset + local_element;
        l_elems[ielem] = global_elems[global_elem_idx];

        for (int idof = 0; idof < ndof; idof++) {
            l_vars[ielem][idof] = vars[ndof * global_elem_idx + idof];
        }

        for (int idofsq = 0; idofsq < ndof*ndof; idofsq++) {
            l_kelems[ielem][idofsq] = 0.0;
        }
    }

    // // call device function
    getElementJacobian(num_elements, num_nodes, l_vars[local_element], l_kelems[local_element]);

}