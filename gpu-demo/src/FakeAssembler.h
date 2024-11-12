#pragma once
#include "FakeShellElement.h"
#include <random>

#ifdef __CUDACC__
#include <assert.h>
#include "cuda_runtime.h"

#include "FakeAssembler_kernels.cuh"

#define CHECK_CUDA(func)                                                   \
  {                                                                        \
    cudaError_t status = (func);                                           \
    if (status != cudaSuccess) {                                           \
      printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__, \
             cudaGetErrorString(status), status);                          \
      exit(EXIT_FAILURE);                                                  \
    }                                                                      \
  }

#endif // end of __CUDACC__ check

class FakeAssembler {
public:
    // make random Assembler data of certain size
    FakeAssembler(int nelems_) {

        // assume nelems is multiple of 100
        num_elements = 100 * (nelems_ / 100);
        num_nodes = 101 * (num_elements / 100 + 1);

        // for shell element
        vars_per_node = FakeShellElement::VARS_PER_NODE;
        num_dof = vars_per_node * num_nodes;

        // now make the FakeShellElement for the whole mesh
        elements = new FakeShellElement[num_elements];
        for (int ielem = 0; ielem < num_elements; ielem++) {
            FakeShellElement element{};
            elements[ielem] = element;
        }

        // initialize Xpts
        Xpts = new double[3*num_nodes];
        int nx = num_elements / 100 + 1;
        int ny = 101;
        for (int ix = 0; ix < nx; ix++) {
            for (int iy = 0; iy < ny; iy++) {
                int inode = ix * ny + iy;
                for (int idim = 0; idim < 3; idim++) {
                    Xpts[3*inode + idim] = (rand() / (double)RAND_MAX);
                }
            }
        }

        // make the structured mesh of a rectangle
        int ielem = 0;
        conn = new int*[num_elements];
        for (int ix = 0; ix < nx-1; ix++) {
            for (int iy = 0; iy < ny-1; iy++) {
                int *local_conn = new int[4];
                local_conn[0] = ix * ny + iy;
                local_conn[1] = ix * ny + iy + 1;
                local_conn[2] = (ix + 1) * ny + iy + 1;
                local_conn[3] = (ix + 1) * ny + iy;
                conn[ielem] = local_conn;
                ielem++;
            }
        }

        // make random disps U in vars array
        vars = new double[num_dof];
        for (int idof = 0; idof < num_dof; idof++) {
            vars[idof] = (rand() / (double)RAND_MAX);
        }

        // allocate data on the device (may want to time this part?)
        #ifdef __CUDACC__
            setupGPU();
            printf("\tGPU has been setup\n");
        #endif
    };

    ~FakeAssembler() {
        // destroy pointers
        #ifdef __CUDACC__
            cleanupGPU();
        #endif // end of CUDACC
    }

    void assembleJacobian(double *residual_, double **jacobian_) {

        #ifdef __CUDACC__
            assembleJacobian_GPU(residual, jacobian);

        #else 
            assembleJacobian_CPU(residual, jacobian);

        #endif

    }

private:

    void assembleJacobian_CPU(double *residual_, double **jacobian_) {
        printf("assemble on the CPU\n");
    }

    #ifdef __CUDACC__

        void assembleJacobian_GPU(double *residual_, double **jacobian_) {
            printf("assemble on the GPU\n");
            
            // launch the kernel (figure out how many threads and  blocks we need)
            // num threads per block is 1024
            int threads_per_block = 1024;
            int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;
            printf("num_blocks = %d\n", num_blocks);

            double **d_kelems = new double*[num_elements];
            for (int ielem = 0; ielem < num_elements; ielem++) {
                d_kelems[ielem] = new double[24*24];
            }

            // setup shared memory for each element on each block
            jacobian_kernel<6><<<num_blocks, threads_per_block>>>(num_elements, num_nodes, d_vars, d_elements, d_kelems);
            

            // TODO : parallelize over gauss pt & derivative pass next
        }

        void setupGPU() {
            // copy data from host to device

            // TODO : remove void**
            // if (cudaMallocManaged(&d_num_elements, sizeof(int)) != cudaSuccess) {
            //     fprintf(stderr, "failed to allocate num_elements int");
            // }

            size_t elemssize = num_elements * sizeof(FakeShellElement);
            CHECK_CUDA(cudaMalloc((void**)&d_elements, elemssize));
            CHECK_CUDA(cudaMemcpy(d_elements, elements, elemssize, cudaMemcpyHostToDevice));

            size_t Xptssize = 3 * num_nodes * sizeof(double);
            CHECK_CUDA(cudaMalloc((void**)&d_Xpts, Xptssize));
            CHECK_CUDA(cudaMemcpy(d_Xpts, Xpts, Xptssize, cudaMemcpyHostToDevice));

            size_t varssize = num_dof * sizeof(double);
            CHECK_CUDA(cudaMalloc((void**)&d_vars, varssize));
            CHECK_CUDA(cudaMemcpy(d_vars, vars, varssize, cudaMemcpyHostToDevice));

            size_t connsize = 4 * num_elements * sizeof(int);
            CHECK_CUDA(cudaMalloc((void**)&d_conn, connsize));
            CHECK_CUDA(cudaMemcpy(d_conn, conn, connsize, cudaMemcpyHostToDevice));

        }

        void cleanupGPU() {
            // copy data from host to device

            CHECK_CUDA(cudaFree(d_elements));
            CHECK_CUDA(cudaFree(d_Xpts));
            CHECK_CUDA(cudaFree(d_vars));
            CHECK_CUDA(cudaFree(d_conn));

        }

    #endif // end of CUDACC check
    

    // CPU data
    int num_elements, num_nodes, vars_per_node, num_dof;
    FakeShellElement *elements;
    double *Xpts, *vars;
    int **conn;

    double *residual, **jacobian;

    // GPU data
    FakeShellElement *d_elements;
    double *d_Xpts, *d_vars;
    int **d_conn;
};