#include "TACSAssembler.h"

// should we template by launch parameters here?
template <LaunchParameters lp>
void TACSAssembler::assembleJacobian_launchGPU(TacsScalar alpha, TacsScalar beta, TacsScalar gamma,
                        TacsScalar *residual, TacsScalar *A,
                        MatrixOrientation matOr = TACS_MAT_NORMAL,
                        const TacsScalar lambda = 1.0) {
    
    // assuming global nodes, connectivity and Xpts data already copied onto device (TODO)

    // launch the kernel (TODO : need to launch separate kernel for each type of elements?)
    dim3 threadsPerBlock()
    dim3 numBlocks()
    assembleJacobian_GPU<<<num_blocks, num_threads>>>(alpha, beta, gamma, residual, A, matOr, lambda);

}

__global__ void TACSAssembler::assembleJacobian_GPU(
    double time, TacsScalar alpha, TacsScalar beta, TacsScalar gamma,
    TacsScalar *residual, TacsScalar *A, MatrixOrientation matOr = TACS_MAT_NORMAL,
    const TacsScalar lambda = 1.0
) {

    // TODO : do we need launch params input as well?
    // create shared memory for each block (some # of elements per block)
    int elements_per_block = 1; // calculate from launch params
    int vars_per_node = 6;
    int nodes_per_elem = 4;
    int dof_per_elem = vars_per_node * nodes_per_elem;

    __shared__ TacsScalar block_vars[elements_per_block][dof_per_elem];
    __shared__ TacsScalar block_dvars[elements_per_block][dof_per_elem];
    __shared__ TacsScalar block_ddvars[elements_per_block][dof_per_elem];
    __shared__ TacsScalar block_Xpts[elements_per_block][3*nodes_per_elem];
    __shared__ TacsScalar block_elemRes[elements_per_block][dof_per_elem];
    __shared__ TacsScalar block_elemJac[elements_per_block][dof_per_elem * dof_per_elem];

    // TODO : copy global data into the above shared memory objects


    // loop over gauss points and derivative pass
    int32_t local_element = threadIdx.z;
    int32_t local_gauss = threadIdx.y;
    int32_t loop_bound = blockDim.x * ceil(static_cast<double>(nderivs) / static_cast<double>(blockDim.x));

    for (int32_t ideriv = threadIdx.x; ideriv < loop_bound; ideriv += blockDim.x ) {
        bool active_thread = local_element < elements_in_block && ideriv < nderivs; 
        // TODO : how to call this on the particular element type? template here?
        // maybe elements[i]->getElementJacobian..
        getElementJacobian(
            block_vars[local_element], block_dvars[local_element], block_ddvars[local_element],
            block_Xpts[local_element], block_elemRes[local_element], block_elemJac[local_element]
        );
    }

    // TODO : now send shared memory back to global memory for element residual and jacobian?
    // atomicAdd here back into global memory?

}

// TODO : this is called for each shell element?
// __device__ void getElementJacobian()