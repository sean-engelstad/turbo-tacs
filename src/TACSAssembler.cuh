
// GPU kernels (outside class def)
#ifdef __CUDACC__

// debugging test kernel
template <typename T>
__global__ void myTestKernel() {
  int ithread = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
  // if (ithread < 1) {
  //   printf("threadIdx (%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z);
  // }
  printf("threadIdx (%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z);
}

// these methods must be defined in header file because CUDA
// doesn't handle templated kernel functions in .cpp file (must be explicitly defined there)

// maybe this should be templated here.. maybe not; what should be templates (the launch params?), template <class ElemType>
template <int elemPerBlock, class ElemType, class Transform, class Constitutive>
__global__ void assembleJacobian_kernel(
    double time, TacsScalar alpha, TacsScalar beta, TacsScalar gamma,
    TacsScalar *d_xpts, TacsScalar *d_vars, TacsScalar *d_dvars, TacsScalar *d_ddvars,
    int numElements, TACSElement **d_elements, int *d_elementNodeIndex, int *d_elementTacsNodes,
    TacsScalar *residual, TacsScalar *A, MatrixOrientation matOr) {

    // assumes that ElemType is the true element type here not TACSElement
    // so no dynamic polymorphism problems (because GPUs don't do dynamic determination of methods)
    
    // TODO : get these from the element class on this kernel
    const int vars_per_node = 6;
    const int nodes_per_elem = 4;
    const int dof_per_elem = vars_per_node * nodes_per_elem; // = 24 here
    const int nderivs = dof_per_elem;

    // this may be over limit of data stored on each block (seems to have run but may be spilling over data to each thread)
    // note only 48 KB or 6000 doubles of shared memory storable per block
    // TODO : later I computed the mass matrix separately.. (separate scope)
    __shared__ TacsScalar block_vars[elemPerBlock][dof_per_elem];
    __shared__ TacsScalar block_Xpts[elemPerBlock][3*nodes_per_elem];
    __shared__ TacsScalar block_res[elemPerBlock][dof_per_elem];
    __shared__ TacsScalar block_mat[elemPerBlock][dof_per_elem * dof_per_elem];
    // TODO : may be better to not copy complex objects like these to shared memory
    // although then we won't be able to get specific transform / constitutive objects for each element if we just call from classtype static method the kernel
    // discussion point here (because only limited shared memory btw and can affect speed if go over it and memory overflows into threads)
    // also can't be TACSElement* here in shared memory because abstract classes not allowed?
    __shared__ ElemType* block_elements[elemPerBlock];

    // TODO : copy global data into the above shared memory
    // want to ensure that shared data copy is distributed among blocks / threads
    // TODO : generalize this part if # elements exceeds grid size? 
    
    // distribute the copy among some # of threads
    // https://forums.developer.nvidia.com/t/copying-data-from-global-memory-to-shared-memory-by-each-thread/9498/5
    
    // temp commented out this section
    int ithread = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    int full_dim = blockDim.x * blockDim.y * blockDim.z;
    int global_ithread = full_dim * blockIdx.x + ithread;
    int ielem = ithread; // shared memory ielem

    // if (global_ithread < 1) {
    //   printf("thread %d before the global to shared memory transfer\n", ithread);
    // }

    // global memory ielem value
    int global_ielem = blockDim.x * blockIdx.x + ielem;
    if (ithread < elemPerBlock && global_ielem < numElements) {
    //   // copy one element's data over from global to shared memory (no for loop!)
    //   // int elemOffset = elementsPerBlock * blockIdx.z;
      
      int ptr = d_elementNodeIndex[global_ielem];   // says the starting node for this elem
      int len = d_elementNodeIndex[global_ielem+1] - ptr; // just says how many nodes are on this element I believe (bunch of integers)
      const int *nodes = &d_elementTacsNodes[ptr];

      // loop over each of the nodes in the element
      for (int inode = 0; inode < len; inode++) {
        int global_node = nodes[inode];
      
        // here we do the main data copying.. (do a deep copy or less than that..)
        // I guess do a deep copy for now
        for (int ivar = 0; ivar < vars_per_node; ivar++) {
          int idof = vars_per_node*inode + ivar;
          int global_idof = vars_per_node * global_node + ivar;
          // any potential speedup here?

          // block_vars[ielem][idof] = d_vars[global_idof];
          
          // // initialize res, mat to zero
          // for (int jdof = 0; jdof < dof_per_elem; jdof++) {
          //   block_mat[ielem][dof_per_elem*idof+jdof] = 0.0;
          // }

          // TODO : kinetic energy mass matrix - to be handled in separate scope later..
          // block_dvars[ielem][idof] = d_dvars[global_idof];
          // block_ddvars[ielem][idof] = d_ddvars[global_idof];

        } // end of ivar for

        // TODO : use spatial_dim arg here
        for (int idim = 0; idim < 3; idim++) {
          int ixpt = 3 * inode + idim; 
          int global_ixpt = 3 * global_node + idim;
          block_Xpts[ielem][ixpt] = d_xpts[global_ixpt];
        }

      } // end of inode for loop

      // copy element object to shared 
      TACSElement* elem = d_elements[global_ielem];   
      ElemType* quad4shell = static_cast<ElemType*>(elem);
      block_elements[ielem] = quad4shell;

    }  // end of ithread check if statement

    // // once each thread is done copying global to shared data
    __syncthreads(); 

    // if (global_ithread < 1) {
    //   printf("thread %d after shared mem transfer and before kelem kernel\n", ithread);
    // }

    // loop over gauss points and derivative pass
    // potentially should change this to Kevin's way of (x,y,z) are (ideriv, igauss, ielement)
    // but doesn't that way have 

    int32_t local_element = threadIdx.x;
    int32_t local_gauss = threadIdx.z;
    int32_t loop_bound = blockDim.x * (nderivs + blockDim.x - 1) / blockDim.x; // effectively ceil function here on nderivs

    // TODO : double check this logic on ideriv...(seems like this only runs one thread for each dim)

    // temp commented out..
    for (int32_t ideriv = threadIdx.y; ideriv < loop_bound; ideriv += blockDim.x ) {
        // if (global_ithread < 1) {
        //   printf("local_element %d < %d, ideriv %d < %d\n", local_element, elemPerBlock, ideriv, nderivs);
        // }
        bool active_thread = local_element < elemPerBlock && ideriv < nderivs;
        if (!active_thread) continue;

        // temporary res, mat data
        TacsScalar res[dof_per_elem], mat_col[dof_per_elem];

        // get constitutive, transform objects
        Transform *transform = block_elements[ielem]->template getTransform<Transform>();
        Constitutive *con = block_elements[ielem]->template getConstitutive<Constitutive>();

        // TODO : see if I get race conditions in each element or need atomic add for residual
        // also for mat I can get rows (since sym Kelem) row-based instead to prevent race conditions
        block_elements[local_element]->addStaticJacobian_kernel(
            ideriv, local_gauss,
            transform, con,
            time, alpha, beta, gamma,
            block_Xpts[local_element], block_vars[local_element],
            &res[0], &mat_col[0]
        );

        // if (global_ithread < 1) {
        //   for (int ivar = 0; ivar < 24; ivar++) {
        //     printf("Kelem[%d,%d] = %.8f\n", ivar, ideriv, mat_col[ivar]);
        //   }          
        // }
    }

    __syncthreads();

    // TODO : separate kinetic energy loop here

    if (global_ithread < 1) {
      printf("thread %d after kelem computation\n", ithread);
    }

    // now do assembly process and atomicAdds from elements in each block to global matrix assembly?

}

#endif // __CUDACC__