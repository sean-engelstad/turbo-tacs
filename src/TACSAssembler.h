/*
  This file is part of TACS: The Toolkit for the Analysis of Composite
  Structures, a parallel finite-element code for structural and
  multidisciplinary design optimization.

  Copyright (C) 2010 University of Toronto
  Copyright (C) 2012 University of Michigan
  Copyright (C) 2014 Georgia Tech Research Corporation
  Additional copyright (C) 2010 Graeme J. Kennedy and Joaquim
  R.R.A. Martins All rights reserved.

  TACS is licensed under the Apache License, Version 2.0 (the
  "License"); you may not use this software except in compliance with
  the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0
*/

// #ifndef TACS_ASSEMBLER_H
// #define TACS_ASSEMBLER_H
#pragma once

/*
  TACSAssembler assembles the residuals and matrices required for
  analysis and sensitivity analysis.
*/

class TACSAssembler;

// Basic analysis classes
#include "TACSElement.h"
#include "TACSObject.h"

// Linear algebra classes
#include "TACSBVecDistribute.h"
#include "TACSParallelMat.h"
#include "TACSSchurMat.h"
#include "TACSSerialPivotMat.h"

// GPU include
#ifdef __CUDACC__

  #include <cuda_runtime.h>
  #include <stdio.h>
  #include <sys/time.h>
  #define CHECK_CUDA(call)                                              \
    {                                                                 \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",         \
                    __func__, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    }

  #define cudaCheckError() {                                                    \
      cudaError_t e=cudaGetLastError();                                         \
      if(e!=cudaSuccess) {                                                     \
          printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
          exit(EXIT_FAILURE);                                                  \
      }                                                                        \
  }


  #include "TACSAssembler.cuh"

  // need to import all element types (in this case just shell elements)
  // #include "elements/shell/TACSShellElementDefs.h"

#endif // __CUDACC__

/*
  TACSAssembler

  This is the main class required for structural analysis using TACS.
  Basic operations required for analysis and design optimization
  should be performed using this class, rather than with element-level
  functions. This class is used to assemble residuals and Jacobians
  required for analysis. The class also implements the routines
  necessary to compute the adjoint. These operations include parallel
  function evaluation, derivative evaluation and the implementation of
  terms required for the adjoint method.

  TACSAssembler can be instantiated and initialized directly through
  API calls, or through other TACS objects which perform the
  initialization process.  Once the TACSAssembler object is
  initialized, however, subsequent calls to the code work regardless
  of how the object was created. In addition, once TACSAssembler is
  created, the parallelism is transparent. All analysis calls are
  collective on the tacs_comm communicator.
*/
class TACSAssembler : public TACSObject {
 public:
  // There are always 3 coordinates (even for 2D problems)
  static const int TACS_SPATIAL_DIM = 3;

  enum OrderingType {
    NATURAL_ORDER,   // Natural ordering
    RCM_ORDER,       // Reverse Cuthill Mackee ordering
    AMD_ORDER,       // Approximate minimum degree
    ND_ORDER,        // Nested disection
    TACS_AMD_ORDER,  // Interface variables ordered last
    MULTICOLOR_ORDER
  };  // Multicolor via greedy algorithm
  enum MatrixOrderingType {
    ADDITIVE_SCHWARZ,
    APPROXIMATE_SCHUR,
    DIRECT_SCHUR,
    GAUSS_SEIDEL
  };

  // Create the TACSAssembler object in parallel
  // -------------------------------------------
  TACSAssembler(MPI_Comm _tacs_comm, int _varsPerNode, int _numOwnedNodes,
                int _numElements, int _numDependentNodes = 0);
  ~TACSAssembler();

  // Set the connectivity in TACS
  // ----------------------------
  int setElementConnectivity(const int *ptr, const int *conn);
  void getElementConnectivity(const int **ptr, const int **conn);
  int setElements(TACSElement **_elements);
  int setDependentNodes(const int *_depNodeIndex, const int *_depNodeToTacs,
                        const double *_depNodeWeights);

  void getAverageStresses(ElementType elem_type, TacsScalar *avgStresses, int compNum);
  void setComplexStepGmatrix(bool flag);

  // Set additional information about the design vector
  // --------------------------------------------------
  void setDesignNodeMap(int _designVarsPerNode,
                        TACSNodeMap *_designVarMap = NULL);
  int setDesignDependentNodes(int numDepDesignVars, const int *_depNodePtr,
                              const int *_depNodes,
                              const double *_depNodeWeights);

  // Associate a Dirichlet boundary condition with the given variables
  // -----------------------------------------------------------------
  void addBCs(int nnodes, const int *nodes, int nbcs = -1,
              const int *vars = NULL, const TacsScalar *vals = NULL);
  void addInitBCs(int nnodes, const int *nodes, int nbcs = -1,
                  const int *vars = NULL, const TacsScalar *vals = NULL);

  // Set Dirichlet BC values at nodes where BCs are imposed
  // ------------------------------------------------------
  void setBCValuesFromVec(TACSBVec *vec);

  // Reorder the unknowns according to the specified reordering
  // ----------------------------------------------------------
  void computeReordering(OrderingType order_type, MatrixOrderingType mat_type);

  // Functions for retrieving the reordering
  // ---------------------------------------
  int isReordered();
  void getReordering(int *oldToNew);
  void reorderVec(TACSBVec *vec);
  void reorderNodes(int num_nodes, int *nodes);

  // Initialize the mesh
  // -------------------
  int initialize();

  // Return important information about the TACSAssembler object
  // -----------------------------------------------------------
  MPI_Comm getMPIComm();
  TACSThreadInfo *getThreadInfo();
  int getVarsPerNode();
  int getDesignVarsPerNode();
  int getNumNodes();
  int getNumDependentNodes();
  int getNumOwnedNodes();
  int getNumElements();
  TACSNodeMap *getNodeMap();
  TACSNodeMap *getDesignNodeMap();
  TACSBcMap *getBcMap();
  TACSBcMap *getInitBcMap();
  TACSBVecDistribute *getBVecDistribute();
  TACSBVecDepNodes *getBVecDepNodes();

  // Get the maximum sizes
  // ---------------------
  int getMaxElementNodes();
  int getMaxElementVariables();
  int getMaxElementDesignVars();

  // Set the nodes in TACS
  // ---------------------
  TACSBVec *createNodeVec();
  void setNodes(TACSBVec *X);
  void getNodes(TACSBVec *X);
  void getNodes(TACSBVec **X);

  // Check for the elements for non-positive determinants
  // ----------------------------------------------------
  void checkElementDeterminants();

  // Set/get the simulation time
  // ---------------------------
  void setSimulationTime(double _time);
  double getSimulationTime();

  // Create vectors
  // --------------
  TACSBVec *createVec();

  // Shortcut to apply boundary conditions
  void applyBCs(TACSVec *vec);
  void applyBCs(TACSMat *mat);
  void applyTransposeBCs(TACSMat *mat);

  // Set the Dirichlet boundary conditions to the state vector
  void setBCs(TACSVec *vec);

  // Methods for manipulating internal variable values
  // -------------------------------------------------
  void zeroVariables();
  void zeroDotVariables();
  void zeroDDotVariables();

  // Methods for setting/getting variables
  // -------------------------------------
  void setVariables(TACSBVec *q, TACSBVec *qdot = NULL, TACSBVec *qddot = NULL);
  void getVariables(TACSBVec *q, TACSBVec *qdot = NULL, TACSBVec *qddot = NULL);
  void getVariables(TACSBVec **q, TACSBVec **qdot = NULL,
                    TACSBVec **qddot = NULL);
  void copyVariables(TACSBVec *q, TACSBVec *qdot = NULL,
                     TACSBVec *qddot = NULL);

  // Create the matrices that can be used for analysis
  // -------------------------------------------------
  TACSParallelMat *createMat();
  TACSSchurMat *createSchurMat(OrderingType order_type = TACS_AMD_ORDER);
  TACSSerialPivotMat *createSerialMat();

  // Retrieve or set the initial conditions for the simulation
  // --------------------------------------------------
  void getInitConditions(TACSBVec *vars, TACSBVec *dvars, TACSBVec *ddvars);
  void setInitConditions(TACSBVec *vars, TACSBVec *dvars, TACSBVec *ddvars);

  // Evaluate the kinetic and potential energy
  // -----------------------------------------
  void evalEnergies(TacsScalar *Te, TacsScalar *Pe);

  // Residual and Jacobian assembly
  // ------------------------------
  void assembleRes(TACSBVec *residual, const TacsScalar lambda = 1.0);

  // template <class ElemType>
  // void assembleJacobian(TacsScalar alpha, TacsScalar beta, TacsScalar gamma,
  //                       TACSBVec *residual, TACSMat *A,
  //                       MatrixOrientation matOr = TACS_MAT_NORMAL,
  //                       const TacsScalar lambda = 1.0);
  /**
    Assemble the Jacobian matrix

    This function assembles the global Jacobian matrix and
    residual. This Jacobian includes the contributions from all
    elements. The Dirichlet boundary conditions are applied to the
    matrix by zeroing the rows of the matrix associated with a boundary
    condition, and setting the diagonal to unity. The matrix assembly
    also performs any communication required so that the matrix can be
    used immediately after assembly.

    @param alpha Coefficient for the variables
    @param beta Coefficient for the time-derivative terms
    @param gamma Coefficientfor the second time derivative term
    @param residual The residual of the governing equations
    @param A The Jacobian matrix
    @param matOr the matrix orientation NORMAL or TRANSPOSE
    @param lambda Scaling factor for the aux element contributions, by default 1
  */
  template <class ElemType, class Transform, class Constitutive>
  void assembleJacobian(TacsScalar alpha, TacsScalar beta, TacsScalar gamma, 
                        TACSBVec *residual, TACSMat *A, 
                        MatrixOrientation matOr = TACS_MAT_NORMAL,
                        const TacsScalar lambda = 1.0) {

    #ifdef __CUDACC__
      // GPU version of assembleJacobian

      // printf("assemble jacobian in GPU\n");
      
      // get serial versions of residual, matrix out of here on the host
      TacsScalar *h_residual, *h_matrix;

      // TODO : for each different type of element
      // get the unique class / typedef with a new method getTacsElementType or something
      // then we need to call the static __device__ addJacobian_kernel method with it
      // using T = Quad4Shell; // not doing this for now

      // for each elemType (iterate over list of typedefs and list of the element objects) 
      // , if different element types (for now assume all same type)

      // TODO : make this more formal to compute optimal launch parameters for this element
      // figure out how many threads and blocks to launch for this element

      // product of dim3 block can't go over 1024
      const int elemPerBlock = 1024 / 24 / 4; // TODO : figure out what the best number is here
      dim3 block(elemPerBlock, 24,4); // threads per block
      dim3 grid((numElements + block.x-1) / block.x); // just 1D grid for now with remaining elements
      // printf("launching kernel with grid <<<%d>>> and block <<<%d,%d,%d>>>\n", grid.x, block.x, block.y, block.z);

      // TODO : later instead of templating by ElemType (import all element types and organize elements
      // into element type groups then call template ElemType for each group of elements I guess).

      // assume all device data such as xpts, vars, connectivity is already updated from design change, etc.
      // launch a kernel after passing in the device data already on GPU
      // TODO : should this be templated by launch params instead?
      // assembleJacobian_kernel<elemPerBlock, ElemType> <<<grid, block>>>(
      //   time, alpha, beta, gamma, 
      //   d_xptVec, d_varsVec, d_dvarsVec, d_ddvarsVec, 
      //   numElements, d_elements, d_elementNodeIndex, d_elementTacsNodes,
      //   h_residual, h_matrix, matOr
      // );

      assembleJacobian_kernel<elemPerBlock, ElemType, Transform, Constitutive> <<<1, 1>>>(
        time, alpha, beta, gamma, 
        d_xptVec, d_varsVec, d_dvarsVec, d_ddvarsVec, 
        numElements, d_elements, d_elementNodeIndex, d_elementTacsNodes,
        h_residual, h_matrix, matOr
      );

      // myTestKernel<TacsScalar> <<<1, 64>>> ();
      // myTestKernel<TacsScalar> <<<1, block>>> ();

      // printf("done launching the kernel\n");
      // check CUDA necessary to catch launch kernel failures
      CHECK_CUDA(cudaDeviceSynchronize());
      // cudaDeviceReset();

      // send data back to BVec's ?
      // apply BCs?

    #else
      // CPU code

      // Zero the residual and the matrix
      if (residual) {
        residual->zeroEntries();
      }
      A->zeroEntries();

      // Run the p-threaded version of the assembly code
      if (thread_info->getNumThreads() > 1) {
        // Set the number of completed elements to zero
        numCompletedElements = 0;
        tacsPInfo->assembler = this;
        tacsPInfo->res = residual;
        tacsPInfo->mat = A;
        tacsPInfo->alpha = alpha;
        tacsPInfo->beta = beta;
        tacsPInfo->gamma = gamma;
        tacsPInfo->lambda = lambda;
        tacsPInfo->matOr = matOr;

        // Create the joinable attribute
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

        for (int k = 0; k < thread_info->getNumThreads(); k++) {
          pthread_create(&threads[k], &attr, TACSAssembler::assembleJacobian_thread,
                        (void *)tacsPInfo);
        }

        // Join all the threads
        for (int k = 0; k < thread_info->getNumThreads(); k++) {
          pthread_join(threads[k], NULL);
        }

        // Destroy the attribute
        pthread_attr_destroy(&attr);
      } else {
        // Retrieve pointers to temporary storage
        TacsScalar *vars, *dvars, *ddvars, *elemRes, *elemXpts;
        TacsScalar *elemWeights, *elemMat;
        getDataPointers(elementData, &vars, &dvars, &ddvars, &elemRes, &elemXpts,
                        NULL, &elemWeights, &elemMat);

        for (int i = 0; i < numElements; i++) {
          int ptr = elementNodeIndex[i];
          int len = elementNodeIndex[i + 1] - ptr;
          const int *nodes = &elementTacsNodes[ptr];
          xptVec->getValues(len, nodes, elemXpts);
          varsVec->getValues(len, nodes, vars);
          dvarsVec->getValues(len, nodes, dvars);
          ddvarsVec->getValues(len, nodes, ddvars);

          // Get the number of variables from the element
          int nvars = elements[i]->getNumVariables();

          // debug set vars to nonzero
          // for (int i1 = 0; i1 < 24; i1++) {
          //   vars[i1] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
          //   // printf("vars[%d] = %.8e\n", i1, vars[i1]);
          // }

          // Compute and add the contributions to the Jacobian
          memset(elemRes, 0, nvars * sizeof(TacsScalar));
          memset(elemMat, 0, nvars * nvars * sizeof(TacsScalar));
          elements[i]->addJacobian(i, time, alpha, beta, gamma, elemXpts, vars,
                                  dvars, ddvars, elemRes, elemMat);

          // end debug
          // exit(0);

          if (residual) {
            residual->setValues(len, nodes, elemRes, TACS_ADD_VALUES);
          }
          addMatValues(A, i, elemMat, elementIData, elemWeights, matOr);
        }
      }

      // Do any matrix and residual assembly if required
      A->beginAssembly();
      if (residual) {
        residual->beginSetValues(TACS_ADD_VALUES);
      }

      A->endAssembly();
      if (residual) {
        residual->endSetValues(TACS_ADD_VALUES);
      }

      // Apply the boundary conditions
      if (residual) {
        residual->applyBCs(bcMap, varsVec);
      }

      // Apply the appropriate boundary conditions
      A->applyBCs(bcMap);

    #endif
  }

  void assembleMatType(ElementMatrixType matType, TACSMat *A,
                       MatrixOrientation matOr = TACS_MAT_NORMAL,
                       const TacsScalar lambda = 1.0);
  void assembleMatCombo(ElementMatrixType matTypes[], TacsScalar scale[],
                        int nmats, TACSMat *A,
                        MatrixOrientation matOr = TACS_MAT_NORMAL,
                        const TacsScalar lambda = 1.0);
  void addJacobianVecProduct(TacsScalar scale, TacsScalar alpha,
                             TacsScalar beta, TacsScalar gamma, TACSBVec *x,
                             TACSBVec *y,
                             MatrixOrientation matOr = TACS_MAT_NORMAL,
                             const TacsScalar lambda = 1.0);

  // Design variable handling
  // ------------------------
  TACSBVec *createDesignVec();
  void getDesignVars(TACSBVec *dvs);
  void setDesignVars(TACSBVec *dvs);
  void getDesignVarRange(TACSBVec *lb, TACSBVec *ub);

  // Return elements and node numbers
  // --------------------------------
  TACSElement **getElements();
  TACSElement *getElement(int elem, TacsScalar *Xpts = NULL,
                          TacsScalar *vars = NULL, TacsScalar *dvars = NULL,
                          TacsScalar *ddvars = NULL);
  TACSElement *getElement(int elem, int *len, const int **nodes);

  // Set the number of threads to work with
  // --------------------------------------
  void setNumThreads(int t);

  // Get information about the output files; For use by TACSToFH5
  // ------------------------------------------------------------
  int getNumComponents();
  void getElementOutputData(ElementType elem_type, int write_flag, int *len,
                            int *nvals, TacsScalar **data);

  // Functions for ordering the variables
  // ------------------------------------
  int getLocalNodeNum(int node);
  int getGlobalNodeNum(int node);
  void computeLocalNodeToNodeCSR(int **_rowp, int **_cols, int nodiag = 0);
  void computeNodeToElementCSR(int **_nodeElem, int **_nodeElemIndex);

  // public GPU routines
  #ifdef __CUDACC__

    // put TACS global FEM data on the GPU device (for serial currently)
    void allocateDeviceData();

  #endif // __CUDACC__

 private:
  // Get the number of design variable numbers
  // -----------------------------------------
  int getNumDesignVars();

  // Get pointers to the start-locations within the data array
  // ---------------------------------------------------------
  void getDataPointers(TacsScalar *data, TacsScalar **v1, TacsScalar **v2,
                       TacsScalar **v3, TacsScalar **v4, TacsScalar **x1,
                       TacsScalar **x2, TacsScalar **weights, TacsScalar **mat);

  // Functions that are used to perform reordering
  // ---------------------------------------------
  int computeExtNodes();
  int computeCouplingNodes(int **_couplingNodes, int **_extPtr = NULL,
                           int **_extCount = NULL, int **_recvPtr = NULL,
                           int **_recvCount = NULL, int **_recvNodes = NULL);
  int computeCouplingElements(int **_celems);

  // Functions for ordering the variables
  // ------------------------------------
  void computeLocalNodeToNodeCSR(int **_rowp, int **_cols, int nrnodes,
                                 const int *rnodes, int nodiag);

  // Compute the connectivity of the multiplier information
  void computeMultiplierConn(int *_num_multipliers, int **_multipliers,
                             int **_indep_ptr, int **_indep_nodes);

  // Compute the reordering for a local matrix
  // -----------------------------------------
  void computeMatReordering(OrderingType order_type, int nvars, int *rowp,
                            int *cols, int *perm, int *new_vars);

  // Scatter the boundary conditions on external nodes
  void scatterExternalBCs(TACSBcMap *bcs);

  // Add values into the matrix
  inline void addMatValues(TACSMat *A, const int elemNum, const TacsScalar *mat,
                           int *item, TacsScalar *temp,
                           MatrixOrientation matOr);

  TACSNodeMap *nodeMap;               // Variable ownership map
  TACSBcMap *bcMap;                   // Boundary condition data
  TACSBcMap *bcInitMap;               // Initial boundary condition data
  TACSBVecDistribute *extDist;        // Distribute the vector
  TACSBVecIndices *extDistIndices;    // The tacsVarNum indices
  TACSBVecDepNodes *depNodes;         // Dependent variable information
  TACSNodeMap *designNodeMap;         // Distribution of design variables
  TACSBVecDistribute *designExtDist;  // Distribute the design variables
  TACSBVecDepNodes *designDepNodes;   // Dependent design variable information

  // Reordering information
  TACSBVecIndices *newNodeIndices;

  // Additional information information for the TACSParallel class
  TACSBVecIndices *parMatIndices;

  // Additional ordering information for the TACSSchurMat class
  // These are created once - all subsequent calls use this data.
  TACSBVecIndices *schurBIndices, *schurCIndices;
  TACSBVecDistribute *schurBMap, *schurCMap;

  // The global simulation time variable
  double time;

  // variables/elements have been initialized
  int meshInitializedFlag;

  // Information about the variables and elements
  int varsPerNode;         // number of variables per node
  int numElements;         // number of elements
  int numNodes;            // number of nodes referenced by this process
  int numOwnedNodes;       // number of nodes owned by this processor
  int numExtNodes;         // number of extneral nodes
  int numDependentNodes;   // number of dependent nodes
  int numMultiplierNodes;  // number of multiplier nodes/elements
  int designVarsPerNode;   // number of design variables at each design "node"

  // Maximum element information
  int maxElementDesignVars;  // maximum number of design variable
  int maxElementNodes;       // maximum number of ind. and dep. element nodes
  int maxElementSize;        // maximum number of variables for any element
  int maxElementIndepNodes;  // maximum number of independent nodes

  // Node numbers that are referred to from this processor
  int *tacsExtNodeNums;  // node numbers associated with TACS
  int extNodeOffset;     // Offset into the external nodes

  // Variables that define the CSR data structure to
  // store the element -> node information
  int *elementNodeIndex;
  int *elementTacsNodes;

  // The local list of elements
  TACSElement **elements;

  // The variables, velocities and accelerations
  TACSBVec *varsVec, *dvarsVec, *ddvarsVec;

  // Memory for the node locations
  TACSBVec *xptVec;

  // Memory for the element residuals and variables
  TacsScalar *elementData;  // Space for element residuals/matrices
  int *elementIData;        // Space for element index data

  // Memory for the design variables and inddex data
  TacsScalar *elementSensData;
  int *elementSensIData;

  // Memory for the initial condition vectors
  TACSBVec *vars0, *dvars0, *ddvars0;

  // The data required to perform parallel operations
  // MPI info
  int mpiRank, mpiSize;
  MPI_Comm tacs_comm;

  // The static member functions that are used to p-thread TACSAssembler
  // operations... These are the most time-consuming operations.
  static void schedPthreadJob(TACSAssembler *tacs, int *index, int total_size);
  static void *assembleRes_thread(void *t);
  static void *assembleJacobian_thread(void *t);
  static void *assembleMatType_thread(void *t);

  // GPU data
  #ifdef __CUDACC__

    // GPU storage data on device (private)
    TACSElement **d_elements = nullptr;
    TacsScalar *d_varsVec = nullptr, *d_dvarsVec = nullptr, *d_ddvarsVec = nullptr;
    TacsScalar *d_xptVec = nullptr;
    int *d_elementNodeIndex = nullptr, *d_elementTacsNodes = nullptr;

  #endif // __CUDACC__

  // Class to store specific information about the threaded
  // operations to perform. Note that assembly operations are
  // relatively easy, while design-variable dependent info is
  // much more challenging!
  class TACSAssemblerPthreadInfo {
   public:
    TACSAssemblerPthreadInfo() {
      assembler = NULL;
      res = NULL;
      mat = NULL;
      alpha = beta = gamma = 0.0;
      lambda = 1.0;
      matType = TACS_STIFFNESS_MATRIX;
      matOr = TACS_MAT_NORMAL;
      numDesignVars = 0;
    }

    // The data required to perform most of the matrix
    // assembly.
    TACSAssembler *assembler;

    // Information for residual assembly
    TACSBVec *res;

    // Information for matrix assembly
    TACSMat *mat;
    TacsScalar alpha, beta, gamma, lambda;
    ElementMatrixType matType;
    MatrixOrientation matOr;

    int numDesignVars;
  } * tacsPInfo;

  // The pthread data required to pthread tacs operations
  int numCompletedElements;     // Keep track of how much work has been done
  TACSThreadInfo *thread_info;  // The pthread object

  // The thread objects
  pthread_t threads[TACSThreadInfo::TACS_MAX_NUM_THREADS];
  pthread_mutex_t tacs_mutex;  // The mutex for coordinating assembly ops.

  // The name of the TACSAssembler object
  static const char *tacsName;
};

/*
  Add the values of the element matrix to the provided TACSMat.

  This code takes into account dependent-nodes (when they exist) by
  adding the inner product of the dependent weights with the element
  matrix.  Note that the integer and scalar temporary storage should
  be allocated once for all elements for efficiency purposes. The
  maximum weight length can be determined by finding the maximum
  number of nodes + max total number of dependent->local nodes in any
  local element.

  input:
  elemNum:    the element number
  mat:        the corresponding element matrix
  itemp:      temporary integer storage len(itemp) >= nnodes+1 + len(vars)
  temp:       temporary scalar storage len(temp) >= len(weights)

  input/output:
  A:          the matrix to which the element-matrix is added
*/
inline void TACSAssembler::addMatValues(TACSMat *A, const int elemNum,
                                        const TacsScalar *mat, int *itemp,
                                        TacsScalar *temp,
                                        MatrixOrientation matOr) {
  int start = elementNodeIndex[elemNum];
  int end = elementNodeIndex[elemNum + 1];
  int nnodes = end - start;
  int nvars = varsPerNode * nnodes;

  // Add the element values to the matrix
  const int *nodeNums = &elementTacsNodes[start];

  if (matOr == TACS_MAT_NORMAL && numDependentNodes == 0) {
    // If we have no dependent nodes, then we don't need to do
    // anything extra here
    A->addValues(nnodes, nodeNums, nnodes, nodeNums, nvars, nvars, mat);
  } else {
    // If we have dependent nodes, then we have to figure out what
    // the weighting matrix is and add then add the element matrix
    const int *depNodePtr = NULL;
    const int *depNodeConn = NULL;
    const double *depNodeWeights = NULL;
    if (depNodes) {
      depNodes->getDepNodes(&depNodePtr, &depNodeConn, &depNodeWeights);
    }

    // Set pointers to the temporary arrays
    int *varp = &itemp[0];
    int *vars = &itemp[nnodes + 1];
    TacsScalar *weights = temp;

    varp[0] = 0;
    for (int i = 0, k = 0; i < nnodes; i++) {
      if (nodeNums[i] >= 0) {
        // This is just a regular node
        weights[k] = 1.0;
        vars[k] = nodeNums[i];
        k++;
      } else {
        // This is a dependent node. Determine the corresponding
        // dependent node number and add the variables
        int dep = -nodeNums[i] - 1;
        for (int j = depNodePtr[dep]; j < depNodePtr[dep + 1]; j++, k++) {
          weights[k] = depNodeWeights[j];
          vars[k] = depNodeConn[j];
        }
      }

      varp[i + 1] = k;
    }

    // Add the values to the matrix
    A->addWeightValues(nnodes, varp, vars, weights, nvars, nvars, mat, matOr);
  }
}

// #endif  // TACS_ASSEMBLER_H
