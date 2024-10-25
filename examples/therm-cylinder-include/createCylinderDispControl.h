#pragma once
#include "TACSAssembler.h"
#include "TACSCreator.h"
#include "TACSMeshLoader.h"

// NOTE : on lines 76-100, I changed removed the rotx BCs from the BC set
// since this might have been affecting the buckling mode solution..

/*
  Create the TACSAssembler object and return the associated TACS
  creator object
*/
void createAssembler(MPI_Comm comm, int order, int nx, int ny, 
                     double length, double radius,
                     bool urStarBC, TacsScalar urStar, 
                     bool ringStiffened, double ringStiffenedRadiusFrac,
                     TACSElement *element, TACSAssembler **_assembler,
                     TACSCreator **_creator) {
  int rank;
  MPI_Comm_rank(comm, &rank);

  double L = length;
  double R = radius;

  // Set the number of nodes/elements on this proc
  int varsPerNode = element->getVarsPerNode();

  // Set up the creator object
  TACSCreator *creator = new TACSCreator(comm, varsPerNode);

  if (rank == 0) {
    // Set the number of elements
    int nnx = (order - 1) * nx + 1;
    int nny = (order - 1) * ny;
    int numNodes = nnx * nny;
    int numElements = nx * ny;

    // Allocate the input arrays into the creator object
    int *ids = new int[numElements];
    int *ptr = new int[numElements + 1];
    int *conn = new int[order * order * numElements];

    // Set the element identifiers to all zero
    memset(ids, 0, numElements * sizeof(int));

    ptr[0] = 0;
    for (int k = 0; k < numElements; k++) {
      // Back out the i, j coordinates from the corresponding
      // element number
      int i = k % nx;
      int j = k / nx;

      // Set the node connectivity
      for (int jj = 0; jj < order; jj++) {
        for (int ii = 0; ii < order; ii++) {
          if (j == ny - 1 && jj == order - 1) {
            conn[order * order * k + ii + order * jj] = ((order - 1) * i + ii);
          } else {
            conn[order * order * k + ii + order * jj] =
                ((order - 1) * i + ii) + ((order - 1) * j + jj) * nnx;
          }
        }
      }

      ptr[k + 1] = order * order * (k + 1);
    }

    // Set the connectivity
    creator->setGlobalConnectivity(numNodes, numElements, ptr, conn, ids);
    delete[] conn;
    delete[] ptr;
    delete[] ids;

    int numBcs;
    int *bcNodes, *bc_ptr, *bc_vars;
    TacsScalar *bc_vals;
    int k = 0;
    
    if (!ringStiffened) {
      numBcs = 2 * nny;
      bcNodes = new int[numBcs];
      bc_ptr = new int[numBcs + 1];
      bc_vars = new int[3 * numBcs];
      bc_vals = new TacsScalar[3 * numBcs];

      // iterate over the bc nodes and values into the bc arrays
      bc_ptr[0] = 0;
      int i = 0; // BC counter

      // previously had u, v, w, rotx but that doesn't bode well for the buckling problem
      // maybe should only constrain rotation at like 1 point or so
      // if do want rotx constrained, then you should set 4 * numBCs instead of 3 * numBCs above and loop to m < 4
      for (int j = 0; j < nny; j++) { // set u, v, w, xrot to zero
        TacsScalar th = -M_PI + (2.0 * M_PI * j) / nny;
        // bc at x- edge
        bc_ptr[i+1] = bc_ptr[i]; // start BC dof counter
        int node = j * nnx;
        bcNodes[k] = node;
        for (int m = 0; m < 3; m++) {
          bc_vars[bc_ptr[i+1]] = m; // DOF m is set to 0 disp
          TacsScalar disp;
          if (urStarBC) {
            if (m == 0) { 
              disp == 0;
            } else if (m == 1) {
              disp = urStar * cos(th);
            } else if (m == 2) {
              disp = urStar * -sin(th);
            }
          } else { disp = 0.0; }
          bc_vals[bc_ptr[i+1]] = disp;
          bc_ptr[i+1]++;
          // printf("x- node %d with DOF %d set to %.2e with total BCs %d\n", node, m, bc_vals[bc_ptr[i+1]-1], bc_ptr[i+1]);
        }
        k++; i++;

        bc_ptr[i+1] = bc_ptr[i]; // start BC dof counter
        node = nnx - 1 + j * nnx;
        bcNodes[k] = node;
        for (int m = 0; m < 3; m++) { // set x = udisp and y,z,rotx to 0
          bc_vars[bc_ptr[i+1]] = m; // DOF m is set to 0 disp
          TacsScalar disp;
          if (urStarBC) {
            if (m == 0) { 
              disp == 0;
            } else if (m == 1) {
              disp = urStar * cos(th);
            } else if (m == 2) {
              disp = urStar * -sin(th);
            }
          } else { disp = 0.0; }
          bc_vals[bc_ptr[i+1]] = disp;
          bc_ptr[i+1]++;
          // printf("x+ node %d with DOF %d set to %.2e with total BCs %d\n", node, m, bc_vals[bc_ptr[i+1]-1], bc_ptr[i+1]);
        }
        k++; i++;
      }
    // end of if (!ringStiffened) section
    } else if (ringStiffened) {

      numBcs = 8 * nny; // total number of nodes w/ BCs specified
      bcNodes = new int[numBcs];
      bc_ptr = new int[numBcs + 1];
      int numDOF_cons = 12 * nny; // length - total # of dof constraints among all nodes
      bc_vars = new int[numDOF_cons]; 
      bc_vals = new TacsScalar[numDOF_cons]; // length - total # of dof constraints among all nodes

      // iterate over the bc nodes and values into the bc arrays
      bc_ptr[0] = 0;
      int i = 0; // BC counter

      // previously had u, v, w, rotx but that doesn't bode well for the buckling problem
      // maybe should only constrain rotation at like 1 point or so
      // if do want rotx constrained, then you should set 4 * numBCs instead of 3 * numBCs above and loop to m < 4
      for (int j = 0; j < nny; j++) { // set u, v, w, xrot to zero
        TacsScalar th = -M_PI + (2.0 * M_PI * j) / nny;
        // bc at x- edge
        bc_ptr[i+1] = bc_ptr[i]; // start BC dof counter
        int node = j * nnx;
        bcNodes[k] = node;
        for (int m = 0; m < 3; m++) {
          bc_vars[bc_ptr[i+1]] = m; // DOF m is set to 0 disp
          TacsScalar disp;
          if (urStarBC) {
            if (m == 0) { 
              disp == 0;
            } else if (m == 1) {
              disp = urStar * cos(th);
            } else if (m == 2) {
              disp = urStar * -sin(th);
            }
          } else { disp = 0.0; }
          bc_vals[bc_ptr[i+1]] = disp;
          bc_ptr[i+1]++;
          // printf("x- node %d with DOF %d set to %.2e with total BCs %d\n", node, m, bc_vals[bc_ptr[i+1]-1], bc_ptr[i+1]);
        }
        k++; i++;

        // bc of x- edge inner ring (set u1 = 0)
        for (int iring = 0; iring < 3; iring++) {
          bc_ptr[i+1] = bc_ptr[i]; // start BC dof counter
          int node = j * nnx + iring + 1;
          bcNodes[k] = node;
          bc_vars[bc_ptr[i+1]] = 0; // DOF m is set to 0 disp
          bc_vals[bc_ptr[i+1]] = 0.0;
          bc_ptr[i+1]++;
          k++; i++;
        }

        // bc of x+ edge inner ring (set u1 = 0)
        for (int iring = 0; iring < 3; iring++) {
          bc_ptr[i+1] = bc_ptr[i]; // start BC dof counter
          int node = nnx - 1 + j * nnx - (iring+1);
          bcNodes[k] = node;
          bc_vars[bc_ptr[i+1]] = 0; // DOF m is set to 0 disp
          bc_vals[bc_ptr[i+1]] = 0.0;
          bc_ptr[i+1]++;
          k++; i++;
        }

        bc_ptr[i+1] = bc_ptr[i]; // start BC dof counter
        node = nnx - 1 + j * nnx;
        bcNodes[k] = node;
        for (int m = 0; m < 3; m++) { // set x = udisp and y,z,rotx to 0
          bc_vars[bc_ptr[i+1]] = m; // DOF m is set to 0 disp
          TacsScalar disp;
          if (urStarBC) {
            if (m == 0) { 
              disp == 0;
            } else if (m == 1) {
              disp = urStar * cos(th);
            } else if (m == 2) {
              disp = urStar * -sin(th);
            }
          } else { disp = 0.0; }
          bc_vals[bc_ptr[i+1]] = disp;
          bc_ptr[i+1]++;
          // printf("x+ node %d with DOF %d set to %.2e with total BCs %d\n", node, m, bc_vals[bc_ptr[i+1]-1], bc_ptr[i+1]);
        }
        k++; i++;
      }
    } // end of if ringStiffened
    
    
    // printf("udisp = %.8f", udisp);

    // Set the boundary conditions
    creator->setBoundaryConditions(numBcs, bcNodes, bc_ptr, bc_vars, bc_vals);

    delete[] bcNodes;
    delete[] bc_ptr;
    delete[] bc_vars;
    delete[] bc_vals;

    // Set the node locations
    TacsScalar *Xpts = new TacsScalar[3 * numNodes];

    for (int j = 0; j < nny; j++) {
      double v = -M_PI + (2.0 * M_PI * j) / nny;
      for (int i = 0; i < nnx; i++) {
        double theta = v;
        double x, y, z;
        double cradius;

        if (i < 4 && ringStiffened) {
          // front ring stiffener
          x = 0;
          cradius = R * (ringStiffenedRadiusFrac + (1.0-ringStiffenedRadiusFrac) * i / 3);
          y = cradius * cos(theta);
          z = -cradius * sin(theta);

        } else if (i >= (nnx - 4) && ringStiffened) {
          // back ring stiffener
          x = L;
          cradius = R * (1.0 - (1.0-ringStiffenedRadiusFrac) * (i - nnx + 4) / 3);
          y = cradius * cos(theta);
          z = -cradius * sin(theta);
        } else if ( (3 <= i && i < nnx - 3) || !ringStiffened) {
          // middle section
          double u;
          if (ringStiffened) {
            u = 1.0 * (i - 3) / (nnx - 7);
          } else {
            u = 1.0 * i / (nnx - 1);
          }
          x = L*u;
          y = R * cos(theta);
          z = -R * sin(theta); 
        }

        int node = i + j * nnx;
        Xpts[3 * node] = x;
        Xpts[3 * node + 1] = y;
        Xpts[3 * node + 2] = z;

      }
    }

    // Set the nodal locations
    creator->setNodes(Xpts);
    delete[] Xpts;
  }

  // Set the one element
  creator->setElements(1, &element);

  // Set the reordering type
  creator->setReorderingType(TACSAssembler::MULTICOLOR_ORDER,
                             TACSAssembler::GAUSS_SEIDEL);

  // Create TACS
  TACSAssembler *assembler = creator->createTACS();

  // Set the elements the node vector
  TACSBVec *X = assembler->createNodeVec();
  X->incref();
  assembler->getNodes(X);

  X->decref();

  // Set the pointers
  *_assembler = assembler;
  *_creator = creator;
}