#include "TACSMeshLoader.h"
#include "TACSAssembler.h"

// dependencies to make element, constitutive objects
#include "TACSShellElementTransform.h"
#include "TACSMaterialProperties.h"
#include "TACSIsoShellConstitutive.h"
#include "TACSShellElementDefs.h"

#include <iostream>
#include <chrono>
#include <thread>

// this example is based off of examples/crm/crm.cpp in TACS

int main() {
    // make the MPI communicator
    MPI_Init(NULL, NULL);
    MPI_Comm comm = MPI_COMM_WORLD;

    int rank;
    MPI_Comm_rank(comm, &rank);

    // build the TACS mesh loader and scan the uCRM BDF file
    if (rank == 0) {
        printf("Scanning BDF file\n");
    }
    
    TACSMeshLoader *mesh = new TACSMeshLoader(comm);
    mesh->incref();
    mesh->scanBDFFile("CRM_box_2nd.bdf");

    // get number of components
    int num_components = mesh->getNumComponents();

    // create the shell ref axis transform
    // TacsScalar refAxis[3] = {1.0, 0.0, 0.0};
    // TACSShellRefAxisTransform *transform = new TACSShellRefAxisTransform(refAxis);
    
    TACSShellTransform *transform = new TACSShellNaturalTransform();

    // set material properties for aluminum (no thermal props input this time)
    TacsScalar rho = 2718;
    TacsScalar specific_heat = 0.0;
    TacsScalar E = 72.0e9;
    TacsScalar nu = 0.33;
    TacsScalar ys = 1e11;
    TacsScalar cte = 0.0;
    TacsScalar kappa = 0.0;
    TACSMaterialProperties *mat = 
          new TACSMaterialProperties(rho, specific_heat, E, nu, ys, cte, kappa);
    
    // create the constitutive and element objects for each TACS component
    if (rank == 0) {
        printf("Creating constitutive and element objects for each TACS component\n");
    }
    
    for (int icomp = 0; icomp < num_components; icomp++) {
        const char *descriptor = mesh->getElementDescript(icomp);
        TacsScalar thick = 0.010; // shell thickness
        TACSIsoShellConstitutive *con = new TACSIsoShellConstitutive(mat, thick);

        // now create the shell element object
        // TACSElement *shell = TacsCreateShellByName(descriptor, transform, con);
        TACSElement *shell = new TACSQuad4ShellOrig(transform, con);

        // set the shell element into the mesh loader for that component
        mesh->setElement(icomp, shell);
    }

    mat->decref();
    transform->decref();

    if (rank == 0) {
        printf("\tdone\n");
    }
    

    // now create the TACS assembler
    if (rank == 0) {
        printf("Creating TACS assembler\n");
    }
    
    int vars_per_node = 6; // for shell elements
    TACSAssembler *assembler = mesh->createTACS(vars_per_node);
    assembler->incref();
    mesh->decref();
    if (rank == 0) {
        printf("Done\n");
    }

    // Solve the linear static analysis
    if (rank == 0) {
        printf("Solving linear system..\n");
    }
    
    // Create the design vector
    TACSBVec *x = assembler->createDesignVec();
    x->incref();

    // Get the design variable values
    assembler->getDesignVars(x);

    // Create matrix and vectors
    TACSBVec *ans = assembler->createVec();  // displacements and rotations
    TACSBVec *f = assembler->createVec();    // loads
    TACSBVec *res = assembler->createVec();  // The residual
    TACSSchurMat *matrix = assembler->createSchurMat();  // stiffness matrix

    // Increment reference count to the matrix/vectors
    ans->incref();
    f->incref();
    res->incref();
    matrix->incref();

    // Allocate the factorization
    int lev = 10000;
    double fill = 10.0;
    int reorder_schur = 1;
    TACSSchurPc *pc = new TACSSchurPc(matrix, lev, fill, reorder_schur);
    pc->incref();

    // Set all the entries in load vector to specified value
    TacsScalar *force_vals;
    int size = f->getArray(&force_vals);
    for (int k = 2; k < size; k += 6) {
        force_vals[k] += 100.0;
    }
    assembler->applyBCs(f);

    // Assemble and factor the stiffness/Jacobian matrix. Factor the
    // Jacobian and solve the linear system for the displacements
    printf("Begin assemble Jacobian\n");
    //auto t1 = std::chrono::high_resolution_clock::now();
    double t1 = MPI_Wtime();

    double alpha = 1.0, beta = 0.0, gamma = 0.0;
    assembler->assembleJacobian<Quad4ShellOrig>(alpha, beta, gamma, NULL, matrix);
 
    double t2 = MPI_Wtime();
    //auto t2 = std::chrono::high_resolution_clock::now();
    //TacsScalar dt = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() / 1000.0;
    double dt = t2 - t1;
    printf("Done with assemble Jacobian in %.8e sec\n", dt);

    // OPTIONAL test assembling the residuals
    // bool test_res = true; // doesn't use in solve but just to see if code works.
    // if (test_res) {
    //     assembler->assembleRes(res, 1.0);
    // }

    pc->factor();  // LU factorization of stiffness matrix
    pc->applyFactor(f, ans);
    assembler->setVariables(ans);

    

    if (rank == 0) {
        printf("Done with RunCRM!\n");
    }

    // Create an TACSToFH5 object for writing output to files
    int write_flag = (TACS_OUTPUT_CONNECTIVITY | TACS_OUTPUT_NODES |
                        TACS_OUTPUT_DISPLACEMENTS | TACS_OUTPUT_STRAINS |
                        TACS_OUTPUT_STRESSES | TACS_OUTPUT_EXTRAS);
    TACSToFH5 *f5 =
        new TACSToFH5(assembler, TACS_BEAM_OR_SHELL_ELEMENT, write_flag);
    f5->incref();
    f5->writeToFile("ucrm.f5");
    f5->decref();

    // decref all data
    x->decref();
    matrix->decref();
    pc->decref();
    ans->decref();
    f->decref();
    res->decref();

    // not working for some reason.. (signal 11)
    // assembler->decref();

    MPI_Finalize();

    return 0;
}
