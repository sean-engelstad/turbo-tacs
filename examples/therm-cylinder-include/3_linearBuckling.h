#include "KSM.h"
#include "TACSBuckling.h"
#include "TACSToFH5.h"
#include "_applyImperfections.h"

double linearBuckling(TACSAssembler *assembler, TACSMat *kmat, TACSSchurPc *pc,
                      int NUM_IMP, TacsScalar *imperfection_sizes) {
    // create the matrices for buckling
    TACSSchurMat *gmat = assembler->createSchurMat();  // geometric stiffness matrix
    TACSSchurMat *aux_mat = assembler->createSchurMat();  // auxillary matrix for shift and invert solver

    // optional other preconditioner settings?
    assembler->assembleMatType(TACS_STIFFNESS_MATRIX, kmat);
    assembler->assembleMatType(TACS_GEOMETRIC_STIFFNESS_MATRIX, gmat);

    int subspaceSize = 10, nrestarts = 15, isFlexible = 0;
    GMRES *lbuckle_gmres = new GMRES(aux_mat, pc, subspaceSize, nrestarts, isFlexible);
    lbuckle_gmres->incref();
    lbuckle_gmres->setTolerances(1e-12, 1e-12);

    // make the buckling solver
    int max_lanczos_vecs = 300, num_eigvals = 100; // num_eigvals = 50;
    double eig_tol = 1e-12;
    double sigma = 10.0;
    TACSLinearBuckling *buckling = new TACSLinearBuckling(assembler, sigma,
                     gmat, kmat, aux_mat, lbuckle_gmres, max_lanczos_vecs, num_eigvals, eig_tol);
    buckling->incref();

    // make a KSM print object for solving buckling
    KSMPrint *ksm_print_buckling = new KSMPrintStdout("BucklingAnalysis", 0, 10);
    ksm_print_buckling->incref();

    // solve the buckling analysis
    buckling->setSigma(10.0);
    buckling->solve(NULL, NULL, ksm_print_buckling);

    // compute linear eigval based on initial thermal buckling estimate
    TacsScalar error;
    TacsScalar linear_eigval = buckling->extractEigenvalue(0, &error);
    printf("linear eigval = %.8e\n", linear_eigval);

    // Create an TACSToFH5 object for writing output to files
    int write_flag = (TACS_OUTPUT_CONNECTIVITY | TACS_OUTPUT_NODES |
                        TACS_OUTPUT_DISPLACEMENTS | TACS_OUTPUT_STRAINS |
                        TACS_OUTPUT_STRESSES | TACS_OUTPUT_EXTRAS);
    TACSToFH5 *f5 = new TACSToFH5(assembler, TACS_BEAM_OR_SHELL_ELEMENT, write_flag);
    f5->incref();

    // write the linear buckling solution to a file
    TACSBVec *phi = assembler->createVec();
    phi->incref();
    buckling->extractEigenvector(0, phi, &error);
    assembler->setVariables(phi);   
    f5->writeToFile("linear-buckle.f5");

    // apply imperfections to the structure
    applyImperfections(assembler, buckling, NUM_IMP, imperfection_sizes);

    // return the eigenvalue
    return TacsRealPart(linear_eigval);
}