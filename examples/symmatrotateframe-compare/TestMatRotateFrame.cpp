#include "a2dcore.h"
#include "TACSElementAlgebra.h"
#include <iostream>
#include <cstdlib> // For std::atoi
#include <complex>

int main(int argc, char* argv[]) {
    
    // make the matrices A,B,C (has to be size 3 as TACS is for size 3x3 matrices)
    A2D::A2DObj<A2D::Mat<TacsScalar,3,3>> A;
    A2D::A2DObj<A2D::SymMat<TacsScalar,3>> B_a2d, B_tacs, C_a2d, C_tacs;

    // initialize these matrices A,B
    for (int i = 0; i < 9; i++) {
        A.value().get_data()[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
    for (int j = 0; j < 6; j++) {
        // enforce that B is SymMat
        B_a2d.value().get_data()[j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        B_tacs.value().get_data()[j] = B_a2d.value().get_data()[j];
    }

    // // known values in 
    // TacsScalar *a = A.value().get_data();
    // a[0] = 1.0; a[4] = 1.0; a[8] = 1.0;
    // TacsScalar *b = B.value().get_data();
    // b[0] = -1; b[1] = -2; b[2] = -3; b[3] = -4; b[4] = -5; b[5] = -6;

    // compute forward with A2D approach
    auto mystack = A2D::MakeStack(
        A2D::SymMatRotateFrame(A, B_a2d, C_a2d)
    );

    // compute forward with TACS approach
    mat3x3SymmTransformTranspose(A.value().get_data(), B_tacs.value().get_data(), C_tacs.value().get_data());

    // compare these values
    printf("forward:---------------\n");
    double fw_max_rel_err = 0.0;
    for (int k = 0; k < 6; k++) {
        printf("ind %d: C_a2d %.8e, C_tacs %.8e\n", k, C_a2d.value().get_data()[k], C_tacs.value().get_data()[k]);
        double rel_err = (C_a2d.value().get_data()[k] - C_tacs.value().get_data()[k]) / C_tacs.value().get_data()[k];
        rel_err = abs(rel_err);
        if (rel_err > fw_max_rel_err) {
            fw_max_rel_err = rel_err;
        }
    }
    printf("forward pass max rel err = %.8e\n", fw_max_rel_err);
    printf("\n");

    // set C tacs bvalue()
    for (int k = 0; k < 6; k++) {
        C_a2d.bvalue().get_data()[k]= static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        C_tacs.bvalue().get_data()[k] = C_a2d.bvalue().get_data()[k];
    }

    // set hvalues
    for (int k = 0; k < 6; k++) {
        C_a2d.hvalue().get_data()[k] = 0.0;
        C_tacs.hvalue().get_data()[k] = 0.0;
    }


    // then compare 1st order grad
    mystack.reverse(); // 1st order grad in a2d

    // 1st order grad in TACS
    mat3x3SymmTransformTransSens(A.value().get_data(), C_tacs.bvalue().get_data(), B_tacs.bvalue().get_data());

    printf("1st order------------------\n");
    double first_max_rel_err = 0.0;
    for (int k = 0; k < 6; k++) {
        printf("ind %d: B_a2d %.8e, B_tacs %.8e\n", k, B_a2d.bvalue().get_data()[k], B_tacs.bvalue().get_data()[k]);
        double rel_err = (B_a2d.bvalue().get_data()[k] - B_tacs.bvalue().get_data()[k]) / B_tacs.bvalue().get_data()[k];
        rel_err = abs(rel_err);
        if (rel_err > first_max_rel_err) {
            first_max_rel_err = rel_err;
        }
    }
    printf("1st orderm max rel err = %.8e\n", first_max_rel_err);
    printf("\n");

    // initialize 2nd order variables and output hessian
    A2D::Mat<TacsScalar,6,6> Chess, Bhess_a2d, Bhess_tacs;
    for (A2D::index_t i = 0; i < 36; i++) {
        Chess.get_data()[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    // 2nd order in a2d (custom code the hextract since we are going from known output hessian not handled by stack)
    for (A2D::index_t i = 0; i < 6; i++) {
        A.pvalue().zero();
        B_a2d.pvalue().zero();
        C_a2d.pvalue().zero();

        // this line is in hextract, def need it
        B_a2d.hvalue().zero(); 

        // set pvalue to unit vector
        B_a2d.pvalue()[i] = 1.0;

        // forward sweep for pvalues
        mystack.hforward();
        // debug view C_a2d.pvalue()
        for (A2D::index_t irow = 0; irow < 6; irow++) {
            printf("C_a2d:pvalue[%d] = %.8e\n", irow, C_a2d.pvalue().get_data()[irow]);
        }

        // compute C_hess projected onto C.pvalue()
        C_a2d.hvalue().zero();
        for (A2D::index_t irow = 0; irow < 6; irow++) {
            for (A2D::index_t icol = 0; icol < 6; icol++) {
                // compute output projected hessian now
                // Chat = C_hessian * C:pvalue
                C_a2d.hvalue().get_data()[irow] += Chess(irow,icol) * C_a2d.pvalue().get_data()[icol];
            }
        }

        mystack.hreverse();

        // extract hessian at B level into Bhess
        for (A2D::index_t k = 0; k < 6; k++) {
            Bhess_a2d(k,i) = B_a2d.hvalue().get_data()[k];
        }
        
    }   
    // how to set hvalue() to give the correct B.hvalue()'s here?
    // mystack.hextract(B_a2d.pvalue(), B_a2d.hvalue(), hess_a2d); // includes reverse

    // 2nd order, hess in tacs
    mat3x3SymmTransformTransHessian(A.value().get_data(), Chess.get_data(), Bhess_tacs.get_data());

    // compare B hessian from each method
    printf("2nd order------------------\n");
    double sec_max_rel_err = 0.0;
    for (A2D::index_t i = 0; i < 36; i++) {
        printf("ind %d: Bhess_a2d %.8e, Bhess_tacs %.8e\n", i, Bhess_a2d.get_data()[i], Bhess_tacs.get_data()[i]);
        double rel_err = (Bhess_a2d.get_data()[i] - Bhess_tacs.get_data()[i]) / Bhess_tacs.get_data()[i];
        rel_err = abs(rel_err);
        if (rel_err > sec_max_rel_err) {
            sec_max_rel_err = rel_err;
        }
    }
    printf("2nd order max rel err = %.8e\n", sec_max_rel_err);

    printf("---------------------------\n\n");
    printf("Summary:\n");
    printf("forward max rel err = %.8e\n", fw_max_rel_err);
    printf("1st order max rel err = %.8e\n", first_max_rel_err);
    printf("2nd order max rel err = %.8e\n", sec_max_rel_err);

    return 0;
}
