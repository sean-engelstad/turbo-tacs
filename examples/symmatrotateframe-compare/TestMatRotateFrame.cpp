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
        B_tacs.value().get_data()[j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
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
    for (int k = 0; k < 6; k++) {
        printf("ind %d: C_a2d %.8e, C_tacs %.8e\n", k, C_a2d.value().get_data()[k], C_tacs.value().get_data()[k]);
    }

    // set C tacs bvalue()
    for (int k = 0; k < 6; k++) {
        C_a2d.bvalue().get_data()[k]= static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        C_tacs.bvalue().get_data()[k] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }


    // then compare 1st order grad
    mystack.reverse(); // 1st order grad in a2d

    // 1st order grad in TACS
    mat3x3SymmTransformSens(A.value().get_data(), C_tacs.bvalue().get_data(), B_tacs.bvalue().get_data());

    for (int k = 0; k < 6; k++) {
        printf("ind %d: B_a2d %.8e, B_tacs %.8e\n", k, B_a2d.bvalue().get_data()[k], B_tacs.bvalue().get_data()[k]);
    }

    // then compare 2nd order hess

    return 0;
}
