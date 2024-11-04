#include "a2dcore.h"
#include "adscalar.h"

int main(int argc, char* argv[]) {
    
    A2D::ADScalar<double,4> X;

    using T = A2D::ADScalar<double,24>;
    T Y[4];

    A2D::A2DObj<A2D::ADScalar<double,4>> A, B, C;
    A.value().value = 1.0;
    B.value().value = 3.0;
    
    auto mystack = A2D::MakeStack(
        A2D::Eval(A * B, C)
    );

    printf("C val = %.8e\n", C.value().value);


    return 0;
}
