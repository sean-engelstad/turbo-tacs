#include "a2dcore.h"
#include "adscalar.h"

int main(int argc, char* argv[]) {
    
    using T = A2D::ADScalar<double,1>;

    T x(1.46e-5);
    T y = sqrt(x);
    printf("x = %.8e\n", x.value);
    printf("y = %.8e\n", y.value);

    return 0;
}
