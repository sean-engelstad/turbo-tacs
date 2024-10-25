#include "../therm-cylinder-include/5_getKDF.h"

// this is a nonlinear buckling example of a cylinder under mechanical loading
// with applied geometric imperfections. The load factor for nonlinear buckling is determined automatically.
// and the KDF (ratio of NL load factor / Linear load factor for buckling) or knockdown factor is computed and saved.


int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    // Get the rank
    MPI_Comm comm = MPI_COMM_WORLD;

    // material and geometric inputs
    double t = 0.002, rt = 100, Lr = 2.0;
    double temperature = 1.0;
    double E = 70e9;

    // mesh, BC, and solver settings
    int nelems = 20000;
    bool urStarBC = false, ringStiffened = false;
    double ringStiffenedRadiusFrac = 0.9;
    int NUM_IMP = 3;
    TacsScalar imperfections[NUM_IMP] = {0.0, 0.0, 0.5 * t };
    double rtol = 1e-6, atol = 1e-10, conv_slope_frac = 0.2;
    double tacsKDF, nasaKDF;

    getKDF(
        comm, t, rt, Lr, temperature, E,
        nelems, urStarBC, ringStiffened, ringStiffenedRadiusFrac,
        NUM_IMP, &imperfections[0],
        rtol, atol, conv_slope_frac,
        &tacsKDF, &nasaKDF
    );

    MPI_Finalize();

    return 0;
}