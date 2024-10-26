#include "../therm-cylinder-include/getKDF.h"

// this is a nonlinear buckling example of a cylinder under mechanical loading
// with applied geometric imperfections. The load factor for nonlinear buckling is determined automatically.
// and the KDF (ratio of NL load factor / Linear load factor for buckling) or knockdown factor is computed and saved.


int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    // Get the rank
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank;
    MPI_Comm_rank(comm, &rank);

    const int NRUNS = 6;
    double rtVals[NRUNS] = {500.0, 300.0, 100.0, 50.0, 25.0, 10.0};
    int meshSizes[3] = {10000, 20000, 40000};
    TacsScalar nasaKDF[NRUNS] = { };
    TacsScalar tacsKDF[NRUNS] = { };
    double E = 70e9; // 70e3
    double conv_eigval = 0.3;
    double conv_slope_frac = 0.1;
    double temperature = 1.0; // maybe should be list?
    bool ringStiffened = false; double ringStiffenedRadiusFrac = 0.9;

    double t = 0.002;
    double Lr = 2.0;
    const int NUM_IMP = 3;
    std::string filePrefix = "_runs/";
    // for debugging
    // TacsScalar imperfections[NUM_IMP] = {0.0 * t, 0.0, 0.0 };
    TacsScalar imperfections[NUM_IMP] = {0.5 * t, 0.0, 0.0 };
    bool useEigvals = false; // use load-disp curve for thermal buckling

    FILE *fp;
    if (rank == 0) {
        fp = fopen("kdfs.csv", "w");
        if (fp) {
            fprintf(fp, "KDFs of cylinder with 0.5 * t mode 1 imperfection, t = %10.3e, L/r = %10.3e\n", t, Lr);
            fprintf(fp, "nelems, r/t, nasaKDF, tacsKDF\n");
            fflush(fp);
        }
    }
    

    // run each KDF simulation for mechanical nonlinear buckling
    int irun = 0;
    for (int inelems = 0; inelems < 3; inelems++) {
        for (int i_rt = NRUNS-1; i_rt >= 0; i_rt--) {
            irun++;
            double rt = rtVals[i_rt];
            // int nelems = meshSizes[irun]; // 5000, 10000
            int nelems = meshSizes[inelems];
            getNonlinearBucklingKDF(
                comm, irun, filePrefix, t, rt, Lr, E, temperature,
                ringStiffened, ringStiffenedRadiusFrac,
                conv_eigval, conv_slope_frac,
                NUM_IMP, &imperfections[0],
                useEigvals, nelems, &nasaKDF[i_rt], &tacsKDF[i_rt]
            );

            // writeout KDFs to csv file
            if (fp) {
                fprintf(fp, "%d, %10.5e, %10.5e, %10.5e\n", nelems, rt, TacsRealPart(nasaKDF[i_rt]), TacsRealPart(tacsKDF[i_rt]));
                fflush(fp);
            }
        }   
    }
    

    MPI_Finalize();

    return 0;
}