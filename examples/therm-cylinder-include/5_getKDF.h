#include "4_runNonlinearStatic.h"

void getKDF(MPI_Comm comm, std::string filePrefix, double t, double rt, double Lr, double temperature, double E,
            int nelems, bool urStarBC, bool ringStiffened, double ringStiffenedRadiusFrac,
            const int NUM_IMP, TacsScalar *imperfections,
            double rtol, double atol, double conv_slope_frac,
            double *tacsKDF, double *nasaKDF) {

    int rank;
    MPI_Comm_rank(comm, &rank);

    // just perfect geometry in this case, can add in imperfections later maybe
    TacsScalar no_imperfections[NUM_IMP] = { };

    // run the perfect cylinder case
    double lamNL_perfect, lamNL_imperfect;
    runNonlinearStatic(
        comm, filePrefix, t, rt, Lr, E, temperature,
        nelems, conv_slope_frac,
        rtol, atol,
        urStarBC, ringStiffened, ringStiffenedRadiusFrac,
        NUM_IMP, no_imperfections,
        &lamNL_perfect
    );

    // run the cylinder in the imperfect case
    runNonlinearStatic(
        comm, filePrefix, t, rt, Lr, E, temperature,
        nelems, conv_slope_frac,
        rtol, atol,
        urStarBC, ringStiffened, ringStiffenedRadiusFrac,
        NUM_IMP, imperfections,
        &lamNL_imperfect
    );

    *tacsKDF = lamNL_imperfect / lamNL_perfect;
    double kdf_phi = 1.0 / 16.0 * sqrt(rt);
    *nasaKDF = 1.0 - 0.901 * (1.0 - exp(-kdf_phi));
    printf("tacs KDF = %.8e, nasa KDF = %.8e\n", *tacsKDF, *nasaKDF);
    
    // write to an output file
    FILE *fp;
    if (rank == 0) {
        std::string filename = filePrefix + "/nlbuckling.out";
        const char *cstr_filename = filename.c_str();
        fp = fopen(cstr_filename, "w");
        
        if (fp) {
            fprintf(fp, "t = %.8e, r/t = %.8e, L/r = %.8e, nelems = %d", t, rt, Lr, nelems);
            fprintf(fp, "tacs KDF = %.8e, nasa KDF = %.8e\n", *tacsKDF, *nasaKDF);
            fflush(fp);
        }
    }
}