#include "../therm-cylinder-include/createCylinderDispControl.h"
#include "TACSShellElementTransform.h"
#include "TACSMaterialProperties.h"
#include "TACSIsoShellConstitutive.h"
#include "TACSShellElementDefs.h"
#include "TACSBuckling.h"
#include "KSM.h"
#include "TACSContinuation.h"

void makeAssembler(MPI_Comm comm, double t, double rt, double Lr, double E, double temperature, 
                   bool urStarBC,
                   bool ringStiffened, double ringStiffenedRadiusFrac,
                   int nelems, TACSAssembler **assembler) {
    double R = t * rt; // m
    double L = R * Lr;
    double udisp = 0.0; // ( for r/t = 25 )to be most accurate want udisp about 1/200 to 1/300 the linear buckling disp

    // select nelems and it will select to retain isotropic elements (good element AR)
    // want dy = 2 * pi * R / ny the hoop elem spacing to be equal dx = L / nx the axial elem spacing
    // and want to choose # elems so that elements have good elem AR
    // int nelems = 5000; // prev 3500 // target (does round stuff)
    double pi = 3.14159265;
    double A = L / 2.0 / pi / R;
    double temp1 = sqrt(nelems * 1.0 / A);
    int ny = (int)temp1;
    double temp2 = A * ny;
    int nx = (int)temp2;
    printf("nx = %d, ny = %d\n", nx, ny);

    TacsScalar rho = 2700.0;
    TacsScalar specific_heat = 921.096;
    TacsScalar nu = 0.3;
    TacsScalar ys = 270.0;
    TacsScalar cte = 10.0e-6;
    TacsScalar kappa = 230.0;
    TACSMaterialProperties *props = new TACSMaterialProperties(rho, specific_heat, E, nu, ys, cte, kappa);

    double urStar = (1 + nu) * cte * R * temperature;

    // TacsScalar axis[] = {1.0, 0.0, 0.0};
    // TACSShellTransform *transform = new TACSShellRefAxisTransform(axis);
    TACSShellTransform *transform = new TACSShellNaturalTransform();
    TACSShellConstitutive *con = new TACSIsoShellConstitutive(props, t);

    TACSCreator *creator = NULL;
    TACSElement *shell = NULL;
    // needs to be nonlinear here otherwise solve will terminate immediately
    shell = new TACSQuad4NonlinearShell(transform, con); 
    shell->incref();

    // createAssembler(comm, 2, nx, ny, udisp, L, R, 
    // ringStiffened, ringStiffenedRadiusFrac,
    // shell, assembler, &creator);

    createAssembler(
        comm, 2, nx, ny,
        L, R,
        urStarBC, urStar,
        ringStiffened, ringStiffenedRadiusFrac,
        shell, assembler, &creator
    );  
}

double linearBuckling(TACSAssembler *assembler, TACSMat *kmat, TACSSchurPc *pc) {
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

    // return the eigenvalue
    return TacsRealPart(linear_eigval);
}

void runNonlinearStatic(MPI_Comm comm, 
    double t, double rt, double Lr, 
    double E, double temperature, 
    int nelems, double conv_slope_frac,
    double rtol, double atol,
    bool urStarBC,
    bool ringStiffened, double ringStiffenedRadiusFrac,
    int num_imperfections, TacsScalar *imperfection_sizes,
    TacsScalar *nasaKDF, TacsScalar *tacsKDF
    ) {
    
    int rank;
    MPI_Comm_rank(comm, &rank);

    // make the assembler
    TACSAssembler *assembler = NULL;
    makeAssembler(comm, t, rt, Lr, E, temperature,
     urStarBC, ringStiffened, ringStiffenedRadiusFrac, nelems, &assembler);

    // Create the design vector
    TACSBVec *x = assembler->createDesignVec(); // design variables
    TACSBVec *u0 = assembler->createVec();  // displacements and rotations
    TACSBVec *f = assembler->createVec();    // loads
    u0->incref(); f->incref(); x->incref();
    f->zeroEntries();
    assembler->getDesignVars(x);

    // make the stiffness matrix
    TACSSchurMat *kmat = assembler->createSchurMat();  // stiffness matrix
    kmat->incref();

    // Allocate the factorization
    int lev = 1e6; double fill = 10.0; int reorder_schur = 1;
    TACSSchurPc *pc = new TACSSchurPc(kmat, lev, fill, reorder_schur);
    pc->incref();

    // Create an TACSToFH5 object for writing output to files
    int write_flag = (TACS_OUTPUT_CONNECTIVITY | TACS_OUTPUT_NODES |
                        TACS_OUTPUT_DISPLACEMENTS | TACS_OUTPUT_STRAINS |
                        TACS_OUTPUT_STRESSES | TACS_OUTPUT_EXTRAS);
    TACSToFH5 *f5 = new TACSToFH5(assembler, TACS_BEAM_OR_SHELL_ELEMENT, write_flag);
    f5->incref();

    // get linear buckling eigenvalue and adjust initial temp
    assembler->setTemperatures(temperature);
    double linear_eigval = linearBuckling(assembler, kmat, pc);
    double lambda_adjustment = linear_eigval / 200.0;
    // temperature *= linear_eigval / 200.0; // good rule is if the linear buckling is at 200 about zero disps, then do delta_lambda = 5.0
    // if (fp) {
    //     fprintf(fp, "linear eigval = %.8e\n", TacsRealPart(linear_eigval));
    //     fflush(fp);
    // }

    // make the linear solver (GMRES)
    int subspaceSize = 10, nrestarts = 15, isFlexible = 0;
    GMRES *gmres = new GMRES(kmat, pc, subspaceSize, nrestarts, isFlexible);
    gmres->incref();
    gmres->setTolerances(1e-12, 1e-12);

    // make a KSM print object for solving buckling
    KSMPrint *ksm_print = new KSMPrintStdout("NonlinearStatic", 0, 10);
    ksm_print->incref();

    // main output file
    FILE *fp;
    if (rank == 0) {
        fp = fopen("load-disp.csv", "w");
        if (fp) {
            fprintf(fp, "iter,|u|,lambda/lamLin,minS11,avgS11,maxS11,avgSlopeFrac\n");
            fflush(fp);
        }
    }

    // initial temperature and settings
    // lambda_adjustment = linear_eigval / 200.0 so that buckling should happen at lambda = 200.0
    TacsScalar delta_lambda = 3.0 * lambda_adjustment; // load-factor step size (for 200 total)
    TacsScalar lambda_init = 10.0 * lambda_adjustment; // init load factor
    TacsScalar lambda = lambda_init;
    TacsScalar res_norm_init = 1.0; double target_res_norm;

    // temp states needed for solve
    TACSBVec *vars = assembler->createVec();
    TACSBVec *old_vars = assembler->createVec();
    TACSBVec *delta_vars = assembler->createVec();
    TACSBVec *temp = assembler->createVec();
    TACSBVec *du = assembler->createVec();
    TACSBVec *update = assembler->createVec();
    TACSBVec *res = assembler->createVec();

    // slope / load-disp checking states
    TacsScalar max_stress, max_stress_old = -1.0e40;
    TacsScalar min_stress, min_stress_old = -1.0e40; // the abs min stresses, etc.
    TacsScalar avg_stress, avg_stress_old = -1.0e40;
    TacsScalar *tempStresses = new TacsScalar[9];
    TacsScalar init_slope, c_slope;
    

    // prelim Newton solve
    // ---------------------------------------------

    double t0 = MPI_Wtime();
    if (ksm_print) {
      char line[256];
      ksm_print->print("Performing initial Newton iterations\n");
      sprintf(line, "%5s %9s %10s\n", "Iter", "t", "|R|");
      ksm_print->print(line);
    }

    // get initial K_t(u0,lambda) and solve for u with updates
    assembler->setTemperatures(lambda * temperature);
    assembler->assembleJacobian(1.0, 0.0, 0.0, res, kmat, TACS_MAT_NORMAL, 1.0, lambda);
    pc->factor();
    gmres->setTolerances(rtol, atol); // rtol, atol
    gmres->solve(f, du);
    vars->axpy(1.0, du); // prelim u0 guess

    // Newton solve loop to get u_init, lambda = lambda_init (for init load factor)   
    for (int inewton = 0; inewton < 100; inewton++) {
        // update nonlinear residual
        assembler->setTemperatures(lambda * temperature); // prob don't need to reset this, but still fine
        assembler->setVariables(vars);
        assembler->assembleRes(res, 1.0, lambda);
        // no load control effects so no res->axpy(-lambda, f)

        // solve K_t(u,lambda') * du = - R(u,lambda'), then do the update u' = u + du
        gmres->solve(res, update); // get -du here
        vars->axpy(-1.0, update);

        // report the residual R(u,lambda')
        TacsScalar res_norm = res->norm();
        if (ksm_print) { // print residual norm of prelim Newton solve
            char line[256];
            sprintf(line, "%5d %9.4f %10.4e\n", inewton + 1, MPI_Wtime() - t0,
                    TacsRealPart(res_norm));
            ksm_print->print(line);
        }

        // check for convergence
        if (inewton == 0) {
            res_norm_init = res_norm;
            target_res_norm = rtol * TacsRealPart(res_norm_init) + atol;
        }
        if (TacsRealPart(res_norm) < target_res_norm) { break; }
    }

    // now do load increments and Newton solves here
    // ---------------------------------------------

    for (int iload = 0; iload < 100; iload++) {
        
        // update the new load factor
        lambda += delta_lambda;
        
        // compute and factor the new stiffness matrix
        assembler->setTemperatures(lambda * temperature);
        assembler->setVariables(vars);
        assembler->assembleJacobian(1.0, 0.0, 0.0, res, kmat, TACS_MAT_NORMAL, 1.0, lambda);

        // outer load increment printout
        if (ksm_print) {
            char line[256];
            sprintf(line, "Outer iteration %3d: t: %9.4f lambda/lamLin %9.4f\n",
                    iload, MPI_Wtime() - t0, TacsRealPart(lambda/linear_eigval));
            ksm_print->print(line);
        }

        // do new Newton solve
        for (int inewton = 0; inewton < 100; inewton++) {
            // update nonlinear residual
            assembler->setTemperatures(lambda * temperature); // prob don't need to reset this, but still fine
            assembler->setVariables(vars);
            assembler->assembleRes(res, 1.0, lambda);
            // no load control effects so no res->axpy(-lambda, f)

            // solve K_t(u,lambda') * du = - R(u,lambda'), then do the update u' = u + du
            gmres->solve(res, update); // get -du here
            vars->axpy(-1.0, update);

            // report the residual R(u,lambda')
            TacsScalar res_norm = res->norm();
            if (ksm_print) { // print residual norm of prelim Newton solve
                char line[256];
                sprintf(line, "\t%5d %9.4f %10.4e\n", inewton + 1, MPI_Wtime() - t0,
                        TacsRealPart(res_norm));
                ksm_print->print(line);
            }

            // check for convergence
            if (inewton == 0) {
                res_norm_init = res_norm;
                target_res_norm = rtol * TacsRealPart(res_norm_init) + atol;
            }
            if (TacsRealPart(res_norm) < target_res_norm) { break; }

            // if starting to have trouble to converge, lower delta_lambda and lambda and reduce rtol
            if (inewton > 30 && inewton % 10 == 0) {
                lambda -= delta_lambda / 2;
                delta_lambda *= 0.5;
                rtol = 1e-6; // lower rtol and the step size in load factor
                target_res_norm = rtol * TacsRealPart(res_norm_init) + atol;

                // compute and factor the new stiffness matrix
                assembler->setTemperatures(lambda * temperature);
                assembler->setVariables(vars);
                assembler->assembleJacobian(1.0, 0.0, 0.0, res, kmat, TACS_MAT_NORMAL, 1.0, lambda);
            }

        } // end of newton iteration for each load step

        // update load-displacement curve
        ElementType etype = TACS_BEAM_OR_SHELL_ELEMENT;
        // get min and max stresses for the load-displacement curve on sigma_11
        assembler->getMaxStresses(etype, &tempStresses[0], 0);
        max_stress = tempStresses[0];

        assembler->getMinStresses(etype, &tempStresses[0], 0);
        min_stress = tempStresses[0];

        assembler->getAverageStresses(etype, &tempStresses[0], 0);
        avg_stress = abs(TacsRealPart(tempStresses[0]));

        // TODO : adding slope check
        if (iload == 0) {
            init_slope = avg_stress / lambda;
        } else {
            c_slope = (avg_stress - avg_stress_old) / delta_lambda;
        }

        // save old avg stress for next time
        avg_stress_old = avg_stress;

        // update load-displacement curve output file
        if (fp) {
            // iter, |u|, lambda/lamLin, minS11, avgS11, maxS11, avgSlopeFrac
            fprintf(fp, "%2d,%15.6e,%15.6e,%15.6e,%15.6e,%15.6e,%15.6e\n", iload+1, TacsRealPart(vars->norm()), TacsRealPart(lambda / linear_eigval),
            TacsRealPart(min_stress), TacsRealPart(avg_stress), TacsRealPart(max_stress), TacsRealPart(c_slope / init_slope));
            fflush(fp);
        }

        // store old max stress, min stress
        max_stress_old = max_stress;
        min_stress_old = min_stress;
        avg_stress_old = avg_stress;
        

        // write out file from this arc length step
        // assembler->setVariables(vars); // set orig values back in
        std::string filename = "_buckling/therm-nlbuckle" + std::to_string(iload) + ".f5";
        const char *cstr_filename = filename.c_str();
        f5->writeToFile(cstr_filename);


    } // end of load increment loop

}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    // Get the rank
    MPI_Comm comm = MPI_COMM_WORLD;

    double t = 0.002;
    double rt = 100;
    double Lr = 2.0;
    double temperature = 1.0; // K (may have to adjust depending on the)
    double E = 70e9; // can also try 70e5 since only scales problem
    double conv_slope_frac = 0.1;
    // worried that if I scale down too much, won't solve as deeply though.

    // just perfect geometry in this case, can add in imperfections later maybe
    const int NUM_IMP = 3;
    TacsScalar imperfections[NUM_IMP] = {0.0 * t, 0.0, 0.0 };
    // TacsScalar imperfections[NUM_IMP] = {0.5 * t, 0.0, 0.0 };
    // TacsScalar imperfections[NUM_IMP] = {0.0, 0.0, 0.0 * t }; // not doing imperfections this time
    bool useEigvals = false; // use load-disp curve for thermal
    int nelems = 20000; // 20000
    std::string filePrefix = "";
    TacsScalar nasaKDF, tacsKDF;

    // attempt to make the BC more realistic and get rid of BL effects..
    bool urStarBC = false;

    bool ringStiffened = false;
    double ringStiffenedRadiusFrac = 0.9;

    // tolerances
    // double rtol = 1e-6, atol = 1e-10; // default
    double rtol = 1e-8, atol = 1e-10; // tighter convergence

    runNonlinearStatic(
        comm, t, rt, Lr, E, temperature,
        nelems, conv_slope_frac,
        rtol, atol,
        urStarBC,
        ringStiffened, ringStiffenedRadiusFrac,
        NUM_IMP, imperfections,
        &nasaKDF, &tacsKDF
    );

    MPI_Finalize();

    return 0;
}