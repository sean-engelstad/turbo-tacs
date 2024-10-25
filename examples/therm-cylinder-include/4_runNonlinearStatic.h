#include "2_makeAssembler.h"
#include "3_linearBuckling.h"
#include "KSM.h"
#include "TACSContinuation.h"
#include "TACSToFH5.h"

void runNonlinearStatic(MPI_Comm comm, 
    double t, double rt, double Lr, 
    double E, double temperature, 
    int nelems, double conv_slope_frac,
    double rtol, double atol,
    bool urStarBC, bool runLinearEigval,
    bool ringStiffened, double ringStiffenedRadiusFrac,
    int NUM_IMP, TacsScalar *imperfection_sizes,
    double *linear_eigval, TacsScalar *lambdaNL
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
    double lambda_adjustment;
    if (runLinearEigval) {
        *linear_eigval = linearBuckling(assembler, kmat, pc, NUM_IMP, imperfection_sizes);
        lambda_adjustment = *linear_eigval / 200.0;
    } else {lambda_adjustment = *linear_eigval;}
    
    // temperature *= linear_eigval / 200.0; // good rule is if the linear buckling is at 200 about zero disps, then do delta_lambda = 5.0

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
        if (runLinearEigval) {
            fp = fopen("load-disp-perfect.csv", "w");
        } else {
            fp = fopen("load-disp-imperfect.csv", "w");
        }
        
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

    // do new Newton solve
    int max_num_restarts = 4;
    int num_restarts = 0;

    for (int iload = 0; iload < 300; iload++) {
        
        // update the new load factor
        lambda += delta_lambda;
        
        // compute and factor the new stiffness matrix
        assembler->setTemperatures(lambda * temperature);
        assembler->setVariables(vars);
        assembler->assembleJacobian(1.0, 0.0, 0.0, res, kmat, TACS_MAT_NORMAL, 1.0, lambda);

        old_vars->copyValues(vars);

        // outer load increment printout
        if (ksm_print) {
            char line[256];
            sprintf(line, "Outer iteration %3d: t: %9.4f lambda/lamLin %9.4f\n",
                    iload, MPI_Wtime() - t0, TacsRealPart(lambda / *linear_eigval));
            ksm_print->print(line);
        }

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
            if ( TacsRealPart(res_norm) >= 1e4 && inewton >= 5 ) {
                lambda -= delta_lambda / 2;
                delta_lambda *= 0.5;
                rtol *= 10.0;
                // rtol = 1e-6; // lower rtol and the step size in load factor
                target_res_norm = rtol * TacsRealPart(res_norm_init) + atol;
                vars->copyValues(old_vars);
                num_restarts++;

                // compute and factor the new stiffness matrix
                assembler->setTemperatures(lambda * temperature);
                assembler->setVariables(vars);
                assembler->assembleJacobian(1.0, 0.0, 0.0, res, kmat, TACS_MAT_NORMAL, 1.0, lambda);
            }

            if (num_restarts > max_num_restarts) {
                break;
            }

        } // end of newton iteration for each load step

        if (num_restarts > max_num_restarts) {
            printf("failed to converge at load step %d", iload);
            if (fp) {
                fprintf(fp, "failed to converge at load step %d\n", iload);
            }
        }

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
            init_slope = min_stress / lambda;
        } else {
            c_slope = (min_stress - min_stress_old) / delta_lambda;
        }

        // save old min stress for next time
        min_stress_old = min_stress;

        // update load-displacement curve output file
        if (fp) {
            // iter, |u|, lambda/lamLin, minS11, avgS11, maxS11, avgSlopeFrac
            fprintf(fp, "%2d,%15.6e,%15.6e,%15.6e,%15.6e,%15.6e,%15.6e\n", iload+1, TacsRealPart(vars->norm()), TacsRealPart(lambda / *linear_eigval),
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

        if (c_slope < conv_slope_frac && iload > 10) {
            // significant drop in stiffness and slope to buckling..
            *lambdaNL = TacsRealPart(lambda / *linear_eigval);
            break; // exit the nonlinear solve loop
        }
    } // end of load increment loop

    // write final nonlinear mode to a file
    if ( runLinearEigval ) {
        // assume perfect case
        f5->writeToFile("nlstatic-perfect.f5");
    } else {
        // assume imperfect case
        f5->writeToFile("nlstatic-imperfect.f5");
    }
    
}