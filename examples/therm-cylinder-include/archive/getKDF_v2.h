#pragma once
#include "createCylinderDispControl.h"
#include "TACSShellElementTransform.h"
#include "TACSMaterialProperties.h"
#include "TACSIsoShellConstitutive.h"
#include "TACSShellElementDefs.h"
#include "TACSBuckling.h"
#include "KSM.h"
#include "TACSContinuation.h"

void getNonlinearBucklingKDF(MPI_Comm comm, int run, 
    std::string filePrefix,
    double t, double rt, double Lr, 
    double E, double temperature,
    double conv_eigval, double conv_slope_frac,
    bool ringStiffened, double ringStiffenedRadiusFrac,
    int num_imperfections, TacsScalar *imperfection_sizes,
    bool useEigvals, int nelems,
    TacsScalar *nasaKDF, TacsScalar *tacsKDF
    ) {

    int rank;
    MPI_Comm_rank(comm, &rank);

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

    // TacsScalar axis[] = {1.0, 0.0, 0.0};
    // TACSShellTransform *transform = new TACSShellRefAxisTransform(axis);
    TACSShellTransform *transform = new TACSShellNaturalTransform();
    TACSShellConstitutive *con = new TACSIsoShellConstitutive(props, t);

    TACSAssembler *assembler = NULL;
    TACSCreator *creator = NULL;
    TACSElement *shell = NULL;
    // needs to be nonlinear here otherwise solve will terminate immediately
    shell = new TACSQuad4NonlinearShell(transform, con); 
    shell->incref();
    createAssembler(comm, 2, nx, ny, udisp, L, R, 
    ringStiffened, ringStiffenedRadiusFrac,
    shell, &assembler, &creator);
    
    // set the temperatures into the structure
    // TacsScalar temperature = 1.0;
    assembler->setTemperatures(temperature);

    // Create the design vector
    TACSBVec *x = assembler->createDesignVec();
    x->incref();

    // Get the design variable values
    assembler->getDesignVars(x);

    // Create matrix and vectors
    TACSBVec *u0 = assembler->createVec();  // displacements and rotations
    TACSBVec *f = assembler->createVec();    // loads
    u0->incref();
    f->incref();

    // create zero loads
    TacsScalar *force_vals;
    int size = f->getArray(&force_vals);
    memset(force_vals, 0.0, size * sizeof(TacsScalar));
    assembler->applyBCs(f);

    // nonlinear static
    // --------------------------------------------
    TACSSchurMat *kmat = assembler->createSchurMat();  // stiffness matrix
    kmat->incref();

    // Allocate the factorization
    int lev = 1e6;
    double fill = 10.0;
    int reorder_schur = 1;
    TACSSchurPc *pc = new TACSSchurPc(kmat, lev, fill, reorder_schur);
    pc->incref();

    int subspaceSize = 10, nrestarts = 15, isFlexible = 0;
    GMRES *gmres = new GMRES(kmat, pc, subspaceSize, nrestarts, isFlexible);
    gmres->incref();
    gmres->setTolerances(1e-12, 1e-12);

    // make a KSM print object for solving buckling
    KSMPrint *ksm_print = new KSMPrintStdout("NonlinearStatic", 0, 10);
    ksm_print->incref();

    // Create an TACSToFH5 object for writing output to files
    int write_flag = (TACS_OUTPUT_CONNECTIVITY | TACS_OUTPUT_NODES |
                        TACS_OUTPUT_DISPLACEMENTS | TACS_OUTPUT_STRAINS |
                        TACS_OUTPUT_STRESSES | TACS_OUTPUT_EXTRAS);
    TACSToFH5 *f5 = new TACSToFH5(assembler, TACS_BEAM_OR_SHELL_ELEMENT, write_flag);
    f5->incref();

    // build the TACS linear buckling analysis object
    // which we will use to check for buckling in the nonlinear load-displacement curve (u,lambda)
    // also use the linear eigenmodes as geometric imperfections in the cylinder
    // ---------------------------------------------------------------------------------------
    // ---------------------------------------------------------------------------------------
    
    // create the matrices for buckling
    TACSSchurMat *gmat = assembler->createSchurMat();  // geometric stiffness matrix
    TACSSchurMat *aux_mat = assembler->createSchurMat();  // auxillary matrix for shift and invert solver

    // optional other preconditioner settings?
    assembler->assembleMatType(TACS_STIFFNESS_MATRIX, kmat);
    assembler->assembleMatType(TACS_GEOMETRIC_STIFFNESS_MATRIX, gmat);

    subspaceSize = 10;
    nrestarts = 15; 
    isFlexible = 0;
    GMRES *lbuckle_gmres = new GMRES(aux_mat, pc, subspaceSize, nrestarts, isFlexible);
    lbuckle_gmres->incref();
    lbuckle_gmres->setTolerances(1e-12, 1e-12);

    // make the buckling solver
    TacsScalar sigma = 10.0; // need high enough num_eigvals to get it right
    int max_lanczos_vecs = 300, num_eigvals = 100; // num_eigvals = 50;
    double eig_tol = 1e-12;

    TACSLinearBuckling *buckling = new TACSLinearBuckling(assembler, sigma,
                     gmat, kmat, aux_mat, lbuckle_gmres, max_lanczos_vecs, num_eigvals, eig_tol);
    buckling->incref();

    // make a KSM print object for solving buckling
    KSMPrint *ksm_print_buckling = new KSMPrintStdout("BucklingAnalysis", 0, 10);
    ksm_print->incref();

    // solve the buckling analysis
    buckling->setSigma(10.0);
    buckling->solve(NULL, NULL, ksm_print_buckling);

    // compute linear eigval based on initial thermal buckling estimate
    TacsScalar error;
    TacsScalar linear_eigval = buckling->extractEigenvalue(0, &error);

    // then adjust init temperature so that the linear eigenvalue will be ~200
    temperature *= TacsRealPart(linear_eigval) / 200.0;
    assembler->setTemperatures(temperature);

    // now run and get new linear eigval and then use this for imperfections
    buckling->solve(NULL, NULL, ksm_print_buckling);
    linear_eigval = buckling->extractEigenvalue(0, &error);

    
    // write principal eigenmode
    std::string fileL = filePrefix + "lin_buckle" + std::to_string(run) + ".f5";
    const char *cstr_fileL = fileL.c_str();
    TACSBVec *phi = assembler->createVec();
    phi->incref();
    TacsScalar error1;
    int iphi;
    double max_imp = 0.0; // get argmax of imperfection sizes
    for (int i = 0; i < num_imperfections; i++) {
        if (TacsRealPart(imperfection_sizes[i]) > max_imp) {
            max_imp = TacsRealPart(imperfection_sizes[i]);
            iphi = i;
        }
    }
    // temp do 2 for thermal buckling
    buckling->extractEigenvector(iphi, phi, &error1);
    assembler->setVariables(phi); 
    f5->writeToFile(cstr_fileL);
    // reset setVars
    assembler->zeroVariables();
    // exit(0);

    // choose imperfection sizes for the cylinder based on the cylinder thickness

    // apply the first few eigenmodes as geometric imperfections to the cylinder
    // TACSBVec *phi = assembler->createVec();
    TACSBVec *xpts = assembler->createNodeVec();
    TACSBVec *phi_uvw = assembler->createNodeVec();
    assembler->getNodes(xpts);
    phi->incref();
    xpts->incref();
    phi_uvw->incref();    
    for (int imode = 0; imode < num_imperfections; imode++) {
        buckling->extractEigenvector(imode, phi, &error);

        // copy the phi for all 6 shell dof into phi_uvw
        // how to copy every 3 out of 6 values from 
        TacsScalar *phi_x, *phi_uvw_x;
        int varSize = phi->getArray(&phi_x);
        int nodeSize = phi_uvw->getArray(&phi_uvw_x);
        int ixpts = 0;
        double max_uvw = 0.0;
        for (int iphi = 0; iphi < varSize; iphi++) {
            int idof = iphi % 6;
            if (idof > 2) { // skip rotx, roty, rotz components of eigenvector
                continue;
            }
            phi_uvw_x[ixpts] = phi_x[iphi];
            ixpts++;
            double abs_disp = abs(TacsRealPart(phi_x[iphi]));
            if (abs_disp > max_uvw) {
                max_uvw = abs_disp;
            }
        }
        
        // normalize the mode by the max uvw disp
        for (int i = 0; i < ixpts; i++) {
            phi_uvw_x[i] /= max_uvw;
        }

        xpts->axpy(imperfection_sizes[imode], phi_uvw); 
        // xpts->axpy(imperfection_sizes[imode] * 100.0, phi_uvw); 
    }
    assembler->setNodes(xpts);
    phi->decref();
    xpts->decref();
    phi_uvw->decref();
    // previously wrote out to f5 here in order to see the geometric imperfection

    // end of TACS linear buckling analysis for the geometric imperfections
    // ---------------------------------------------------------------------------------------
    
    // start the nonlinear static analysis, with linear buckling used to check for nonlinear buckling
    // ---------------------------------------------------------------------------------------
    // ---------------------------------------------------------------------------------------

    // begin writing out to output file
    FILE *fp;
    if (rank == 0) {
        std::string file1 = filePrefix + "nl_buckling" + std::to_string(run) + ".out";
        const char *cstr_file1 = file1.c_str();
        fp = fopen(cstr_file1, "w");

        if (fp) {
            fprintf(fp, "$ Nonlinear mechanical buckling of cylinder : t = %10.3e, r/t = %10.3e, L/r = %10.3e, nelems=%d\n", t, rt, Lr, nelems);
            fprintf(fp, "iter, lambda,         |u|/lambda,     dlambda_ds,     loc_eigval,     error,          LIN_buckle,     pred_NL_buckle\n");
            fflush(fp);
        }
    }

    // write another file for plotting load-displacement curve
    FILE *fp2;
    if (rank == 0) {
        std::string file2 = filePrefix + "load-disp" + std::to_string(run) + ".csv";
        const char *cstr_file2 = file2.c_str();
        fp2 = fopen(cstr_file2, "w");
        if (fp2) {
            fprintf(fp2, "iter,|u|,lambda,minS11,avgS11,maxS11,avgSlopeFrac\n");
            fflush(fp2);
        }
    }

    // NOTE : the following is modified code from TACSContinuation.cpp
    // but with linear buckling checks to determine when nonlinear buckling occurs
    // it may also include check of the max disp vs load factor curve (1st or 2nd derivatives of lambda(u))

    // nonlinear static important input settings
    // ------------------------------------------------------------
    double lambda_init = TacsRealPart(linear_eigval) * 0.05;
    double target_delta_lambda = 5.0;
    int max_continuation_iters = 800; // default 300 // for prelim Newton solve to lambda_init
    int max_correction_iters = 100; // for the arc length method static regime
    int max_correction_restarts = 2; // restart for nonlinear static regime
    double correction_rtol = 1e-7; // this needs improvement and an atol (since already fairly low)
    double correction_atol = 1e-10;
    double correction_dtol = 1e3; // if divergence tolerance is huge it's failing to solve and breaks out prelim newton loop
    double krylov_rtol = 1e-6; // krylov refers to the second solve where it reaches more severe loss of stability
    double krylov_atol = 1e-10; 
    double tangent_rtol = 1e-6; // for prelim newton solve section
    double tangent_atol = 1e-10;
    int num_arclength_per_lbuckle = 10; // 5 is default // num arc length steps for each linear buckling check
    // for some reason SEP solver can't drive eigenvalue below zero with shift and invert (needs work)
    bool hit_buckling = false;
    TacsScalar nonlinear_eigval;

    // prelim build objects
    // ------------------------------------------------------------

    TACSBVec *vars = assembler->createVec();
    TACSBVec *old_vars = assembler->createVec();
    TACSBVec *delta_vars = assembler->createVec();
    TACSBVec *temp = assembler->createVec();
    TACSBVec *tangent = assembler->createVec();
    TACSBVec *update = assembler->createVec();
    TACSBVec *res = assembler->createVec();

    TACSContinuationPathMat *path_mat = new TACSContinuationPathMat(kmat, f, tangent, 0.0);
    path_mat->incref();

    TacsScalar lambda = lambda_init;          // The load factor
    TacsScalar lambda_old = 0.0;      // The previous load factor
    TacsScalar target_delta_r = 0.0;  // Target change in r = (u - u_k)^T(u - u_k)
    TacsScalar max_stress, max_stress_old = -1.0e40;
    TacsScalar min_stress, min_stress_old = -1.0e40; // the abs min stresses, etc.
    TacsScalar avg_stress, avg_stress_old = -1.0e40;
    TacsScalar *tempStresses = new TacsScalar[9];
    TacsScalar init_slope, c_slope;

    double t0 = MPI_Wtime();
    if (ksm_print) {
      char line[256];
      ksm_print->print("Performing initial Newton iterations\n");
      sprintf(line, "%5s %9s %10s\n", "Iter", "t", "|R|");
      ksm_print->print(line);
    }

    // Initial newton solve to prelim convergence of (u0, lambda0)
    // -----------------------------------------------------------

    // Compute and factor the tangent stiffness matrix
    assembler->setTemperatures(lambda * temperature);
    assembler->assembleJacobian(1.0, 0.0, 0.0, res, kmat, TACS_MAT_NORMAL, 1.0, lambda);
    pc->factor();
    gmres->setTolerances(tangent_rtol, tangent_atol);
    gmres->solve(f, tangent); // compute tangent update du to reach first equilibrium state
    vars->axpy(lambda, tangent); 
    TacsScalar res_norm_init = 1.0;
    double target_res_norm;

    // Newton solve until convergence to (u0, lambda0)
    for (int k = 0; k < max_continuation_iters; k++) {
        // update the nonlinear residaul
        assembler->setTemperatures(lambda * temperature);
        assembler->setVariables(vars);
        assembler->assembleRes(res, 1.0, lambda); // lambda input here scales the disp control BCs with load factor too
        res->axpy(-lambda, f); // add in load control effects

        TacsScalar res_norm = res->norm();
        if (ksm_print) { // print residual norm of prelim Newton solve
            char line[256];
            sprintf(line, "%5d %9.4f %10.4e\n", k + 1, MPI_Wtime() - t0,
                    TacsRealPart(res_norm));
            ksm_print->print(line);
        }
        if (k == 0) {
            res_norm_init = res_norm;
            // TODO : we should probably add absolute tolerances into the TACSContinuation.cpp solver
            // and/or a TACSNonlinearBuckling.cpp solver too
            target_res_norm = correction_rtol * TacsRealPart(res_norm_init) + correction_atol;
        }
        if (TacsRealPart(res_norm) < target_res_norm) {
            break;
        }

        // solve for the update and new state variables u
        gmres->solve(res, update);
        vars->axpy(-1.0, update);
    }

    // done with prelim newton solve for (u0, lambda0)
    // ------------------------------------------------
    
    // arc-length continuation step for each new (u,lambda) point
    // ----------------------------------------------------------
    int num_failures = 0;
    TacsScalar eigval = 0.0, loc_error = 0.0;
    TacsScalar sigma_init = 1.0;

    TacsScalar dlambda_ds = 0.0; 
    for (int iarclength = 0; !hit_buckling && iarclength < max_continuation_iters; iarclength++) {
        // new arclength step => update the residual, etc.
        assembler->setTemperatures(lambda * temperature);
        assembler->setVariables(vars);
        assembler->assembleJacobian(1.0, 0.0, 0.0, res, kmat, TACS_MAT_NORMAL, 1.0, lambda);
        res->axpy(-lambda, f); // add in load control effects
        pc->factor();
        gmres->setTolerances(tangent_rtol, tangent_atol);
        TacsScalar delta_s = 1.0;

        // adjustments to lambda step for next arclength step
        if (iarclength == 0) {
            gmres->setOperators(kmat, pc);
            gmres->solve(f, tangent);
            // TODO : need to fix this for disp control
            TacsScalar tnorm = tangent->norm();
            dlambda_ds = 1.0 / sqrt(1.0 + tnorm * tnorm);
            tangent->scale(dlambda_ds);
        } else {
            // do I need to adjust the disp control here?
            // dlambda_ds is just 1.0
            gmres->setOperators(path_mat, pc);
            kmat->mult(tangent, res); // res = -(Kmat * tangent - load * dlambda_ds)
            res->axpy(-dlambda_ds, f);
            res->scale(-1.0);
            path_mat->resetConstraint(dlambda_ds);
            gmres->solve(res, temp);
            dlambda_ds += path_mat->extract(temp);
            tangent->axpy(1.0, temp);
        }

        if (ksm_print) {
            char line[256];
            sprintf(line, "Outer iteration %3d: t: %9.4f dp_ds: %10.4e\n",
                    iarclength, MPI_Wtime() - t0, TacsRealPart(dlambda_ds));
            ksm_print->print(line);
            sprintf(line, "%5s %9s %10s %10s %10s\n", "Iter", "t", "|R|", "lambda",
                    "|u|");
            ksm_print->print(line);
        }

        // perform correction iterations until you return to (u,lambda) equillibrium
        old_vars->copyValues(vars);
        lambda_old = lambda;
        gmres->setTolerances(krylov_rtol, krylov_atol);
        int fail_flag = 1;
        int nrestarts = 0;
        for (; fail_flag && (nrestarts < max_correction_restarts); nrestarts++) {
        // Perform an update based on the calculated value of ds
        // This ensures that the step lenght constraint is satisfied
        vars->axpy(delta_s, tangent);
        lambda = lambda + dlambda_ds * delta_s;

        // Set the ksm to use the path_mat object
        gmres->setOperators(path_mat, pc);

        // Now compute the next iteration
        TacsScalar init_res_norm = 0.0;
        for (int j = 0; j < max_correction_iters; j++) {
            assembler->setTemperatures(lambda * temperature);
            assembler->setVariables(vars);
            assembler->assembleRes(res, 1.0, lambda);
            res->axpy(-lambda, f);
            TacsScalar res_norm = res->norm();
            if (ksm_print) {
            char line[256];
            sprintf(line, "%5d %9.4f %10.3e %10.3e %10.3e\n", j, MPI_Wtime() - t0,
                    TacsRealPart(res_norm), TacsRealPart(lambda),
                    TacsRealPart(vars->norm()));
            ksm_print->print(line);
            }

            // Set the initial norm or check the rtol/dtol
            if (j == 0) {
            init_res_norm = res_norm;
            target_res_norm = correction_rtol * TacsRealPart(res_norm_init) + correction_atol;
            } else if (TacsRealPart(res_norm) < target_res_norm) {
            fail_flag = 0;
            break;
            } else if (TacsRealPart(res_norm) >
                    correction_dtol * TacsRealPart(init_res_norm)) {
            break;
            }

            path_mat->resetConstraint(dlambda_ds);
            gmres->solve(res, temp);

            TacsScalar delta_lambda = path_mat->extract(temp);
            lambda = lambda - delta_lambda;
            vars->axpy(-1.0, temp);
        } // end of correction iterations

        // The corrector has failed. Try again with a smaller step size.
        if (fail_flag) {
            vars->copyValues(old_vars);
            lambda = lambda_old;
            delta_s = 0.5 * delta_s;

            if (ksm_print) {
            char line[256];
            sprintf(line,
                    "Failed to converge, retrying with step size = %10.3e\n",
                    TacsRealPart(delta_s));
            ksm_print->print(line);
            }
        }
        } // end of correction restarts

        if (fail_flag) {
            num_failures++;
            if (num_failures >= 3) {
                printf("Failed to converge anymore.. stopped at lambda = %.5f\n", TacsRealPart(lambda));
                break; // exit the loop
            }
        }
        
        // perform eigenvalue checks
        // TODO : fix this to set lambda into the linear buckling problem and do linear static
        if (iarclength % num_arclength_per_lbuckle == 0 && iarclength != 0 && useEigvals) {
            assembler->setBCs(vars);
            assembler->setBCs(old_vars);
            delta_vars->copyValues(vars);
            delta_vars->axpy(-1.0, old_vars); // delta_u(s) = u(s+ds) - u(s) [change in vars along load path for two equilib points]
            buckling->setSigma(1.0);

            // tell Abaqus the temp change on the path
            assembler->setTemperaturePaths((lambda - lambda_old) * temperature);

            // option 1 - finds K_t(u) + lambda * |du| * d/ds K_t(u + s * du / |du|) 
            //     but then if |du| is much smaller than |u|, the lambda does not predict multiples on original lambda correctly
            buckling->solve_local(NULL, vars, ksm_print_buckling, delta_vars);
            eigval = buckling->extractEigenvalue(0, &loc_error);

            if (TacsRealPart(eigval) < 10.0 && useEigvals) {
                // now set more rapid buckling checking near the final eigenvalue
                num_arclength_per_lbuckle = 3;
            }

            if (!fail_flag && fp) { // not failed write out to file
                // prediction for 
                // pred_lambda_NL = lambda * eigval
                TacsScalar pred_lambda_NL = lambda + eigval * lambda * delta_vars->norm() / vars->norm();
                TacsScalar unorm_over_lambda = vars->norm() / lambda; // shows geometric NL stiffening
                fprintf(fp, "%2d %15.6e %15.6e %15.6e %15.6e %15.6e %15.6e %15.6e \n", iarclength + 1, 
                    TacsRealPart(lambda), TacsRealPart(unorm_over_lambda), TacsRealPart(dlambda_ds),
                    TacsRealPart(eigval), TacsRealPart(loc_error), TacsRealPart(linear_eigval), 
                    TacsRealPart(pred_lambda_NL));
                fflush(fp);
            }
            if (fail_flag && fp) {
                // report failed deformation / solve
                fprintf(fp, "failed solve.. with residual %.8e\n", TacsRealPart(res->norm()));
                fflush(fp);
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
        if (iarclength == 0 && !useEigvals) {
            init_slope = avg_stress / lambda;
        } else if (!useEigvals) {
            c_slope = (avg_stress - avg_stress_old) / (lambda - lambda_old);
        }

        // save old avg stress for next time
        avg_stress_old = avg_stress;

        // update load-displacement curve output file
        if (fp2) {
            // iter, |u|, lambda, minS11, avgS11, maxS11, avgSlopeFrac
            fprintf(fp2, "%2d,%15.6e,%15.6e,%15.6e,%15.6e,%15.6e,%15.6e\n", iarclength+1, TacsRealPart(vars->norm()), TacsRealPart(lambda),
            TacsRealPart(min_stress), TacsRealPart(avg_stress), TacsRealPart(max_stress), TacsRealPart(c_slope / init_slope));
            fflush(fp2);
        }

        // store old max stress, min stress
        max_stress_old = max_stress;
        min_stress_old = min_stress;
        avg_stress_old = avg_stress;
        

        // write out file from this arc length step
        assembler->setVariables(vars); // set orig values back in
        std::string filename = "_buckling/therm-nlbuckle" + std::to_string(iarclength) + ".f5";
        const char *cstr_filename = filename.c_str();
        f5->writeToFile(cstr_filename);

        // eigenvalue based stopping criterion
        if (iarclength % num_arclength_per_lbuckle == 0 && iarclength != 0 && useEigvals) {
            printf("eigval = %.8e\n", TacsRealPart(eigval));
            if (TacsRealPart(eigval) < conv_eigval || TacsRealPart(loc_error) > 1e-1 || std::isnan(TacsRealPart(loc_error))) {
                nonlinear_eigval = lambda;
                break; // break out of the arc length loop and we are done with nonlinear buckling!
            }
        }

        // load-disp curve stopping criterion
        if (TacsRealPart(c_slope / init_slope) < conv_slope_frac && iarclength > 0 && !useEigvals) {
            // buckling
            nonlinear_eigval = lambda;
            break;
        }

    } // end of outer arc length steps for loop


    // end of nonlinear static analysis for nonlinear buckling
    // ---------------------------------------------------------------------------------------

    // compute the KDF
    TacsScalar KDF = nonlinear_eigval / linear_eigval;
    printf("LIN eigval %.5e, NL eigval %.5e\n", TacsRealPart(linear_eigval), TacsRealPart(nonlinear_eigval));
    printf("KDF %.5e\n", TacsRealPart(KDF));

    // compare with literature KDF based on r/t from NASA SP-8007
    // TacsScalar rt = R / t;
    TacsScalar kdf_phi = 1.0/16.0 * sqrt(rt);
    TacsScalar experimental_KDF = 1.0 - 0.901 * (1.0 - exp(-kdf_phi));

    if (fp) {
        fprintf(fp, "soln:: LIN_eigval %15.6e NL_eigval %15.6e \n", TacsRealPart(linear_eigval),
            TacsRealPart(nonlinear_eigval));
        fprintf(fp, "\t r/t %15.4e, numerical KDF %15.6e, NASA SP-8007 KDF %15.6e \n", 
            TacsRealPart(rt), TacsRealPart(KDF), TacsRealPart(experimental_KDF));
        fflush(fp);
    }

    // write final solution to f5
    std::string file2 = filePrefix + "nl_buckling" + std::to_string(run) + ".f5";
    const char *cstr_file2 = file2.c_str();
    f5->writeToFile(cstr_file2);
    f5->decref();

    *nasaKDF = experimental_KDF;
    *tacsKDF = KDF;
}