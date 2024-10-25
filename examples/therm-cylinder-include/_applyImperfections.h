void applyImperfections(TACSAssembler *assembler, TACSLinearBuckling *buckling, int NUM_IMP, TacsScalar *imperfection_sizes) {
    // apply the first few eigenmodes as geometric imperfections to the cylinder
    TACSBVec *phi = assembler->createVec();
    TACSBVec *xpts = assembler->createNodeVec();
    TACSBVec *phi_uvw = assembler->createNodeVec();
    assembler->getNodes(xpts);
    phi->incref();
    xpts->incref();
    phi_uvw->incref();    
    TacsScalar error;
    for (int imode = 0; imode < NUM_IMP; imode++) {
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
}