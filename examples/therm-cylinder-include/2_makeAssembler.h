#include "TACSMaterialProperties.h"
#include "TACSShellElementTransform.h"
#include "1_createCylinderDispControl.h"
#include "TACSShellElementDefs.h"
#include "TACSIsoShellConstitutive.h"

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