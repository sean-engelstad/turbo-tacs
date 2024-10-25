#include "TACSAssembler.h"
#include "TACSCreator.h"
#include "TACSElementAlgebra.h"
#include "TACSElementVerification.h"
#include "TACSIsoShellConstitutive.h"
#include "TACSShellElementDefs.h"
#include "TACSToFH5.h"
#include "tacslapack.h"
#include "../therm-cylinder-include/createCylinderDispControl.h"

// """
// Code adapted from TACS example:
// examples/shell/cylinder.cpp
// since caps2tacs mesh generation isn't the best right now
// """



int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  // Get the rank
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank;
  MPI_Comm_rank(comm, &rank);

  // main inputs
  int nelems = 20000;
  double E = 70e9; // Pa
  double Lr = 2.0;
  double rt = 100.0;
  double t = 0.002; // mm
  TacsScalar temperature = 1.0; // K
  bool ringStiffened = true;
  double ringStiffenedRadiusFrac = 0.9;

  // solve mesh size, geom size
  double R = t * rt; // m
  double L = R * Lr;
  double udisp = 0.0; // ( for r/t = 25 )to be most accurate want udisp about 1/200 to 1/300 the linear buckling disp
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

  // amount of radial / hoop expansion
  double urStar = (1+nu) * cte * temperature * R;
  bool urStarBC = true; // whether to do ur=ur* BC for the hoop ends

  // TacsScalar axis[] = {1.0, 0.0, 0.0};
  // TACSShellTransform *transform = new TACSShellRefAxisTransform(axis);
  TACSShellTransform *transform = new TACSShellNaturalTransform();
  TACSShellConstitutive *con = new TACSIsoShellConstitutive(props, t);

  TACSAssembler *assembler = NULL;
  TACSCreator *creator = NULL;
  TACSElement *shell = NULL;
  // needs to be nonlinear here otherwise solve will terminate immediately
  shell = new TACSQuad4Shell(transform, con); 
  shell->incref();
  createAssembler(
    comm, 2, nx, ny,
    L, R,
    true, urStar,
    false, 0.9,
    shell, &assembler, &creator
  );  

  assembler->incref();
  creator->incref();

  // Free the creator object
  creator->decref();

  // Create matrix and vectors
  TACSBVec *ans = assembler->createVec();  // displacements and rotations
  TACSBVec *res = assembler->createVec();  // The residual
  TACSSchurMat *mat = assembler->createSchurMat();  // stiffness matrix

  // Increment reference count to the matrix/vectors
  ans->incref();
  res->incref();
  mat->incref();

  // Allocate the factorization
  int lev = 10000;
  double fill = 10.0;
  int reorder_schur = 1;
  TACSSchurPc *pc = new TACSSchurPc(mat, lev, fill, reorder_schur);
  pc->incref();

  assembler->setTemperatures(temperature); // 10 K

  // apply displacement control BCs to the residual
  assembler->applyBCs(res);

  assembler->assembleJacobian(1.0, 0.0, 0.0, res, mat);
  pc->factor();  // LU factorization of stiffness matrix
  pc->applyFactor(res, ans);

  ans->scale(-1.0);
  assembler->setVariables(ans);

  // Output for visualization
  ElementType etype = TACS_BEAM_OR_SHELL_ELEMENT;
  int write_flag = (TACS_OUTPUT_NODES | TACS_OUTPUT_CONNECTIVITY |
                    TACS_OUTPUT_DISPLACEMENTS | TACS_OUTPUT_STRAINS |
                    TACS_OUTPUT_STRESSES | TACS_OUTPUT_EXTRAS);
  TACSToFH5 *f5 = new TACSToFH5(assembler, etype, write_flag);
  f5->incref();
  f5->writeToFile("cylinder_solution.f5");

  shell->decref();
  assembler->decref();

  MPI_Finalize();
}
