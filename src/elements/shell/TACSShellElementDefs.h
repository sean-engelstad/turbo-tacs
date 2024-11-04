#ifndef TACS_SHELL_ELEMENT_DEFS_H
#define TACS_SHELL_ELEMENT_DEFS_H

#include "TACSDirector.h"
#include "TACSShellElement.h"
#include "TACSShellElementModel.h"
#include "TACSShellElementQuadBasis.h"

// for verification
#include "TACSShellElement_orig.h"

/*
  Linear shell elements with appropriate quadrature schemes
*/
typedef TACSShellElement<TACSQuadLinearQuadrature, TACSShellQuadBasis<2>,
                         TACSLinearizedRotation, TACSShellLinearModel>
    TACSQuad4Shell;

// typedef TACSShellElementOrig<TACSQuadLinearQuadrature, TACSShellQuadBasis<2>,
//                          TACSLinearizedRotation, TACSShellLinearModel>
//     TACSQuad4ShellOrig;

// typedef TACSShellElement<TACSQuadQuadraticQuadrature, TACSShellQuadBasis<3>,
//                          TACSLinearizedRotation, TACSShellLinearModel>
//     TACSQuad9Shell;

// typedef TACSShellElement<TACSQuadCubicQuadrature, TACSShellQuadBasis<4>,
//                          TACSLinearizedRotation, TACSShellLinearModel>
//     TACSQuad16Shell;

/*
  Shell elements with a linearized rotation and nonlinear strain expressions
*/
typedef TACSShellElement<TACSQuadLinearQuadrature, TACSShellQuadBasis<2>,
                         TACSLinearizedRotation, TACSShellNonlinearModel>
    TACSQuad4NonlinearShell;

// typedef TACSShellElement<TACSQuadQuadraticQuadrature, TACSShellQuadBasis<3>,
//                          TACSLinearizedRotation, TACSShellNonlinearModel>
//     TACSQuad9NonlinearShell;

// typedef TACSShellElement<TACSQuadCubicQuadrature, TACSShellQuadBasis<4>,
//                          TACSLinearizedRotation, TACSShellNonlinearModel>
//     TACSQuad16NonlinearShell;

// /*
//   Moderate rotation shell elements with appropriate quadrature schemes
// */
// typedef TACSShellElement<TACSQuadLinearQuadrature, TACSShellQuadBasis<2>,
//                          TACSQuadraticRotation, TACSShellLinearModel>
//     TACSQuad4ShellModRot;

// typedef TACSShellElement<TACSQuadQuadraticQuadrature, TACSShellQuadBasis<3>,
//                          TACSQuadraticRotation, TACSShellLinearModel>
//     TACSQuad9ShellModRot;

// typedef TACSShellElement<TACSQuadCubicQuadrature, TACSShellQuadBasis<4>,
//                          TACSQuadraticRotation, TACSShellLinearModel>
//     TACSQuad16ShellModRot;

// /*
//   Moderate rotation shell elements with nonlinear strain and appropriate
//   quadrature schemes
// */
// typedef TACSShellElement<TACSQuadLinearQuadrature, TACSShellQuadBasis<2>,
//                          TACSQuadraticRotation, TACSShellNonlinearModel>
//     TACSQuad4NonlinearShellModRot;

// typedef TACSShellElement<TACSQuadQuadraticQuadrature, TACSShellQuadBasis<3>,
//                          TACSQuadraticRotation, TACSShellNonlinearModel>
//     TACSQuad9NonlinearShellModRot;

// typedef TACSShellElement<TACSQuadCubicQuadrature, TACSShellQuadBasis<4>,
//                          TACSQuadraticRotation, TACSShellNonlinearModel>
//     TACSQuad16NonlinearShellModRot;

/**
  Create a TACS shell element based on the name of the shell.

  @param name The name of the shell element
  @param transform The transformation used for the shell
  @param con The shell constitutive object
*/
inline TACSElement *TacsCreateShellByName(const char *name,
                                          TACSShellTransform *transform,
                                          TACSShellConstitutive *con) {
  TACSElement *shell = NULL;
  // if (strcmp(name, "TACSQuad4ShellModRot") == 0) {
  //   shell = new TACSQuad4ShellModRot(transform, con);
  // } else if (strcmp(name, "TACSQuad9ShellModRot") == 0) {
  //   shell = new TACSQuad9ShellModRot(transform, con);
  // } else if (strcmp(name, "TACSQuad16ShellModRot") == 0) {
  //   shell = new TACSQuad16ShellModRot(transform, con);
  // } else 
  if (strcmp(name, "TACSQuad4Shell") == 0 ||
             strcmp(name, "CQUAD") == 0 || strcmp(name, "CQUAD4") == 0 ||
             strcmp(name, "CQUADR") == 0) {
    shell = new TACSQuad4Shell(transform, con);
  // } else if (strcmp(name, "TACSQuad9Shell") == 0 ||
  //            strcmp(name, "CQUAD9") == 0) {
  //   shell = new TACSQuad9Shell(transform, con);
  // } else if (strcmp(name, "TACSQuad16Shell") == 0) {
  //   shell = new TACSQuad16Shell(transform, con);
  } else if (strcmp(name, "TACSQuad4NonlinearShell") == 0) {
    shell = new TACSQuad4NonlinearShell(transform, con);
  // } else if (strcmp(name, "TACSQuad9NonlinearShell") == 0) {
  //   shell = new TACSQuad9NonlinearShell(transform, con);
  // } else if (strcmp(name, "TACSQuad16NonlinearShell") == 0) {
  //   shell = new TACSQuad16NonlinearShell(transform, con);
  }

  return shell;
}

#endif  // TACS_SHELL_ELEMENT_DEFS_H
