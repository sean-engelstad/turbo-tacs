/*
  This file is part of TACS: The Toolkit for the Analysis of Composite
  Structures, a parallel finite-element code for structural and
  multidisciplinary design optimization.

  Copyright (C) 2010 University of Toronto
  Copyright (C) 2012 University of Michigan
  Copyright (C) 2014 Georgia Tech Research Corporation
  Additional copyright (C) 2010 Graeme J. Kennedy and Joaquim
  R.R.A. Martins All rights reserved.

  TACS is licensed under the Apache License, Version 2.0 (the
  "License"); you may not use this software except in compliance with
  the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0
*/

#ifndef TACS_ISO_SHELL_CONSTITUTIVE_H
#define TACS_ISO_SHELL_CONSTITUTIVE_H

#include "TACSMaterialProperties.h"
#include "TACSShellConstitutive.h"

/**
  This constitutive class defines the stiffness properties for a
  first-order shear deformation theory type element. This class
  is derived from the TACSConstitutive object, but is still
  a pure virtual base class.
*/
class TACSIsoShellConstitutive : public TACSShellConstitutive {
 public:
  TACSIsoShellConstitutive(TACSMaterialProperties *props, TacsScalar _t = 1.0,
                           int _tNum = -1, TacsScalar _tlb = 0.0,
                           TacsScalar _tub = 1.0, TacsScalar _tOffset = 0.0);
  ~TACSIsoShellConstitutive();

  // Retrieve the global design variable numbers
  int getDesignVarNums(int elemIndex, int dvLen, int dvNums[]);

  // Set the element design variable from the design vector
  int setDesignVars(int elemIndex, int dvLen, const TacsScalar dvs[]);

  // Get the element design variables values
  int getDesignVars(int elemIndex, int dvLen, TacsScalar dvs[]);

  // Get the lower and upper bounds for the design variable values
  int getDesignVarRange(int elemIndex, int dvLen, TacsScalar lb[],
                        TacsScalar ub[]);

  // Evaluate the material density
  TacsScalar evalDensity(int elemIndex, const double pt[],
                         const TacsScalar X[]);

  // Add the derivative of the density
  void addDensityDVSens(int elemIndex, TacsScalar scale, const double pt[],
                        const TacsScalar X[], int dvLen, TacsScalar dfdx[]);

  // Evaluate the mass moments
  template <typename T>
  void evalMassMoments(int elemIndex, const T pt[], const T X[],
                       T moments[]);

  // Add the sensitivity of the mass moments
  void addMassMomentsDVSens(int elemIndex, const double pt[],
                            const TacsScalar X[], const TacsScalar scale[],
                            int dvLen, TacsScalar dfdx[]);

  // Evaluate the specific heat
  TacsScalar evalSpecificHeat(int elemIndex, const double pt[],
                              const TacsScalar X[]);

  int symind(int irow, int icol, int N);

  // Evaluate the stress
  void evalStress(int elemIndex, const double pt[], const TacsScalar X[],
                  const TacsScalar strain[], TacsScalar stress[]);

  
  // Evaluate the stress
  template <typename T>
  __DEVICE__ void evalStress_kernel(
        int elemIndex, const T pt[],
        const T X[], const T e[],
        T s[]) {
    if (properties) {
      T A[6], B[6], D[6], As[3], drill;

      // Compute the tangent stiffness matrix
      properties->evalTangentStiffness2D_kernel<T>(A);

      // The bending-stretch coupling matrix is zero in this case
      B[0] = B[1] = B[2] = B[3] = B[4] = B[5] = 0.0;

      // Scale the in-plane matrix and bending stiffness
      // matrix by the appropriate quantities
      T I = t * t * t / 12.0;
      for (int i = 0; i < 6; i++) {
        D[i] = I * A[i];
        A[i] *= t;
        B[i] += -tOffset * t * A[i];
        D[i] += tOffset * tOffset * t * t * A[i];
      }

      // Set the through-thickness shear stiffness
      As[0] = As[2] = (5.0 / 6.0) * A[5];
      As[1] = 0.0;

      drill = 0.5 * DRILLING_REGULARIZATION * (As[0] + As[2]);

      // Evaluate the stress
      computeStress_kernel<T>(A, B, D, As, drill, e, s);
    } else {
      s[0] = s[1] = s[2] = 0.0;
      s[3] = s[4] = s[5] = 0.0;
      s[6] = s[7] = s[8] = 0.0;
    }
  }
                  
  // compute the ABD matrix
  template <typename T>
  void getABDmatrix(int elemIndex, const T pt[],
               const T X[], T ABD[]);

  // Evaluate the tangent stiffness
  void evalTangentStiffness(int elemIndex, const double pt[],
                            const TacsScalar X[], TacsScalar C[]);

  // Add the contribution
  void addStressDVSens(int elemIndex, TacsScalar scale, const double pt[],
                       const TacsScalar X[], const TacsScalar strain[],
                       const TacsScalar psi[], int dvLen, TacsScalar dfdx[]);

  // Calculate the point-wise failure criteria
  TacsScalar evalFailure(int elemIndex, const double pt[], const TacsScalar X[],
                         const TacsScalar e[]);

  // Evaluate the derivative of the failure criteria w.r.t. the strain
  TacsScalar evalFailureStrainSens(int elemIndex, const double pt[],
                                   const TacsScalar X[], const TacsScalar e[],
                                   TacsScalar sens[]);

  // Add the derivative of the failure criteria w.r.t. the design variables
  void addFailureDVSens(int elemIndex, TacsScalar scale, const double pt[],
                        const TacsScalar X[], const TacsScalar strain[],
                        int dvLen, TacsScalar dfdx[]);

  // Evaluate the thermal strain
  void evalThermalStrain(int elemIndex, const double pt[], const TacsScalar X[],
                         TacsScalar theta, TacsScalar strain[]);

  // Evaluate the heat flux, given the thermal gradient
  void evalHeatFlux(int elemIndex, const double pt[], const TacsScalar X[],
                    const TacsScalar grad[], TacsScalar flux[]);

  // Evaluate the tangent of the heat flux
  void evalTangentHeatFlux(int elemIndex, const double pt[],
                           const TacsScalar X[], TacsScalar C[]);

  // Add the derivative of the heat flux
  void addHeatFluxDVSens(int elemIndex, TacsScalar scale, const double pt[],
                         const TacsScalar X[], const TacsScalar grad[],
                         const TacsScalar psi[], int dvLen, TacsScalar dfdx[]);

  // The name of the constitutive object
  const char *getObjectName();

  // Retrieve the design variable for plotting purposes
  TacsScalar evalDesignFieldValue(int elemIndex, const double pt[],
                                  const TacsScalar X[], int index);

 private:
  // Material properties class
  TACSMaterialProperties *properties;

  // Store information about the design variable
  TacsScalar kcorr;  // The shear correction factor
  TacsScalar t, tlb, tub, tOffset;
  int tNum;
  TacsScalar ksWeight;  // ks weight used in failure calc

  // The object name
  static const char *constName;
};

#endif  // TACS_ISO_SHELL_CONSTITUTIVE_H
