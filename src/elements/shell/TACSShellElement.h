// #ifndef TACS_SHELL_ELEMENT_H
// #define TACS_SHELL_ELEMENT_H

#pragma once

#include "TACSDirector.h"
#include "TACSElement.h"
#include "TACSElementAlgebra.h"
#include "TACSElementTypes.h"
#include "TACSElementVerification.h"
#include "TACSShellConstitutive.h"
#include "TACSShellElementModel.h"
#include "TACSShellElementTransform.h"
#include "TACSShellUtilities.h"

// A2D imports for automatic differentiation
#include "adscalar.h"
#include "a2dcore.h"

TacsScalar A2D_VecDot(A2D::Vec<TacsScalar, 3> a, A2D::Vec<TacsScalar, 3> b) {
  TacsScalar myDot = 0.0;
  A2D::VecDot<TacsScalar, 3>(a, b, myDot);
  return myDot;
};

template <class quadrature, class basis, class director, class model>
class TACSShellElement : public TACSElement {
 public:
  // Offset within the solution vector to the rotational
  // parametrization defined via the director class. Here the offset
  // is 3 corresponding to the (u, v, w) displacements of the
  // mid-surface of the shell.
  static const int offset = 3;

  // The number of variables defined at each node of the shell
  // element.  There are 3 mid-surface displacements plus however many
  // parameters are defined by the director class for the
  // parametrization.
  static const int vars_per_node = offset + director::NUM_PARAMETERS;

  // The number of nodes for this element. This is derived from the
  // basis function class. This is just a handy re-definition since
  // this constant is used in many locations within the element.
  static const int num_nodes = basis::NUM_NODES;

  bool complexStepGmatrix = false;
  TacsScalar temperature = 0.0;

  TACSShellElement(TACSShellTransform *_transform,
                   TACSShellConstitutive *_con) {
    transform = _transform;
    transform->incref();

    con = _con;
    con->incref();

    // For linear models, we'll need to switch to a nonlinear implementation to
    // capture geometric effects
    if (typeid(model) == typeid(TACSShellLinearModel)) {
      nlElem = new TACSShellElement<quadrature, basis, director,
                                    TACSShellNonlinearModel>(transform, con);
    }
    // For nonlinear models we can use the current class instance
    else {
      nlElem = this;
    }
  }

  // new default constructor for the shell element needed for GPU.
  TACSShellElement() {
    transform = nullptr;
    con = nullptr;
    nlElem = nullptr;
  };

  ~TACSShellElement() {
    if (transform) {
      transform->decref();
    }

    if (con) {
      con->decref();
    }

    // free nonlinear element pointer
    if (nlElem != this) {
      delete nlElem;
    }
  }

  const char *getObjectName() { return "TACSShellElement"; }

  int getVarsPerNode() { return vars_per_node; }
  int getNumNodes() { return num_nodes; }
  void setTemperature(TacsScalar _temp) {temperature = _temp;}
  TacsScalar getTemperature() { return temperature; }

  template <class Transform>
  __HOST_DEVICE__ Transform* getTransform() {return static_cast<Transform*>(transform);}

  template <class Constitutive>
  __HOST_DEVICE__ Constitutive* getConstitutive() {return static_cast<Constitutive*>(con);}

  ElementLayout getLayoutType() { return basis::getLayoutType(); }

  ElementType getElementType() { return TACS_BEAM_OR_SHELL_ELEMENT; }

  int getNumQuadraturePoints() { return quadrature::getNumQuadraturePoints(); }

  double getQuadratureWeight(int n) {
    return quadrature::getQuadratureWeight(n);
  }

  void setComplexStepGmatrix(bool complexStepFlag) {
    complexStepGmatrix = complexStepFlag;
#ifndef TACS_USE_COMPLEX  // real mode
    printf(
        "Warning : the routine setComplexStepGmatrix on shell elements doesn't "
        "do anything in real mode.");
#endif  // TACS_USE_COMPLEX
  };

  bool getComplexStepGmatrix() { return complexStepGmatrix; };

  double getQuadraturePoint(int n, double pt[]) {
    return quadrature::getQuadraturePoint(n, pt);
  }

  int getNumElementFaces() { return quadrature::getNumElementFaces(); }

  int getNumFaceQuadraturePoints(int face) {
    return quadrature::getNumFaceQuadraturePoints(face);
  }

  double getFaceQuadraturePoint(int face, int n, double pt[],
                                double tangent[]) {
    return quadrature::getFaceQuadraturePoint(face, n, pt, tangent);
  }

  int getDesignVarNums(int elemIndex, int dvLen, int dvNums[]) {
    return con->getDesignVarNums(elemIndex, dvLen, dvNums);
  }

  int setDesignVars(int elemIndex, int dvLen, const TacsScalar dvs[]) {
    return con->setDesignVars(elemIndex, dvLen, dvs);
  }

  int getDesignVars(int elemIndex, int dvLen, TacsScalar dvs[]) {
    return con->getDesignVars(elemIndex, dvLen, dvs);
  }

  int getDesignVarRange(int elemIndex, int dvLen, TacsScalar lb[],
                        TacsScalar ub[]) {
    return con->getDesignVarRange(elemIndex, dvLen, lb, ub);
  }

  void computeEnergies(int elemIndex, double time, const TacsScalar Xpts[],
                       const TacsScalar vars[], const TacsScalar dvars[],
                       TacsScalar *Te, TacsScalar *Pe);

  void addResidual(int elemIndex, double time, const TacsScalar *Xpts,
                   const TacsScalar *vars, const TacsScalar *dvars,
                   const TacsScalar *ddvars, TacsScalar *res);

  void addJacobian(int elemIndex, double time, TacsScalar alpha,
                   TacsScalar beta, TacsScalar gamma, const TacsScalar Xpts[],
                   const TacsScalar vars[], const TacsScalar dvars[],
                   const TacsScalar ddvars[], TacsScalar res[],
                   TacsScalar mat[]);

  void getMatType(ElementMatrixType matType, int elemIndex, double time,
                  const TacsScalar Xpts[], const TacsScalar vars[],
                  TacsScalar mat[]);

  void addAdjResProduct(int elemIndex, double time, TacsScalar scale,
                        const TacsScalar psi[], const TacsScalar Xpts[],
                        const TacsScalar vars[], const TacsScalar dvars[],
                        const TacsScalar ddvars[], int dvLen,
                        TacsScalar dfdx[]);

  int evalPointQuantity(int elemIndex, int quantityType, double time, int n,
                        double pt[], const TacsScalar Xpts[],
                        const TacsScalar vars[], const TacsScalar dvars[],
                        const TacsScalar ddvars[], TacsScalar *detXd,
                        TacsScalar *quantity);

  void addPointQuantityDVSens(int elemIndex, int quantityType, double time,
                              TacsScalar scale, int n, double pt[],
                              const TacsScalar Xpts[], const TacsScalar vars[],
                              const TacsScalar dvars[],
                              const TacsScalar ddvars[],
                              const TacsScalar dfdq[], int dvLen,
                              TacsScalar dfdx[]);

  void addPointQuantitySVSens(int elemIndex, int quantityType, double time,
                              TacsScalar alpha, TacsScalar beta,
                              TacsScalar gamma, int n, double pt[],
                              const TacsScalar Xpts[], const TacsScalar vars[],
                              const TacsScalar dvars[],
                              const TacsScalar ddvars[],
                              const TacsScalar dfdq[], TacsScalar dfdu[]);

  void getOutputData(int elemIndex, ElementType etype, int write_flag,
                     const TacsScalar Xpts[], const TacsScalar vars[],
                     const TacsScalar dvars[], const TacsScalar ddvars[],
                     int ld_data, TacsScalar *data);

  void getAverageStresses(int elemIndex, ElementType etype,
                          const TacsScalar Xpts[], const TacsScalar vars[],
                          const TacsScalar dvars[], const TacsScalar ddvars[],
                          TacsScalar *avgStresses);

  // GPU code
  #ifdef __CUDACC__

    template <class Transform, class Constitutive>
    __device__ void addStaticJacobian_kernel(
      int ideriv, int igauss,
      Transform *transform, Constitutive *constitutive,
      double time, TacsScalar alpha, TacsScalar beta, TacsScalar gamma,
      const TacsScalar Xpts[], const TacsScalar vars[],
      TacsScalar res[], TacsScalar mat[]
    );

    // TODO : dynamic kernel
    // __device__ void addDynamicJacobian_kernel(
    //   int ideriv, int igauss,
    //   double time, TacsScalar alpha, TacsScalar beta, TacsScalar gamma,
    //   const TacsScalar Xpts[], const TacsScalar vars[], const TacsScalar dvars[], const TacsScalar ddvars[],
    //   TacsScalar res[], TacsScalar mat[]
    // );

  #endif // __CUDACC__

 private:
  // Set sizes for the different components
  static const int usize = 3 * num_nodes;
  static const int dsize = 3 * num_nodes;
  static const int csize = 9 * num_nodes;

  TACSShellTransform *transform;
  TACSShellConstitutive *con;
  TACSElement *nlElem;
};

/*
  Compute the kinetic and potential energies of the shell
*/
template <class quadrature, class basis, class director, class model>
void TACSShellElement<quadrature, basis, director, model>::computeEnergies(
    int elemIndex, double time, const TacsScalar *Xpts, const TacsScalar *vars,
    const TacsScalar *dvars, TacsScalar *Te, TacsScalar *Ue) {
  // Zero the kinetic and potential energies
  TacsScalar Telem = 0.0;
  TacsScalar Uelem = 0.0;

  // Compute the number of quadrature points
  const int nquad = quadrature::getNumQuadraturePoints();

  // Compute the node normal directions
  TacsScalar fn[3 * num_nodes], Xdn[9 * num_nodes];
  TacsShellComputeNodeNormals<TacsScalar, basis>(Xpts, fn, Xdn);

  // Store information about the transformation and derivatives at each node for
  // the drilling degrees of freedom
  TacsScalar etn[num_nodes];
  TacsScalar XdinvTn[9 * num_nodes], Tn[9 * num_nodes];
  TacsScalar u0xn[9 * num_nodes], Ctn[csize];
  TacsShellComputeDrillStrain<TacsScalar, vars_per_node, offset, basis, director, model>(
      transform, Xdn, fn, vars, XdinvTn, Tn, u0xn, Ctn, etn);

  // Compute the director rates
  TacsScalar d[dsize], ddot[dsize];
  director::template computeDirectorRates<TacsScalar,vars_per_node, offset, num_nodes>(
      vars, dvars, fn, d, ddot);

  // Set the total number of tying points needed for this element
  TacsScalar ety[basis::NUM_TYING_POINTS];
  model::template computeTyingStrain<TacsScalar,vars_per_node, basis>(Xpts, fn, vars, d,
                                                           ety);

  // Loop over each quadrature point and add the residual contribution
  for (int quad_index = 0; quad_index < nquad; quad_index++) {
    // Get the quadrature weight
    double pt[3];
    double weight = quadrature::getQuadraturePoint(quad_index, pt);
    
    // Compute X, X,xi and the interpolated normal n0
    A2D::Mat<TacsScalar, 3, 3> T;
    A2D::Vec<TacsScalar, 3> X, n0;
    A2D::Mat<TacsScalar, 3, 2> Xxi;
    TacsScalar et; // do we need a2d scalar with this drill strain?

    basis::template interpFields<TacsScalar,3, 3>(pt, Xpts, X.get_data());
    basis::template interpFieldsGrad<TacsScalar,3, 3>(pt, Xpts, Xxi.get_data());
    basis::template interpFields<TacsScalar,3, 3>(pt, fn, n0.get_data());
    basis::template interpFields<TacsScalar,1, 1>(pt, etn, &et);

    // Compute the transformation at the quadrature point
    transform->computeTransform(Xxi.get_data(), n0.get_data(), T.get_data());

    // Evaluate the displacement gradient at the point
    A2D::Mat<TacsScalar, 3, 3> XdinvT, XdinvzT, u0x, u1x;
    TacsScalar detXd; // can we setup scalars like this in new vs A2D?
    detXd = TacsShellComputeDispGrad<TacsScalar, vars_per_node, basis>(
        pt, Xpts, vars, fn, d, Xxi.get_data(), n0.get_data(), T.get_data(), 
        XdinvT.get_data(), XdinvzT.get_data(), u0x.get_data(), u1x.get_data());
    detXd *= weight;

    // Evaluate the tying components of the strain
    // gty might not need to be A2D obj here
    // A2D::SymMat<TacsScalar, 3> gty; // The symmetric components of the tying strain
    TacsScalar gty[6];
    basis::interpTyingStrain(pt, ety, gty);

    // Compute the symmetric parts of the tying strain
    // e0ty might not need to be A2D obj here
    // A2D::SymMat<TacsScalar, 3> e0ty;
    TacsScalar e0ty[6];  // e0ty = XdinvT^{T}*gty*XdinvT
    mat3x3SymmTransformTranspose(XdinvT.get_data(), gty, e0ty);

    // Compute the set of strain components
    // based on beam element code => may not need A2D obj for strain
    TacsScalar e[9];  // The components of the strain
    model::evalStrain(u0x.get_data(), u1x.get_data(), e0ty, e);
    e[8] = et;

    // Compute the corresponding stresses
    // based on beam element code => may not need A2D obj for stress
    TacsScalar s[9];
    con->evalStress(elemIndex, pt, X.get_data(), e, s);

    Uelem +=
        0.5 * detXd *
        (s[0] * e[0] + s[1] * e[1] + s[2] * e[2] + s[3] * e[3] + s[4] * e[4] +
         s[5] * e[5] + s[6] * e[6] + s[7] * e[7] + s[8] * e[8]);

    // Evaluate the mass moments
    TacsScalar moments[3];
    con->evalMassMoments(elemIndex, pt, X.get_data(), moments);

    // Compute the velocities and the director velocities
    A2D::Vec<TacsScalar, 3> u0dot, d0dot;
    // TacsScalar u0dot[3], d0dot[3];
    basis::template interpFields<TacsScalar,vars_per_node, 3>(pt, dvars, u0dot.get_data());
    basis::template interpFields<TacsScalar,3, 3>(pt, ddot, d0dot.get_data());

    Telem += 0.5 * detXd * 
              (moments[0] * A2D_VecDot(u0dot, u0dot) +
              2.0 * moments[1] * A2D_VecDot(u0dot, d0dot) +
              moments[2] * A2D_VecDot(d0dot, d0dot));
    // Telem += 0.5 * detXd *
    //          (moments[0] * vec3Dot(u0dot, u0dot) +
    //           2.0 * moments[1] * vec3Dot(u0dot, d0dot) +
    //           moments[2] * vec3Dot(d0dot, d0dot));
  }

  *Te = Telem;
  *Ue = Uelem;
}

/*
  Add the residual to the provided vector
*/
template <class quadrature, class basis, class director, class model>
void TACSShellElement<quadrature, basis, director, model>::addResidual(
    int elemIndex, double time, const TacsScalar Xpts[],
    const TacsScalar vars[], const TacsScalar dvars[],
    const TacsScalar ddvars[], TacsScalar res[]) {
  // Compute the number of quadrature points
  const int nquad = quadrature::getNumQuadraturePoints();

  // Derivative of the director field and matrix at each point
  TacsScalar dd[dsize];
  memset(dd, 0, 3 * num_nodes * sizeof(TacsScalar));

  // Compute the node normal directions
  TacsScalar fn[3 * num_nodes], Xdn[9 * num_nodes];
  TacsShellComputeNodeNormals<TacsScalar,basis>(Xpts, fn, Xdn);

  // Compute the drill strain penalty at each node
  TacsScalar etn[num_nodes], detn[num_nodes];
  memset(detn, 0, num_nodes * sizeof(TacsScalar));

  // Store information about the transformation and derivatives at each node for
  // the drilling degrees of freedom
  TacsScalar XdinvTn[9 * num_nodes], Tn[9 * num_nodes];
  TacsScalar u0xn[9 * num_nodes], Ctn[csize];
  TacsShellComputeDrillStrain<TacsScalar,vars_per_node, offset, basis, director, model>(
      transform, Xdn, fn, vars, XdinvTn, Tn, u0xn, Ctn, etn);

  TacsScalar d[dsize], ddot[dsize], dddot[dsize];
  director::template computeDirectorRates<TacsScalar,vars_per_node, offset, num_nodes>(
      vars, dvars, ddvars, fn, d, ddot, dddot);

  // Compute the tying strain values
  TacsScalar ety[basis::NUM_TYING_POINTS], dety[basis::NUM_TYING_POINTS];
  memset(dety, 0, basis::NUM_TYING_POINTS * sizeof(TacsScalar));
  model::template computeTyingStrain<TacsScalar,vars_per_node, basis>(Xpts, fn, vars, d,
                                                           ety);

  // Loop over each quadrature point and add the residual contribution
  for (int quad_index = 0; quad_index < nquad; quad_index++) {

    // Get the quadrature weight
    double pt[3];
    double weight = quadrature::getQuadraturePoint(quad_index, pt);

    // interpolation section
    // ----------------------------------------

    // passive A2D Objs used in interpolation
    A2D::Vec<TacsScalar,3> X, n0;
    A2D::Mat<TacsScalar,3,2> Xxi, nxi;
    A2D::Mat<TacsScalar,3,3> T;
    // A2D::SymMat<TacsScalar,3> gty;
    
    // active A2D objs used in interpolation
    A2D::ADObj<A2D::Vec<TacsScalar,1>> et;
    A2D::ADObj<A2D::Vec<TacsScalar,3>> d0;
    A2D::ADObj<A2D::Mat<TacsScalar,3,2>> d0xi, u0xi;
    A2D::ADObj<A2D::SymMat<TacsScalar,3>> e0ty, e0ty_tmp, gty;

    // interpolate coordinates, director, midplane displacements with the basis
    basis::template interpFields<TacsScalar,3, 3>(pt, Xpts, X.get_data());
    basis::template interpFields<TacsScalar,3, 3>(pt, fn, n0.get_data());
    basis::template interpFields<TacsScalar,1, 1>(pt, etn, et.value().get_data());
    basis::template interpFields<TacsScalar,3, 3>(pt, d, d0.value().get_data());

    basis::template interpFieldsGrad<TacsScalar,3, 3>(pt, Xpts, Xxi.get_data());
    basis::template interpFieldsGrad<TacsScalar,3, 3>(pt, fn, nxi.get_data());
    basis::template interpFieldsGrad<TacsScalar,3, 3>(pt, d, d0xi.value().get_data());
    basis::template interpFieldsGrad<TacsScalar,vars_per_node, 3>(pt, vars, u0xi.value().get_data());

    basis::interpTyingStrain(pt, ety, gty.value().get_data());

    // setup before A2D strain energy stack
    // ------------------------------------

    // Compute the transformation at the quadrature point
    transform->computeTransform(Xxi.get_data(), n0.get_data(), T.get_data()); 

    // compute ABD matrix from shell theory (prospective)
    // A2D::SymMat<TacsScalar,9> ABD; // normally ABD is 6x6, but this one includes transverse shear and drill strains
    A2D::Mat<TacsScalar,9,9> ABD;
    con->getABDmatrix(0, pt, X.get_data(), ABD.get_data()); // TODO make this routine

    // passive variables for strain energy stack
    A2D::Vec<TacsScalar, 3> zero;
    A2D::ADObj<A2D::Mat<TacsScalar, 3, 3>> Xd, Xdz, Xdinv, XdinvT;

    // active variables for strain energy stack
    A2D::ADObj<TacsScalar> detXd, ES_dot, Uelem;
    A2D::ADObj<A2D::Vec<TacsScalar,9>> E, S;
    A2D::ADObj<A2D::Mat<TacsScalar, 3, 3>> u0x_tmp, u1x_tmp1, u1x_tmp2, u1x_tmp3, u1x_term1, u1x_term2, u1x_sum; // temp variables
    A2D::ADObj<A2D::Mat<TacsScalar, 3, 3>> u0xi_frame, u1xi_frame, u0x, u1x;

    const A2D::MatOp NORMAL = A2D::MatOp::NORMAL, TRANSPOSE = A2D::MatOp::TRANSPOSE;
    const A2D::ShellStrainType STRAIN_TYPE = A2D::ShellStrainType::LINEAR; // if condition on type of model here..

    // TODO : fix order of MatRotateFrame (backwards)
    auto prelim_coord_stack = A2D::MakeStack(
      A2D::ShellAssembleFrame(Xxi, n0, Xd), 
      A2D::ShellAssembleFrame(nxi, zero, Xdz), 
      A2D::MatInv(Xd, Xdinv),
      A2D::MatDet(Xd, detXd),
      A2D::MatMatMult(Xdinv, T, XdinvT)
    ); // auto evaluates on runtime
    // want this to not be included in Hessian/gradient backprop

    // printf("detXd = %.8e\n", detXd.value());
    // for (int i = 0; i < 9; i++) {
    //   printf("Xd[%d] = %.8e\n", i, Xd.value().get_data()[i]);
    //   printf("Xdz[%d] = %.8e\n", i, Xdz.value().get_data()[i]);
    //   printf("Xdinv[%d] = %.8e\n", i, Xdinv.value().get_data()[i]);
    //   printf("XdinvT[%d] = %.8e\n", i, XdinvT.value().get_data()[i]);     
    // }

    // compute the strain energy from d0, d0xi, u0xi
    auto strain_energy_stack = A2D::MakeStack(
      // part 1 - compute shell basis and transform matrices (passive portion)
      A2D::ShellAssembleFrame(u0xi, d0, u0xi_frame),
      A2D::ShellAssembleFrame(d0xi, zero, u1xi_frame),
      // part 2 - compute u0x midplane disp gradient
      A2D::MatMatMult(u0xi_frame, Xdinv, u0x_tmp),
      A2D::MatRotateFrame(T, u0x_tmp, u0x),
      // part 3 - compute u1x director disp gradient
      A2D::MatMatMult(u1xi_frame, Xdinv, u1x_term1),
      // computes u0xi_frame * Xdinv * Xdz * Xdinv => u1x_term2 
      A2D::MatMatMult(u0xi_frame, Xdinv, u1x_tmp1), 
      A2D::MatMatMult(u1x_tmp1, Xdz, u1x_tmp2), 
      A2D::MatMatMult(u1x_tmp2, Xdinv, u1x_term2),
      // compute final u1x = T^T * (u0x * Xdinv - u1x * Xdinv * Xdz * Xdinv) * T
      A2D::MatSum(1.0, u1x_term1, -1.0, u1x_term2, u1x_sum), // for some reason this entry has no hzero?
      A2D::MatRotateFrame(T, u1x_sum, u1x),
      // part 4 - compute transformed tying strain e0ty
      A2D::SymMatRotateFrame(XdinvT, gty, e0ty),
      // part 5 - compute strains, stresses and then strain energy
      A2D::ShellStrain<STRAIN_TYPE>(u0x, u1x, e0ty, et, E),
      A2D::MatVecMult(ABD, E, S),
      // part 6 - compute strain energy
      A2D::VecDot(E, S, ES_dot),
      A2D::Eval(0.5 * weight * detXd * ES_dot, Uelem)
    );

    // reverse mode AD for the strain energy stack
    // -------------------------------------------------
    Uelem.bvalue() = 1.0;
    strain_energy_stack.reverse();

    // reverse through the basis back to the director class, drill strain, tying strain
    basis::template addInterpFieldsTranspose<TacsScalar, 1, 1>(pt, et.bvalue().get_data(), detn);
    basis::template addInterpFieldsTranspose<TacsScalar, 3, 3>(pt, d0.bvalue().get_data(), dd);

    basis::template addInterpFieldsGradTranspose<TacsScalar, 3, 3>(pt, d0xi.bvalue().get_data(), dd);
    basis::template addInterpFieldsGradTranspose<TacsScalar, vars_per_node, 3>(pt, u0xi.bvalue().get_data(), res);

    basis::addInterpTyingStrainTranspose(pt, gty.bvalue().get_data(), dety);

    // setup before kinetic energy stack
    // ------------------------------------

    // passive variables
    A2D::Vec<TacsScalar,3> moments, u0ddot, d0ddot;

    // active variables
    A2D::ADObj<TacsScalar> uu_term, ud_term1, ud_term2, dd_term, dTelem_dt;
    A2D::ADObj<A2D::Vec<TacsScalar,3>> u0dot, d0dot;

    // evaluate mass moments
    con->evalMassMoments(elemIndex, pt, X.get_data(), moments.get_data());
    // interpolate first time derivatives
    basis::template interpFields<TacsScalar,vars_per_node, 3>(pt, dvars, u0dot.value().get_data());
    basis::template interpFields<TacsScalar,3, 3>(pt, ddot, d0dot.value().get_data());
    // interpolate second time derivatives
    basis::template interpFields<TacsScalar,vars_per_node, 3>(pt, ddvars, u0ddot.get_data());
    basis::template interpFields<TacsScalar,3, 3>(pt, dddot, d0ddot.get_data());

    // due to integration by parts, residual is based on dT/dt, time derivative of KE so 1st and 2nd time derivatives used
    //   double check: but Jacobian should be obtained with a cross Hessian d^2(dT/dt)/du0dot/du0ddot (and same for directors d0)
    auto kinetic_energy_stack = A2D::MakeStack(
      A2D::VecDot(u0dot, u0ddot, uu_term),
      A2D::VecDot(u0dot, d0ddot, ud_term1),
      A2D::VecDot(u0ddot, d0dot, ud_term2),
      A2D::VecDot(d0dot, d0ddot, dd_term),
      Eval(detXd * (moments[0] * uu_term + moments[1] * (ud_term1 + ud_term2) + moments[2] * dd_term), dTelem_dt)
    );

    // now reverse to from dTelem_dt => u0dot, d0dot sensitivities
    dTelem_dt.bvalue() = 1.0;
    kinetic_energy_stack.reverse();

    // backpropagate the time derivatives to the residual
    basis::template addInterpFieldsTranspose<TacsScalar,vars_per_node, 3>(pt, u0dot.bvalue().get_data(), res);
    basis::template addInterpFieldsTranspose<TacsScalar,3, 3>(pt, d0dot.value().get_data(), dd);
  }

  // Add the contribution to the residual from the drill strain
  TacsShellAddDrillStrainSens<TacsScalar,vars_per_node, offset, basis, director, model>(
      Xdn, fn, vars, XdinvTn, Tn, u0xn, Ctn, detn, res);

  // Add the contributions from the tying strain
  model::template addComputeTyingStrainTranspose<TacsScalar,vars_per_node, basis>(
      Xpts, fn, vars, d, dety, res, dd);

  // Add the contributions to the director field
  director::template addDirectorResidual<TacsScalar,vars_per_node, offset, num_nodes>(
      vars, dvars, ddvars, fn, dd, res);

  // Add the contribution from the rotation constraint (defined by the
  // rotational parametrization) - if any
  director::template addRotationConstraint<TacsScalar,vars_per_node, offset, num_nodes>(
      vars, res);
}

/*
  Add the residual to the provided vector
*/
template <class quadrature, class basis, class director, class model>
void TACSShellElement<quadrature, basis, director, model>::addJacobian(
    int elemIndex, double time, TacsScalar alpha, TacsScalar beta,
    TacsScalar gamma, const TacsScalar Xpts[], const TacsScalar vars[],
    const TacsScalar dvars[], const TacsScalar ddvars[], TacsScalar res[],
    TacsScalar mat[]) {
  // Compute the number of quadrature points
  const int nquad = quadrature::getNumQuadraturePoints();

  // this implementation uses reverse AD for 1st derivs res = dPi/dU
  // and forward AD for 2nd derivs Kelem = d^2Pi/dU^2
  // worry about transient case later..

  // set input ADScalar types for variables
  const int N = num_nodes * vars_per_node;
  // using T = A2D::ADScalar<TacsScalar,24>;
  using T = A2D::ADScalar<double,24>;

  T U[N], dU[N], ddU[N], resA2D[N], XptsAD[3*num_nodes];

  // TODO : check Kinetic energy part works too
  for (int ivar = 0; ivar < N; ivar++) {
    U[ivar].value = vars[ivar];
    U[ivar].deriv[ivar] = 1.0;  // dUi / dUi = 1.0 (self-derivative only NZ values)
    dU[ivar].value = dvars[ivar];
    ddU[ivar].value = ddvars[ivar];
    ddU[ivar].deriv[ivar] = 1.0;
  }
  for (int inode = 0; inode < 3 * num_nodes; inode++) {
    XptsAD[inode] = T(Xpts[inode]);
  }

  // Derivative of the director field and matrix at each 
  T dd[dsize];

  // Compute the node normal directions
  T fn[3 * num_nodes], Xdn[9 * num_nodes];
  TacsShellComputeNodeNormals<T, basis>(XptsAD, fn, Xdn);

  // Compute the drill strain penalty at each node
  T etn[num_nodes], detn[num_nodes];

  // Store information about the transformation and derivatives at each node for
  // the drilling degrees of freedom
  T XdinvTn[9 * num_nodes], Tn[9 * num_nodes];
  T u0xn[9 * num_nodes], Ctn[csize];
  TacsShellComputeDrillStrain<T, vars_per_node, offset, basis, director, model>(
      transform, Xdn, fn, U, XdinvTn, Tn, u0xn, Ctn, etn);

  T d[dsize], ddot[dsize], dddot[dsize];
  director::template computeDirectorRates<T, vars_per_node, offset, num_nodes>(
      U, dU, ddU, fn, d, ddot, dddot);

  // Compute the tying strain values
  T ety[basis::NUM_TYING_POINTS], dety[basis::NUM_TYING_POINTS];
  model::template computeTyingStrain<T, vars_per_node, basis>(XptsAD, fn, U, d,
                                                           ety);

  // Loop over each quadrature point and add the residual contribution
  // TODO : for GPU version, this for loop will be outside this call
  for (int quad_index = 0; quad_index < nquad; quad_index++) {

    // Get the quadrature weight
    T pt[3];
    T weight = quadrature::template getQuadraturePoint<T>(quad_index, pt);

    // interpolation section
    // ----------------------------------------

    // passive A2D Objs used in interpolation
    A2D::Vec<T,3> X, n0;
    A2D::Mat<T,3,2> Xxi, nxi;
    A2D::Mat<T,3,3> Tmat;
    
    // active A2D objs used in interpolation
    A2D::ADObj<A2D::Vec<T,1>> et;
    A2D::ADObj<A2D::Vec<T,3>> d0;
    A2D::ADObj<A2D::Mat<T,3,2>> d0xi, u0xi;
    A2D::ADObj<A2D::SymMat<T,3>> e0ty, e0ty_tmp, gty;

    // interpolate coordinates, director, midplane displacements with the basis
    basis::template interpFields<T, 3, 3>(pt, XptsAD, X.get_data());
    basis::template interpFields<T, 3, 3>(pt, fn, n0.get_data());
    basis::template interpFields<T, 1, 1>(pt, etn, et.value().get_data());
    basis::template interpFields<T, 3, 3>(pt, d, d0.value().get_data());

    basis::template interpFieldsGrad<T, 3, 3>(pt, XptsAD, Xxi.get_data());
    basis::template interpFieldsGrad<T, 3, 3>(pt, fn, nxi.get_data());
    basis::template interpFieldsGrad<T, 3, 3>(pt, d, d0xi.value().get_data());
    basis::template interpFieldsGrad<T, vars_per_node, 3>(pt, U, u0xi.value().get_data());

    basis::template interpTyingStrain<T>(pt, ety, gty.value().get_data());

    // setup before A2D strain energy stack
    // ------------------------------------

    // Compute the transformation at the quadrature point
    transform->computeTransform<T>(Xxi.get_data(), n0.get_data(), Tmat.get_data()); 

    // compute ABD matrix from shell theory (prospective)
    A2D::Mat<T,9,9> ABD;
    con->getABDmatrix<T>(0, pt, X.get_data(), ABD.get_data()); // TODO make this routine

    // passive variables for strain energy stack
    A2D::Vec<T, 3> zero;
    A2D::ADObj<A2D::Mat<T, 3, 3>> Xd, Xdz, Xdinv, XdinvT;

    // active variables for strain energy stack
    A2D::ADObj<T> detXd, ES_dot, Uelem;
    A2D::ADObj<A2D::Vec<T,9>> E, S;
    A2D::ADObj<A2D::Mat<T, 3, 3>> u0x_tmp, u1x_tmp1, u1x_tmp2, u1x_tmp3, u1x_term1, u1x_term2, u1x_sum; // temp variables
    A2D::ADObj<A2D::Mat<T, 3, 3>> u0xi_frame, u1xi_frame, u0x, u1x;

    const A2D::MatOp NORMAL = A2D::MatOp::NORMAL, TRANSPOSE = A2D::MatOp::TRANSPOSE;
    const A2D::ShellStrainType STRAIN_TYPE = A2D::ShellStrainType::LINEAR; // if condition on type of model here..

    // TODO : fix order of MatRotateFrame (backwards)
    auto prelim_coord_stack = A2D::MakeStack(
      A2D::ShellAssembleFrame(Xxi, n0, Xd), 
      A2D::ShellAssembleFrame(nxi, zero, Xdz), 
      A2D::MatInv(Xd, Xdinv),
      A2D::MatDet(Xd, detXd),
      A2D::MatMatMult(Xdinv, Tmat, XdinvT)
    ); // auto evaluates on runtime
    // want this to not be included in Hessian/gradient backprop

    // printf("detXd = %.8e\n", detXd.value());
    // for (int i = 0; i < 9; i++) {
    //   printf("Xd[%d] = %.8e\n", i, Xd.value().get_data()[i]);
    //   printf("Xdz[%d] = %.8e\n", i, Xdz.value().get_data()[i]);
    //   printf("Xdinv[%d] = %.8e\n", i, Xdinv.value().get_data()[i]);
    //   printf("XdinvT[%d] = %.8e\n", i, XdinvT.value().get_data()[i]);     
    // }

    using Trev = A2D::ADObj<T>;

    // compute the strain energy from d0, d0xi, u0xi
    auto strain_energy_stack = A2D::MakeStack(
      // part 1 - compute shell basis and transform matrices (passive portion)
      A2D::ShellAssembleFrame(u0xi, d0, u0xi_frame),
      A2D::ShellAssembleFrame(d0xi, zero, u1xi_frame),
      // part 2 - compute u0x midplane disp gradient
      A2D::MatMatMult(u0xi_frame, Xdinv, u0x_tmp),
      A2D::MatRotateFrame(Tmat, u0x_tmp, u0x),
      // part 3 - compute u1x director disp gradient
      A2D::MatMatMult(u1xi_frame, Xdinv, u1x_term1),
      // computes u0xi_frame * Xdinv * Xdz * Xdinv => u1x_term2 
      A2D::MatMatMult(u0xi_frame, Xdinv, u1x_tmp1), 
      A2D::MatMatMult(u1x_tmp1, Xdz, u1x_tmp2), 
      A2D::MatMatMult(u1x_tmp2, Xdinv, u1x_term2),
      // compute final u1x = T^T * (u0x * Xdinv - u1x * Xdinv * Xdz * Xdinv) * T
      A2D::MatSum(1.0, u1x_term1, -1.0, u1x_term2, u1x_sum), // for some reason this entry has no hzero?
      A2D::MatRotateFrame(Tmat, u1x_sum, u1x),
      // part 4 - compute transformed tying strain e0ty
      A2D::SymMatRotateFrame(XdinvT, gty, e0ty),
      // part 5 - compute strains, stresses and then strain energy
      A2D::ShellStrain<STRAIN_TYPE>(u0x, u1x, e0ty, et, E),
      A2D::MatVecMult(ABD, E, S),
      // part 6 - compute strain energy
      A2D::VecDot(E, S, ES_dot),
      A2D::Eval(Trev(T(0.5) * weight) * detXd * ES_dot, Uelem)
    );

    // reverse mode AD for the strain energy stack
    // -------------------------------------------------
    Uelem.bvalue() = T(1.0);
    strain_energy_stack.reverse();

    // reverse through the basis back to the director class, drill strain, tying strain
    basis::template addInterpFieldsTranspose<T, 1, 1>(pt, et.bvalue().get_data(), detn);
    basis::template addInterpFieldsTranspose<T, 3, 3>(pt, d0.bvalue().get_data(), dd);

    basis::template addInterpFieldsGradTranspose<T, 3, 3>(pt, d0xi.bvalue().get_data(), dd);
    basis::template addInterpFieldsGradTranspose<T, vars_per_node, 3>(pt, u0xi.bvalue().get_data(), resA2D);

    basis::template addInterpTyingStrainTranspose<T>(pt, gty.bvalue().get_data(), dety);

    // setup before kinetic energy stack
    // ------------------------------------

    // passive variables
    A2D::Vec<T,3> moments;
    A2D::ADObj<A2D::Vec<T,3>> u0ddot, d0ddot; // had to make no longer passive just for forward scalar..

    // active variables
    A2D::ADObj<T> uu_term, ud_term1, ud_term2, dd_term, dTelem_dt;
    A2D::ADObj<A2D::Vec<T,3>> u0dot, d0dot;

    // evaluate mass moments
    con->evalMassMoments<T>(elemIndex, pt, X.get_data(), moments.get_data());
    // interpolate first time derivatives
    basis::template interpFields<T, vars_per_node, 3>(pt, dU, u0dot.value().get_data());
    basis::template interpFields<T, 3, 3>(pt, ddot, d0dot.value().get_data());
    // interpolate second time derivatives
    basis::template interpFields<T, vars_per_node, 3>(pt, ddU, u0ddot.value().get_data());
    basis::template interpFields<T, 3, 3>(pt, dddot, d0ddot.value().get_data());

    // due to integration by parts, residual is based on dT/dt, time derivative of KE so 1st and 2nd time derivatives used
    //   double check: but Jacobian should be obtained with a cross Hessian d^2(dT/dt)/du0dot/du0ddot (and same for directors d0)
    auto kinetic_energy_stack = A2D::MakeStack(
      A2D::VecDot(u0dot, u0ddot, uu_term),
      A2D::VecDot(u0dot, d0ddot, ud_term1),
      A2D::VecDot(u0ddot, d0dot, ud_term2),
      A2D::VecDot(d0dot, d0ddot, dd_term),
      Eval(detXd * (Trev(moments[0]) * uu_term + Trev(moments[1]) * (ud_term1 + ud_term2) + Trev(moments[2]) * dd_term), dTelem_dt)
    );

    // now reverse to from dTelem_dt => u0dot, d0dot sensitivities
    dTelem_dt.bvalue() = T(1.0);
    kinetic_energy_stack.reverse();

    // backpropagate the time derivatives to the residual
    basis::template addInterpFieldsTranspose<T, vars_per_node, 3>(pt, u0dot.bvalue().get_data(), resA2D);
    basis::template addInterpFieldsTranspose<T, 3, 3>(pt, d0dot.bvalue().get_data(), dd);
  }

  // Add the contribution to the residual from the drill strain
  TacsShellAddDrillStrainSens<T, vars_per_node, offset, basis, director, model>(
      Xdn, fn, U, XdinvTn, Tn, u0xn, Ctn, detn, resA2D);

  // Add the contributions from the tying strain
  model::template addComputeTyingStrainTranspose<T, vars_per_node, basis>(
      XptsAD, fn, U, d, dety, resA2D, dd);

  // Add the contributions to the director field
  director::template addDirectorResidual<T, vars_per_node, offset, num_nodes>(
      U, dU, ddU, fn, dd, resA2D);

  // Add the contribution from the rotation constraint (defined by the
  // rotational parametrization) - if any
  director::template addRotationConstraint<T, vars_per_node, offset, num_nodes>(
      U, resA2D);

  // use forward AD to compute Kelem = dres/dU (TODO : transient case and kinetic energy matrix also)
  if (mat) {
    for (int ivar = 0; ivar < N; ivar++) {
      res[ivar] = resA2D[ivar].value;
      for (int jvar = 0; jvar < N; jvar++) {
        mat[N * ivar + jvar] = resA2D[ivar].deriv[jvar];
      }
    }
  }
  // end of addJacobian!
}

#ifdef __CUDACC__

// should this be a static method ? I think not, but we do need to ensure each warp
// calls the same kernel addJacobian method (for same elementType) I think.

// or maybe each warp will do one of 24 columns of Kelem for same element and some fraction
// of the gauss indices and we will do it that way..
template <class quadrature, class basis, class director, class model>
template <class Transform, class Constitutive>
__device__
void TACSShellElement<quadrature, basis, director, model>::addStaticJacobian_kernel(
  int ideriv, int igauss,
  Transform *transform, Constitutive *constitutive,
  double time, TacsScalar alpha, TacsScalar beta, TacsScalar gamma,
  const TacsScalar Xpts[], const TacsScalar vars[],
  TacsScalar* res, TacsScalar *mat
) {

  // this implementation uses reverse AD for 1st derivs res = dPi/dU
  // and forward AD for 2nd derivs Kelem = d^2Pi/dU^2
  // worry about transient case later..

  // set input ADScalar types for variables
  const int N = num_nodes * vars_per_node;
  // instead of storing 24 derivatives just store one of them with 
  // ideriv indicating which derivative of residual to compute
  using T = A2D::ADScalar<double,1>;

  // be careful with how many floats you can store on each thread..
  // NOTE : took out dynamics for now since can't go over 255 doubles per thread (so only doing statics)
  // TODO : write Telem (dynamic stiffness matrix computation on separate kernel?)
  T U[N], resA2D[N], XptsAD[3*num_nodes];
  // T dU[N], ddU[N];

  // TODO : check Kinetic energy part works too
  for (int ivar = 0; ivar < N; ivar++) {
    U[ivar] = T(vars[ivar]);
    resA2D[ivar] = T(0.0);
    // printf("U[%d] = %.8f, deriv = %.8f\n", ivar, U[ivar].value, U[ivar].deriv[0]);
  }
  // instead of if statement in for loop, do this to prevent branch statement problems which would
  // slow down the thread
  // since only storing one derivative, seed this one as the only nonzero one (all others are zero)
  U[ideriv].deriv[0] = 1.0;
  
  for (int inode = 0; inode < 3 * num_nodes; inode++) {
    XptsAD[inode] = T(Xpts[inode]);
    // printf("Xpts[%d] = %.8f, deriv = %.8f\n", inode, XptsAD[inode].value, XptsAD[inode].deriv[0]);
  }

  // Derivative of the director field and matrix at each 
  T dd[dsize];
  for (int i = 0; i < dsize; i++) {
    dd[i] = T(0.0);
  }

  // TODO : make some of these TacsScalar and not T type to remove amount of data on the thread later
  // TODO : check if we can directly mix between T and TacsScalar type

  // Compute the node normal directions
  T fn[3 * num_nodes], Xdn[9 * num_nodes];
  TacsShellComputeNodeNormals<T, basis>(XptsAD, fn, Xdn);
  
  // for (int inode = 0; inode < 3 * num_nodes; inode++) {
  //   printf("fn[%d] = %.8f, deriv = %.8f\n", inode, fn[inode].value, fn[inode].deriv[0]);
  //   printf("Xdn[%d] = %.8f, deriv = %.8f\n", inode, Xdn[inode].value, Xdn[inode].deriv[0]);
  // }

  // Compute the drill strain penalty at each node
  T etn[num_nodes], detn[num_nodes];
  for (int inode = 0; inode < num_nodes; inode++) {
    detn[inode] = T(0.0);
  }

  // Store information about the transformation and derivatives at each node for
  // the drilling degrees of freedom
  // TODO : may want to also do interpolation in this method as well to reduce load on each thread..
  T XdinvTn[9 * num_nodes], Tn[9 * num_nodes];
  T u0xn[9 * num_nodes], Ctn[csize];
  TacsShellComputeDrillStrain_kernel<T, vars_per_node, offset, basis, director, model, Transform>(
      transform, Xdn, fn, U, XdinvTn, Tn, u0xn, Ctn, etn);

  // for (int inode = 0; inode < num_nodes; inode++) {
  //   printf("etn[%d] = %.8f, deriv = %.8f\n", inode, etn[inode].value, etn[inode].deriv[0]);
  // }

  T d[dsize];
  // T ddot[dsize], dddot[dsize]; // comment out for just static part to reduce load on each thread
  director::template computeDirectorRates<T, vars_per_node, offset, num_nodes>(U, fn, d);

  // for (int i = 0; i < dsize; i++) {
  //   printf("d[%d] = %.8f, deriv = %.8f\n", i, d[i].value, d[i].deriv[0]);
  // }

  // Compute the tying strain values
  T ety[basis::NUM_TYING_POINTS], dety[basis::NUM_TYING_POINTS];
  model::template computeTyingStrain<T, vars_per_node, basis>(XptsAD, fn, U, d, ety);

  // for (int i = 0; i < basis::NUM_TYING_POINTS; i++) {
  //   printf("ety[%d] = %.8f, deriv = %.8f\n", i, ety[i].value, ety[i].deriv[0]);
  // }
  // return;

  // beginning of what was gauss pt loop
  // ------------------------------------
  // now done outside the kernel call for lighter weight kernel

  { // scope block for local variables in what was gauss pt loop
  // TODO : can this be changed back to TacsScalar pt[3] type here? and fix the templates for that
  // Get the quadrature weight
  T pt[3];
  T weight = quadrature::template getQuadraturePoint<T>(igauss, pt);

  // for (int ipt = 0; ipt < 3; ipt++) {
  //   printf("ipt[%d] = %.8e\n", ipt, pt[ipt].value);
  // }

  // Compute X, X,xi and the interpolated normal n0
  T X[3], Xxi[6], n0[3], Tmat[9], et;
  basis::template interpFields<T, 3, 3>(pt, XptsAD, X);
  basis::template interpFieldsGrad<T, 3, 3>(pt, XptsAD, Xxi);
  basis::template interpFields<T, 3, 3>(pt, fn, n0);
  basis::template interpFields<T, 1, 1>(pt, etn, &et);

  // for (int i = 0; i < 6; i++) {
  //   printf("Xxi[%d] = %.8e\n", i, Xxi[i].value);
  // }

  // Compute the transformation at the quadrature point
  // for (int i = 0; i < 9; i++) {
  //   Tmat[i] = T(0.0);
  // }
  transform->template computeTransform<T>(Xxi, n0, Tmat);

  // for (int i = 0; i < 9; i++) {
  //   printf("Tmat[%d] = %.8e\n", i, Tmat[i].value);
  // }

  // Evaluate the displacement gradient at the point
  T XdinvT[9], XdinvzT[9];
  T u0x[9], u1x[9];
  T detXd = TacsShellComputeDispGrad<T, vars_per_node, basis>(
      pt, XptsAD, U, fn, d, Xxi, n0, Tmat, XdinvT, XdinvzT, u0x, u1x);
  detXd *= weight;

  // for (int i = 0; i < 9; i++) {
  //   printf("u0x[%d] = %.8e, deriv = %.8e\n", i, u0x[i].value, u0x[i].deriv[0]);
  //   printf("u1x[%d] = %.8e, deriv = %.8e\n", i, u1x[i].value, u1x[i].deriv[0]);
  // }

  // Evaluate the tying components of the strain
  T gty[6];  // The symmetric components of the tying strain
  basis::template interpTyingStrain<T>(pt, ety, gty);

  // for (int i = 0; i < 6; i++) {
  //   printf("ety[%d] = %.8e\n", i, ety[i].value);
  // }

  // Compute the symmetric parts of the tying strain
  T e0ty[6];  // e0ty = XdinvT^{T}*gty*XdinvT
  mat3x3SymmTransformTranspose<T>(XdinvT, gty, e0ty);

  // for (int i = 0; i < 6; i++) {
  //   printf("e0ty[%d] = %.8e\n", i, e0ty[i].value);
  // }

  // Compute the set of strain components
  T e[9];  // The components of the strain
  model::template evalStrain<T>(u0x, u1x, e0ty, e);
  e[8] = et;

  // for (int i = 0; i < 9; i++) {
  //   printf("e[%d] = %.8e\n", i, e[i].value);
  // }

  // Compute the corresponding stresses
  T s[9];
  for (int i = 0; i < 9; i++) {
    s[i] = 0.0; // initialize stress
  }
  // this is not working right now.. need to change this somehow, can't access properties
  // constitutive->template evalStress_kernel<T>(0, pt, X, e, s);
  // maybe get ABD matrix to do this?

  for (int i = 0; i < 9; i++) {
    printf("s[%d] = %.8e, deriv = %.8e\n", i, s[i].value, s[i].deriv[0]);
  }

  // Compute the derivative of the product of the stress and strain
  // with respect to u0x, u1x and e0ty
  T du0x[9], du1x[9], de0ty[6];
  for (int i = 0; i < 9; i++) {
    du0x[i] = du1x[i] = de0ty[i] = T(0.0);
  }

  model::template evalStrainSens<T>(detXd, s, u0x, u1x, du0x, du1x, de0ty);

  // printf("detXd = %.8e\n", detXd);
  // for (int i = 0; i < 9; i++) {
  //   printf("du0x[%d] = %.8e\n", i, du0x[i].value);
  //   printf("du1x[%d] = %.8e\n", i, du1x[i].value);
  //   printf("de0ty[%d] = %.8e\n", i, de0ty[i].value);
  // }

  // Add the contribution to the drilling strain
  T det = detXd * s[8];
  basis::template addInterpFieldsTranspose<T, 1, 1>(pt, &det, detn);

  // printf("det = %.8e\n", det.value);
  // for (int inode = 0; inode < num_nodes; inode++) {
  //   printf("detn[%d] = %.8e\n", inode, detn[inode].value);
  // }

  // Add the contributions to the residual from du0x, du1x and dCt
  TacsShellAddDispGradSens<T, vars_per_node, basis>(pt, Tmat, XdinvT, XdinvzT, du0x,
                                                  du1x, resA2D, dd);

  // for (int i = 0; i < 9; i++) {
  //   printf("resA2D[%d] = %.8e\n", i, resA2D[i].value);
  //   printf("dd[%d] = %.8e\n", i, dd[i].value);
  // }

  // Compute the of the tying strain w.r.t. derivative w.r.t. the coefficients
  T dgty[6];
  mat3x3SymmTransformTransSens<T>(XdinvT, de0ty, dgty);

  // for (int i = 0; i < 6; i++) {
  //   printf("dgty[%d] = %.8e\n", i, dgty[i].value);
  // }

  // Evaluate the tying strain
  for (int i = 0; i < 6; i++) {
    dety[i] = T(0.0);
  }
  basis::template addInterpTyingStrainTranspose<T>(pt, dgty, dety);

  // for (int i = 0; i < 6; i++) {
  //   printf("dety[%d] = %.8e\n", i, dety[i].value);
  // }
  
  } // scope block for local variables
  // end of what was the gauss pt loop call (now done outside kernel)

  // for (int i = 0; i < N; i++) {
  //   printf("0 - resA2D[%d] = %.8e\n", i, resA2D[i].value);
  // }

  // Add the contribution to the residual from the drill strain
  TacsShellAddDrillStrainSens<T, vars_per_node, offset, basis, director, model>(
      Xdn, fn, U, XdinvTn, Tn, u0xn, Ctn, detn, resA2D);

  // for (int i = 0; i < N; i++) {
  //   printf("1 - resA2D[%d] = %.8e\n", i, resA2D[i].value);
  // }

  // Add the contributions from the tying strain
  model::template addComputeTyingStrainTranspose<T, vars_per_node, basis>(
      XptsAD, fn, U, d, dety, resA2D, dd);

  // for (int i = 0; i < N; i++) {
  //   printf("2 - resA2D[%d] = %.8e\n", i, resA2D[i].value);
  // }

  // Add the contributions to the director field
  // director::template addDirectorResidual<T, vars_per_node, offset, num_nodes>(
  //     U, dU, ddU, fn, dd, resA2D);
  director::template addDirectorResidual<T, vars_per_node, offset, num_nodes>(
      U, fn, dd, resA2D);

  // for (int i = 0; i < N; i++) {
  //   printf("3 - resA2D[%d] = %.8e\n", i, resA2D[i].value);
  // }

  // Add the contribution from the rotation constraint (defined by the
  // rotational parametrization) - if any
  director::template addRotationConstraint<T, vars_per_node, offset, num_nodes>(
      U, resA2D);

  // for (int i = 0; i < N; i++) {
  //   printf("4 - resA2D[%d] = %.8e\n", i, resA2D[i].value);
  // }
  
  // res[0] = 0.0;
  // return; // debug

  // use forward AD to compute Kelem = dres/dU (TODO : transient case and kinetic energy matrix also)
  if (mat) {
    // printf("mat exists!\n");

    for (int ivar = 0; ivar < N; ivar++) {
      res[ivar] = resA2D[ivar].value;
      
      // write in the ideriv column of the matrix
      // but we store Kelem^T since sym and easier to prevent race conditions for __shared__ memory access
      // mem[N*ivar+ideriv] => transposed => mat[N*ideriv + ivar] with pointer from mat[N*ideriv] is mat[ivar] below
      mat[ivar] = resA2D[ivar].deriv[0];
      // printf("res[%d] = %.8f\n", ivar, res[ivar]);
      // printf("mat[%d] = %.8f\n", ivar, mat[ivar]);
    }
  }

  // printf("shellElement kernel : checkpoint end\n", checkpoint++);
}

#endif // __CUDACC__

template <class quadrature, class basis, class director, class model>
void TACSShellElement<quadrature, basis, director, model>::getMatType(
    ElementMatrixType matType, int elemIndex, double time,
    const TacsScalar Xpts[], const TacsScalar vars[], TacsScalar mat[]) {
  memset(mat, 0,
         vars_per_node * num_nodes * vars_per_node * num_nodes *
             sizeof(TacsScalar));
  TacsScalar *path;
  TacsScalar alpha, beta, gamma, dh, norm;
  alpha = beta = gamma = 0.0;
  // Create dummy residual vector
  TacsScalar res[vars_per_node * num_nodes];
  memset(res, 0, vars_per_node * num_nodes * sizeof(TacsScalar));

  dh = 1e-4;  // default for without override
  double dh_mag = 1e-4;

  bool _complexStepGmatrix = getComplexStepGmatrix();

#ifdef TACS_USE_COMPLEX
  if (_complexStepGmatrix) {
    dh_mag = 1e-30;
    dh = TacsScalar(0.0, dh_mag);
  }
#endif  // TACS_USE_COMPLEX

  // Set alpha or gamma based on if this is a stiffness or mass matrix
  if (matType == TACS_STIFFNESS_MATRIX) {
    alpha = 1.0;
  } else if (matType == TACS_MASS_MATRIX) {
    gamma = 1.0;
  } else {  // TACS_GEOMETRIC_STIFFNESS_MATRIX
    // Approximate geometric stiffness using directional derivative of
    // tangential stiffness projected along path of current state vars

    // compute norm for normalizing path vec
    norm = 0.0;
    for (int i = 0; i < vars_per_node * num_nodes; i++) {
      norm += vars[i] * vars[i];
    }

    // include thermal path in norm
    norm += temperature * temperature;

    if (TacsRealPart(norm) == 0.0) {
      norm = 1.0;
    } else {
      norm = sqrt(norm);
    }

    // Central difference the tangent stiffness matrix
    alpha = 0.5 * norm / dh_mag;

    // fwd step
    path = new TacsScalar[vars_per_node * num_nodes];
    for (int i = 0; i < vars_per_node * num_nodes; i++) {
      path[i] = dh * vars[i] / norm;
    }

    // temperature perturbation as well (for thermal buckling)
    TacsScalar my_dh = dh_mag;
    nlElem->setTemperature(temperature + my_dh * temperature / norm);
    // printf("temperature1 = %.8e\n", nlElem->getTemperature());

    nlElem->addJacobian(elemIndex, time, alpha, beta, gamma, Xpts, path, vars,
                        vars, res, mat);

    // bwd step
    for (int i = 0; i < vars_per_node * num_nodes; i++) {
      path[i] = -dh * vars[i] / norm;
    }

    // temperature perturbation as well (for thermal buckling)
    nlElem->setTemperature(temperature - my_dh * temperature / norm);
    // printf("temperature2 = %.8e\n", nlElem->getTemperature());

    nlElem->addJacobian(elemIndex, time, -alpha, beta, gamma, Xpts, path, vars,
                        vars, res, mat);

    // rescale by 1.0/i if complex_step_override is on
#ifdef TACS_USE_COMPLEX
    if (_complexStepGmatrix) {
      // take imaginary part of the element matrix
      for (int i = 0; i < vars_per_node * num_nodes * vars_per_node * num_nodes;
           i++) {
        mat[i] = TacsScalar(TacsImagPart(mat[i]), 0.0);
      }
    }
#endif  // TACS_USE_COMPLEX

    delete[] path;

    return;
  }
  // Add appropriate Jacobian to matrix
  addJacobian(elemIndex, time, alpha, beta, gamma, Xpts, vars, vars, vars, res,
              mat);
}

template <class quadrature, class basis, class director, class model>
void TACSShellElement<quadrature, basis, director, model>::addAdjResProduct(
    int elemIndex, double time, TacsScalar scale, const TacsScalar psi[],
    const TacsScalar Xpts[], const TacsScalar vars[], const TacsScalar dvars[],
    const TacsScalar ddvars[], int dvLen, TacsScalar dfdx[]) {
  // Compute the number of quadrature points
  const int nquad = quadrature::getNumQuadraturePoints();

  // Compute the node normal directions
  TacsScalar fn[3 * num_nodes], Xdn[9 * num_nodes];
  TacsShellComputeNodeNormals<TacsScalar,basis>(Xpts, fn, Xdn);

  // Store information about the transformation and derivatives at each node for
  // the drilling degrees of freedom
  TacsScalar etn[num_nodes], etnd[num_nodes];
  TacsScalar XdinvTn[9 * num_nodes], Tn[9 * num_nodes];
  TacsScalar u0xn[9 * num_nodes], Ctn[csize];
  TacsShellComputeDrillStrainDeriv<vars_per_node, offset, basis, director,
                                   model>(transform, Xdn, fn, vars, psi,
                                          XdinvTn, Tn, u0xn, Ctn, etn, etnd);

  // Compute the director rates and their derivatives
  TacsScalar d[dsize], ddot[dsize], dddot[dsize], dd[dsize];
  director::template computeDirectorRatesDeriv<vars_per_node, offset,
                                               num_nodes>(
      vars, dvars, ddvars, psi, fn, d, ddot, dddot, dd);

  // Set the total number of tying points needed for this element
  TacsScalar ety[basis::NUM_TYING_POINTS], etyd[basis::NUM_TYING_POINTS];
  model::template computeTyingStrainDeriv<vars_per_node, basis>(
      Xpts, fn, vars, d, psi, dd, ety, etyd);

  // Loop over each quadrature point and add the residual contribution
  for (int quad_index = 0; quad_index < nquad; quad_index++) {
    // Get the quadrature weight
    double pt[3];
    double weight = quadrature::getQuadraturePoint(quad_index, pt);

    // Compute X, X,xi and the interpolated normal n0
    TacsScalar X[3], Xxi[6], n0[3], T[9], et, etd;
    basis::template interpFields<TacsScalar,3, 3>(pt, Xpts, X);
    basis::template interpFieldsGrad<TacsScalar,3, 3>(pt, Xpts, Xxi);
    basis::template interpFields<TacsScalar,3, 3>(pt, fn, n0);
    basis::template interpFields<TacsScalar,1, 1>(pt, etn, &et);
    basis::template interpFields<TacsScalar,1, 1>(pt, etnd, &etd);

    // Compute the transformation at the quadrature point
    transform->computeTransform(Xxi, n0, T);

    // Evaluate the displacement gradient at the point
    TacsScalar XdinvT[9], XdinvzT[9];
    TacsScalar u0x[9], u1x[9], u0xd[9], u1xd[9];
    TacsScalar detXd = TacsShellComputeDispGradDeriv<vars_per_node, basis>(
        pt, Xpts, vars, fn, d, Xxi, n0, T, psi, dd, XdinvT, XdinvzT, u0x, u1x,
        u0xd, u1xd);
    detXd *= weight;

    // Evaluate the tying components of the strain
    TacsScalar gty[6], gtyd[6];  // The symmetric components of the tying strain
    basis::interpTyingStrain(pt, ety, gty);
    basis::interpTyingStrain(pt, etyd, gtyd);

    // Compute the symmetric parts of the tying strain
    TacsScalar e0ty[6], e0tyd[6];  // e0ty = XdinvT^{T}*gty*XdinvT
    mat3x3SymmTransformTranspose(XdinvT, gty, e0ty);
    mat3x3SymmTransformTranspose(XdinvT, gtyd, e0tyd);

    // Compute the set of strain components
    TacsScalar e[9];   // The components of the strain
    TacsScalar ed[9];  // The directional derivative components of the strain
    model::evalStrainDeriv(u0x, u1x, e0ty, u0xd, u1xd, e0tyd, e, ed);
    e[8] = et;
    ed[8] = etd;

    // The directional derivative of the strain along the adjoint direction
    con->addStressDVSens(elemIndex, scale * detXd, pt, X, e, ed, dvLen, dfdx);

    // Evaluate the second time derivatives
    TacsScalar u0ddot[3], d0ddot[3];
    basis::template interpFields<TacsScalar,vars_per_node, 3>(pt, ddvars, u0ddot);
    basis::template interpFields<TacsScalar,3, 3>(pt, dddot, d0ddot);

    TacsScalar du0ddot[3], dd0ddot[3];
    basis::template interpFields<TacsScalar,vars_per_node, 3>(pt, psi, du0ddot);
    basis::template interpFields<TacsScalar,3, 3>(pt, dd, dd0ddot);

    TacsScalar coef[3];
    coef[0] = scale * detXd * vec3Dot(u0ddot, du0ddot);
    coef[1] =
        scale * detXd * (vec3Dot(u0ddot, dd0ddot) + vec3Dot(du0ddot, d0ddot));
    coef[2] = scale * detXd * vec3Dot(d0ddot, dd0ddot);

    // Add the contribution from the dynamics
    con->addMassMomentsDVSens(elemIndex, pt, X, coef, dvLen, dfdx);
  }
}

template <class quadrature, class basis, class director, class model>
int TACSShellElement<quadrature, basis, director, model>::evalPointQuantity(
    int elemIndex, int quantityType, double time, int n, double pt[],
    const TacsScalar Xpts[], const TacsScalar vars[], const TacsScalar dvars[],
    const TacsScalar ddvars[], TacsScalar *detXd, TacsScalar *quantity) {
  // Compute the node normal directions
  TacsScalar fn[3 * num_nodes];
  TacsShellComputeNodeNormals<TacsScalar,basis>(Xpts, fn);

  if (quantityType == TACS_FAILURE_INDEX ||
      quantityType == TACS_STRAIN_ENERGY_DENSITY) {
    if (quantity) {
      // Compute the director rates
      TacsScalar d[dsize], ddot[dsize];
      director::template computeDirectorRates<TacsScalar,vars_per_node, offset, num_nodes>(
          vars, dvars, fn, d, ddot);

      // Set the total number of tying points needed for this element
      TacsScalar ety[basis::NUM_TYING_POINTS];
      model::template computeTyingStrain<TacsScalar,vars_per_node, basis>(Xpts, fn, vars,
                                                               d, ety);

      // Compute X, X,xi and the interpolated normal n0
      TacsScalar X[3], Xxi[6], n0[3], T[9];
      basis::template interpFields<TacsScalar,3, 3>(pt, Xpts, X);
      basis::template interpFieldsGrad<TacsScalar,3, 3>(pt, Xpts, Xxi);
      basis::template interpFields<TacsScalar,3, 3>(pt, fn, n0);

      // Compute the transformation at the quadrature point
      transform->computeTransform(Xxi, n0, T);

      // Evaluate the displacement gradient at the point
      TacsScalar XdinvT[9], XdinvzT[9];
      TacsScalar u0x[9], u1x[9];
      *detXd = TacsShellComputeDispGrad<TacsScalar, vars_per_node, basis>(
          pt, Xpts, vars, fn, d, Xxi, n0, T, XdinvT, XdinvzT, u0x, u1x);

      // Evaluate the tying components of the strain
      TacsScalar gty[6];  // The symmetric components of the tying strain
      basis::interpTyingStrain(pt, ety, gty);

      // Compute the symmetric parts of the tying strain
      TacsScalar e0ty[6];  // e0ty = XdinvT^{T}*gty*XdinvT
      mat3x3SymmTransformTranspose(XdinvT, gty, e0ty);

      // Compute the set of strain components
      TacsScalar e[9];  // The components of the strain
      model::evalStrain(u0x, u1x, e0ty, e);
      e[8] = 0.0;

      if (quantityType == TACS_FAILURE_INDEX) {
        *quantity = con->evalFailure(elemIndex, pt, X, e);
      } else {  // quantityType == TACS_STRAIN_ENERGY_DENSITY
        TacsScalar s[9];
        con->evalStress(elemIndex, pt, X, e, s);
        *quantity = 0.0;
        for (int i = 0; i < 9; i++) {
          *quantity += e[i] * s[i];
        }
      }
    }

    return 1;
  } else if (quantityType == TACS_ELEMENT_DENSITY) {
    if (quantity) {
      TacsScalar Xxi[6], n0[3], X[3];
      basis::template interpFields<TacsScalar,3, 3>(pt, Xpts, X);
      basis::template interpFieldsGrad<TacsScalar,3, 3>(pt, Xpts, Xxi);
      basis::template interpFields<TacsScalar,3, 3>(pt, fn, n0);

      TacsScalar Xd[9];
      TacsShellAssembleFrame(Xxi, n0, Xd);
      *detXd = det3x3(Xd);
      *quantity = con->evalDensity(elemIndex, pt, X);
    }

    return 1;
  } else if (quantityType == TACS_ELEMENT_DISPLACEMENT) {
    if (quantity) {
      TacsScalar Xxi[6], n0[3], X[3];
      basis::template interpFields<TacsScalar,3, 3>(pt, Xpts, X);
      basis::template interpFieldsGrad<TacsScalar,3, 3>(pt, Xpts, Xxi);
      basis::template interpFields<TacsScalar,3, 3>(pt, fn, n0);

      TacsScalar Xd[9];
      TacsShellAssembleFrame(Xxi, n0, Xd);
      *detXd = det3x3(Xd);

      // Compute the interpolated displacements
      basis::template interpFields<TacsScalar,vars_per_node, 3>(pt, vars, quantity);
    }

    return 3;
  } else if (quantityType == TACS_ELEMENT_DENSITY_MOMENT) {
    if (quantity) {
      TacsScalar Xxi[6], n0[3], X[3];
      basis::template interpFields<TacsScalar,3, 3>(pt, Xpts, X);
      basis::template interpFieldsGrad<TacsScalar,3, 3>(pt, Xpts, Xxi);
      basis::template interpFields<TacsScalar,3, 3>(pt, fn, n0);

      TacsScalar Xd[9];
      TacsShellAssembleFrame(Xxi, n0, Xd);
      *detXd = det3x3(Xd);
      TacsScalar moments[3];
      con->evalMassMoments(elemIndex, pt, X, moments);
      TacsScalar density = moments[0];

      quantity[0] = density * X[0] + moments[1] * n0[0];
      quantity[1] = density * X[1] + moments[1] * n0[1];
      quantity[2] = density * X[2] + moments[1] * n0[2];
    }

    return 3;
  } else if (quantityType == TACS_ELEMENT_MOMENT_OF_INERTIA) {
    if (quantity) {
      // Compute X, X,xi and the interpolated normal n0
      TacsScalar X[3], Xxi[6], n0[3], T[9];
      basis::template interpFields<TacsScalar,3, 3>(pt, Xpts, X);
      basis::template interpFieldsGrad<TacsScalar,3, 3>(pt, Xpts, Xxi);
      basis::template interpFields<TacsScalar,3, 3>(pt, fn, n0);

      // Compute the transformation at the quadrature point
      transform->computeTransform(Xxi, n0, T);

      TacsScalar Xd[9];
      TacsShellAssembleFrame(Xxi, n0, Xd);
      *detXd = det3x3(Xd);

      TacsScalar I0[6] = {0.0};

      // Evaluate the self MOI
      TacsScalar moments[3];
      con->evalMassMoments(elemIndex, pt, X, moments);
      TacsScalar density = moments[0];
      I0[0] = I0[3] = moments[2] - moments[1] * moments[1] / density;
      // Compute T*I0*T^{T}
      mat3x3SymmTransform(T, I0, quantity);
      TacsScalar dXcg[3];
      for (int i = 0; i < 3; i++) {
        dXcg[i] = X[i] + moments[1] / density * n0[i];
      }

      // Use parallel axis theorem to move MOI to origin
      quantity[0] += density * (dXcg[1] * dXcg[1] + dXcg[2] * dXcg[2]);
      quantity[1] += -density * dXcg[0] * dXcg[1];
      quantity[2] += -density * dXcg[0] * dXcg[2];
      quantity[3] += density * (dXcg[0] * dXcg[0] + dXcg[2] * dXcg[2]);
      quantity[4] += -density * dXcg[2] * dXcg[1];
      quantity[5] += density * (dXcg[0] * dXcg[0] + dXcg[1] * dXcg[1]);
    }

    return 6;
  }

  else if (quantityType == TACS_ELEMENT_ENCLOSED_VOLUME) {
    if (quantity) {
      // Compute X, X,xi and the interpolated normal n0
      TacsScalar Xxi[6], n0[3], X[3];
      basis::template interpFields<TacsScalar,3, 3>(pt, Xpts, X);
      basis::template interpFieldsGrad<TacsScalar,3, 3>(pt, Xpts, Xxi);
      basis::template interpFields<TacsScalar,3, 3>(pt, fn, n0);

      TacsScalar Xd[9];
      TacsShellAssembleFrame(Xxi, n0, Xd);
      *detXd = det3x3(Xd);

      // Compute 1/3*int[x * n]dA
      // This can be shown to equivalent to the volume through Gauss' Theorem
      quantity[0] = (X[0] * n0[0] + X[1] * n0[1] + X[2] * n0[2]) / 3.0;
    }

    return 1;
  }

  return 0;
}

template <class quadrature, class basis, class director, class model>
void TACSShellElement<quadrature, basis, director, model>::
    addPointQuantityDVSens(int elemIndex, int quantityType, double time,
                           TacsScalar scale, int n, double pt[],
                           const TacsScalar Xpts[], const TacsScalar vars[],
                           const TacsScalar dvars[], const TacsScalar ddvars[],
                           const TacsScalar dfdq[], int dvLen,
                           TacsScalar dfdx[]) {
  if (quantityType == TACS_FAILURE_INDEX ||
      quantityType == TACS_STRAIN_ENERGY_DENSITY) {
    // Compute the node normal directions
    TacsScalar fn[3 * num_nodes];
    TacsShellComputeNodeNormals<TacsScalar,basis>(Xpts, fn);

    // Compute the director rates
    TacsScalar d[dsize], ddot[dsize];
    director::template computeDirectorRates<TacsScalar,vars_per_node, offset, num_nodes>(
        vars, dvars, fn, d, ddot);

    // Set the total number of tying points needed for this element
    TacsScalar ety[basis::NUM_TYING_POINTS];
    model::template computeTyingStrain<TacsScalar,vars_per_node, basis>(Xpts, fn, vars, d,
                                                             ety);

    // Compute X, X,xi and the interpolated normal n0
    TacsScalar X[3], Xxi[6], n0[3], T[9];
    basis::template interpFields<TacsScalar,3, 3>(pt, Xpts, X);
    basis::template interpFieldsGrad<TacsScalar,3, 3>(pt, Xpts, Xxi);
    basis::template interpFields<TacsScalar,3, 3>(pt, fn, n0);

    // Compute the transformation at the quadrature point
    transform->computeTransform(Xxi, n0, T);

    // Evaluate the displacement gradient at the point
    TacsScalar XdinvT[9], XdinvzT[9];
    TacsScalar u0x[9], u1x[9];
    TacsShellComputeDispGrad<TacsScalar, vars_per_node, basis>(
        pt, Xpts, vars, fn, d, Xxi, n0, T, XdinvT, XdinvzT, u0x, u1x);

    // Evaluate the tying components of the strain
    TacsScalar gty[6];  // The symmetric components of the tying strain
    basis::interpTyingStrain(pt, ety, gty);

    // Compute the symmetric parts of the tying strain
    TacsScalar e0ty[6];  // e0ty = XdinvT^{T}*gty*XdinvT
    mat3x3SymmTransformTranspose(XdinvT, gty, e0ty);

    // Compute the set of strain components
    TacsScalar e[9];  // The components of the strain
    model::evalStrain(u0x, u1x, e0ty, e);
    e[8] = 0.0;

    if (quantityType == TACS_FAILURE_INDEX) {
      con->addFailureDVSens(elemIndex, scale * dfdq[0], pt, X, e, dvLen, dfdx);
    } else {  // quantityType == TACS_STRAIN_ENERGY_DENSITY
      TacsScalar s[9];
      con->evalStress(elemIndex, pt, X, e, s);
      con->addStressDVSens(elemIndex, scale * dfdq[0], pt, X, e, e, dvLen,
                           dfdx);
    }
  } else if (quantityType == TACS_ELEMENT_DENSITY) {
    TacsScalar X[3];
    basis::template interpFields<TacsScalar,3, 3>(pt, Xpts, X);

    con->addDensityDVSens(elemIndex, scale * dfdq[0], pt, X, dvLen, dfdx);
  } else if (quantityType == TACS_ELEMENT_DENSITY_MOMENT) {
    // Compute the node normal directions
    TacsScalar fn[3 * num_nodes];
    TacsShellComputeNodeNormals<TacsScalar,basis>(Xpts, fn);

    // Compute X and the interpolated normal n0
    TacsScalar X[3], n0[3];
    basis::template interpFields<TacsScalar,3, 3>(pt, Xpts, X);
    basis::template interpFields<TacsScalar,3, 3>(pt, fn, n0);

    TacsScalar dfdmoments[3] = {0.0};

    for (int i = 0; i < 3; i++) {
      dfdmoments[0] += scale * dfdq[i] * X[i];
      dfdmoments[1] += scale * dfdq[i] * n0[i];
    }

    con->addMassMomentsDVSens(elemIndex, pt, X, dfdmoments, dvLen, dfdx);
  } else if (quantityType == TACS_ELEMENT_MOMENT_OF_INERTIA) {
    // Compute the node normal directions
    TacsScalar fn[3 * num_nodes];
    TacsShellComputeNodeNormals<TacsScalar,basis>(Xpts, fn);

    // Compute X, X,xi and the interpolated normal n0
    TacsScalar X[3], Xxi[6], n0[3], T[9];
    basis::template interpFields<TacsScalar,3, 3>(pt, Xpts, X);
    basis::template interpFieldsGrad<TacsScalar,3, 3>(pt, Xpts, Xxi);
    basis::template interpFields<TacsScalar,3, 3>(pt, fn, n0);

    // Compute the transformation at the quadrature point
    transform->computeTransform(Xxi, n0, T);

    TacsScalar Xd[9];
    TacsShellAssembleFrame(Xxi, n0, Xd);

    TacsScalar dfdI0[6] = {0.0};

    // Evaluate the self MOI
    TacsScalar moments[3];
    con->evalMassMoments(elemIndex, pt, X, moments);
    TacsScalar density = moments[0];

    // Evaluate the self MOI
    TacsScalar dfdmoments[3];
    mat3x3SymmTransformSens(T, dfdq, dfdI0);
    dfdmoments[2] = scale * (dfdI0[0] + dfdI0[3]);
    dfdmoments[1] = -scale * 2.0 * moments[1] / density * (dfdI0[0] + dfdI0[3]);
    dfdmoments[0] = scale * moments[1] * moments[1] / density / density *
                    (dfdI0[0] + dfdI0[3]);

    TacsScalar dXcg[3];
    for (int i = 0; i < 3; i++) {
      dXcg[i] = X[i] + moments[1] / density * n0[i];
    }

    // Use parallel axis theorem to move MOI to origin
    dfdmoments[0] +=
        scale * dfdq[0] *
        (dXcg[1] * dXcg[1] + dXcg[2] * dXcg[2] -
         2.0 * moments[1] / density * (dXcg[1] * n0[1] + dXcg[2] * n0[2]));
    dfdmoments[0] -=
        scale * dfdq[1] *
        (dXcg[0] * dXcg[1] -
         moments[1] / density * (dXcg[0] * n0[1] + dXcg[1] * n0[0]));
    dfdmoments[0] -=
        scale * dfdq[2] *
        (dXcg[0] * dXcg[2] -
         moments[1] / density * (dXcg[0] * n0[2] + dXcg[2] * n0[0]));
    dfdmoments[0] +=
        scale * dfdq[3] *
        (dXcg[0] * dXcg[0] + dXcg[2] * dXcg[2] -
         2.0 * moments[1] / density * (dXcg[0] * n0[0] + dXcg[2] * n0[2]));
    dfdmoments[0] -=
        scale * dfdq[4] *
        (dXcg[2] * dXcg[1] -
         moments[1] / density * (dXcg[1] * n0[2] + dXcg[2] * n0[1]));
    dfdmoments[0] +=
        scale * dfdq[5] *
        (dXcg[0] * dXcg[0] + dXcg[1] * dXcg[1] -
         2.0 * moments[1] / density * (dXcg[0] * n0[0] + dXcg[1] * n0[1]));

    dfdmoments[1] +=
        scale * dfdq[0] * 2.0 * (dXcg[1] * n0[1] + dXcg[2] * n0[2]);
    dfdmoments[1] -= scale * dfdq[1] * (dXcg[0] * n0[1] + dXcg[1] * n0[0]);
    dfdmoments[1] -= scale * dfdq[2] * (dXcg[0] * n0[2] + dXcg[2] * n0[0]);
    dfdmoments[1] +=
        scale * dfdq[3] * 2.0 * (dXcg[0] * n0[0] + dXcg[2] * n0[2]);
    dfdmoments[1] -= scale * dfdq[4] * (dXcg[1] * n0[2] + dXcg[2] * n0[1]);
    dfdmoments[1] +=
        scale * dfdq[5] * 2.0 * (dXcg[0] * n0[0] + dXcg[1] * n0[1]);

    con->addMassMomentsDVSens(elemIndex, pt, X, dfdmoments, dvLen, dfdx);
  }
}

template <class quadrature, class basis, class director, class model>
void TACSShellElement<quadrature, basis, director, model>::
    addPointQuantitySVSens(int elemIndex, int quantityType, double time,
                           TacsScalar alpha, TacsScalar beta, TacsScalar gamma,
                           int n, double pt[], const TacsScalar Xpts[],
                           const TacsScalar vars[], const TacsScalar dvars[],
                           const TacsScalar ddvars[], const TacsScalar dfdq[],
                           TacsScalar dfdu[]) {
  if (quantityType == TACS_FAILURE_INDEX ||
      quantityType == TACS_STRAIN_ENERGY_DENSITY) {
    // Derivative of the director field
    TacsScalar dd[dsize];
    memset(dd, 0, 3 * num_nodes * sizeof(TacsScalar));

    // Zero the contributions to the tying strain derivatives
    TacsScalar dety[basis::NUM_TYING_POINTS];
    memset(dety, 0, basis::NUM_TYING_POINTS * sizeof(TacsScalar));

    // Compute the node normal directions
    TacsScalar fn[3 * num_nodes], Xdn[9 * num_nodes];
    TacsShellComputeNodeNormals<TacsScalar,basis>(Xpts, fn, Xdn);

    TacsScalar d[dsize], ddot[dsize], dddot[dsize];
    director::template computeDirectorRates<TacsScalar,vars_per_node, offset, num_nodes>(
        vars, dvars, ddvars, fn, d, ddot, dddot);

    // Set the total number of tying points needed for this element
    TacsScalar ety[basis::NUM_TYING_POINTS];
    model::template computeTyingStrain<TacsScalar,vars_per_node, basis>(Xpts, fn, vars, d,
                                                             ety);

    // Compute X, X,xi and the interpolated normal n0
    TacsScalar X[3], Xxi[6], n0[3], T[9];
    basis::template interpFields<TacsScalar,3, 3>(pt, Xpts, X);
    basis::template interpFieldsGrad<TacsScalar,3, 3>(pt, Xpts, Xxi);
    basis::template interpFields<TacsScalar,3, 3>(pt, fn, n0);

    // Compute the transformation at the quadrature point
    transform->computeTransform(Xxi, n0, T);

    // Evaluate the displacement gradient at the point
    TacsScalar XdinvT[9], XdinvzT[9];
    TacsScalar u0x[9], u1x[9];
    TacsShellComputeDispGrad<TacsScalar, vars_per_node, basis>(
        pt, Xpts, vars, fn, d, Xxi, n0, T, XdinvT, XdinvzT, u0x, u1x);

    // Evaluate the tying components of the strain
    TacsScalar gty[6];  // The symmetric components of the tying strain
    basis::interpTyingStrain(pt, ety, gty);

    // Compute the symmetric parts of the tying strain
    TacsScalar e0ty[6];  // e0ty = XdinvT^{T}*gty*XdinvT
    mat3x3SymmTransformTranspose(XdinvT, gty, e0ty);

    // Compute the set of strain components
    TacsScalar e[9];  // The components of the strain
    model::evalStrain(u0x, u1x, e0ty, e);
    e[8] = 0.0;

    TacsScalar sens[9];
    if (quantityType == TACS_FAILURE_INDEX) {
      // Compute the sensitivity of the failure index w.r.t. the strain
      con->evalFailureStrainSens(elemIndex, pt, X, e, sens);
    } else {  // quantityType == TACS_STRAIN_ENERGY_DENSITY
      // Compute the sensitivity of the strain energy density w.r.t. the strain
      con->evalStress(elemIndex, pt, X, e, sens);
      for (int i = 0; i < 9; i++) {
        sens[i] *= 2.0;
      }
    }

    // Compute the derivative of the product of the stress and strain
    // with respect to u0x, u1x and e0ty
    TacsScalar du0x[9], du1x[9], de0ty[6];
    model::template evalStrainSens<TacsScalar>(alpha * dfdq[0], sens, u0x, u1x, du0x, du1x, de0ty);

    // Add the contributions to the residual from du0x, du1x and dCt
    TacsShellAddDispGradSens<TacsScalar, vars_per_node, basis>(pt, T, XdinvT, XdinvzT, du0x,
                                                   du1x, dfdu, dd);

    // Compute the of the tying strain w.r.t. derivative w.r.t. the coefficients
    TacsScalar dgty[6];
    mat3x3SymmTransformTransSens(XdinvT, de0ty, dgty);

    // Evaluate the tying strain
    basis::addInterpTyingStrainTranspose(pt, dgty, dety);

    // Set the total number of tying points needed for this element
    model::template addComputeTyingStrainTranspose<TacsScalar,vars_per_node, basis>(
        Xpts, fn, vars, d, dety, dfdu, dd);

    // Add the contributions to the director field
    director::template addDirectorResidual<TacsScalar,vars_per_node, offset, num_nodes>(
        vars, dvars, ddvars, fn, dd, dfdu);
  } else if (quantityType == TACS_ELEMENT_DISPLACEMENT) {
    // Compute the interpolated displacements
    basis::template addInterpFieldsTranspose<TacsScalar,vars_per_node, 3>(pt, dfdq, dfdu);
  }
}

/*
  Get the element data for the basis
*/
template <class quadrature, class basis, class director, class model>
void TACSShellElement<quadrature, basis, director, model>::getAverageStresses(
    int elemIndex, ElementType etype, const TacsScalar Xpts[],
    const TacsScalar vars[], const TacsScalar dvars[],
    const TacsScalar ddvars[], TacsScalar *avgStresses) {
  if (etype == TACS_BEAM_OR_SHELL_ELEMENT) {
    // Get the number of nodes associated with the visualization
    int num_vis_nodes = TacsGetNumVisNodes(basis::getLayoutType());

    // Compute the node normal directions
    TacsScalar fn[3 * num_nodes], Xdn[9 * num_nodes];
    TacsShellComputeNodeNormals<TacsScalar,basis>(Xpts, fn, Xdn);

    // Store information about the transformation and derivatives at each node
    // for the drilling degrees of freedom
    TacsScalar etn[num_nodes];
    TacsScalar XdinvTn[9 * num_nodes], Tn[9 * num_nodes];
    TacsScalar u0xn[9 * num_nodes], Ctn[csize];
    TacsShellComputeDrillStrain<TacsScalar,vars_per_node, offset, basis, director, model>(
        transform, Xdn, fn, vars, XdinvTn, Tn, u0xn, Ctn, etn);

    TacsScalar d[dsize], ddot[dsize], dddot[dsize];
    director::template computeDirectorRates<TacsScalar, vars_per_node, offset, num_nodes>(
        vars, dvars, ddvars, fn, d, ddot, dddot);

    // Set the total number of tying points needed for this element
    TacsScalar ety[basis::NUM_TYING_POINTS];
    model::template computeTyingStrain<TacsScalar,vars_per_node, basis>(Xpts, fn, vars, d,
                                                             ety);

    TacsScalar loc_avgStresses[9];
    // memset(loc_avgStresses,0,9);
    for (int i = 0; i < 9; i++) {
      loc_avgStresses[i] = 0.0;
    }

    // Loop over each quadrature point and add the residual contribution
    for (int index = 0; index < num_vis_nodes; index++) {
      // Get the quadrature weight
      double pt[3];
      basis::getNodePoint(index, pt);

      // Compute X, X,xi and the interpolated normal n0
      TacsScalar X[3], Xxi[6], n0[3], T[9], et;
      basis::template interpFields<TacsScalar,3, 3>(pt, Xpts, X);
      basis::template interpFieldsGrad<TacsScalar,3, 3>(pt, Xpts, Xxi);
      basis::template interpFields<TacsScalar,3, 3>(pt, fn, n0);
      basis::template interpFields<TacsScalar,1, 1>(pt, etn, &et);

      // Compute the transformation at the quadrature point
      transform->computeTransform(Xxi, n0, T);

      // Evaluate the displacement gradient at the point
      TacsScalar XdinvT[9], XdinvzT[9];
      TacsScalar u0x[9], u1x[9];
      TacsShellComputeDispGrad<TacsScalar, vars_per_node, basis>(
          pt, Xpts, vars, fn, d, Xxi, n0, T, XdinvT, XdinvzT, u0x, u1x);

      // Evaluate the tying components of the strain
      TacsScalar gty[6];  // The symmetric components of the tying strain
      basis::interpTyingStrain(pt, ety, gty);

      // Compute the symmetric parts of the tying strain
      TacsScalar e0ty[6];  // e0ty = XdinvT^{T}*gty*XdinvT
      mat3x3SymmTransformTranspose(XdinvT, gty, e0ty);

      // Compute the set of strain components
      TacsScalar e[9];  // The components of the strain
      model::evalStrain(u0x, u1x, e0ty, e);
      e[8] = et;

      // Compute the corresponding stresses
      TacsScalar s[9];
      con->evalStress(elemIndex, pt, X, e, s);

      for (int i = 0; i < 9; i++) {
        loc_avgStresses[i] += s[i];
      }
    }

    // average the average stresses among the quadrature points
    for (int i = 0; i < 9; i++) {
      loc_avgStresses[i] /= num_vis_nodes;
      avgStresses[i] += loc_avgStresses[i];
    }
  }
}

/*
  Get the element data for the basis
*/
template <class quadrature, class basis, class director, class model>
void TACSShellElement<quadrature, basis, director, model>::getOutputData(
    int elemIndex, ElementType etype, int write_flag, const TacsScalar Xpts[],
    const TacsScalar vars[], const TacsScalar dvars[],
    const TacsScalar ddvars[], int ld_data, TacsScalar *data) {
  if (etype == TACS_BEAM_OR_SHELL_ELEMENT) {
    // Get the number of nodes associated with the visualization
    int num_vis_nodes = TacsGetNumVisNodes(basis::getLayoutType());

    // Compute the node normal directions
    TacsScalar fn[3 * num_nodes], Xdn[9 * num_nodes];
    TacsShellComputeNodeNormals<TacsScalar,basis>(Xpts, fn, Xdn);

    // Store information about the transformation and derivatives at each node
    // for the drilling degrees of freedom
    TacsScalar etn[num_nodes];
    TacsScalar XdinvTn[9 * num_nodes], Tn[9 * num_nodes];
    TacsScalar u0xn[9 * num_nodes], Ctn[csize];
    TacsShellComputeDrillStrain<TacsScalar,vars_per_node, offset, basis, director, model>(
        transform, Xdn, fn, vars, XdinvTn, Tn, u0xn, Ctn, etn);

    TacsScalar d[dsize], ddot[dsize], dddot[dsize];
    director::template computeDirectorRates<TacsScalar,vars_per_node, offset, num_nodes>(
        vars, dvars, ddvars, fn, d, ddot, dddot);

    // Set the total number of tying points needed for this element
    TacsScalar ety[basis::NUM_TYING_POINTS];
    model::template computeTyingStrain<TacsScalar,vars_per_node, basis>(Xpts, fn, vars, d,
                                                             ety);

    // Loop over each quadrature point and add the residual contribution
    for (int index = 0; index < num_vis_nodes; index++) {
      // Get the quadrature weight
      double pt[3];
      basis::getNodePoint(index, pt);

      // Compute X, X,xi and the interpolated normal n0
      TacsScalar X[3], Xxi[6], n0[3], T[9], et;
      basis::template interpFields<TacsScalar,3, 3>(pt, Xpts, X);
      basis::template interpFieldsGrad<TacsScalar,3, 3>(pt, Xpts, Xxi);
      basis::template interpFields<TacsScalar,3, 3>(pt, fn, n0);
      basis::template interpFields<TacsScalar,1, 1>(pt, etn, &et);

      // Compute the transformation at the quadrature point
      transform->computeTransform(Xxi, n0, T);

      // Evaluate the displacement gradient at the point
      TacsScalar XdinvT[9], XdinvzT[9];
      TacsScalar u0x[9], u1x[9];
      TacsShellComputeDispGrad<TacsScalar, vars_per_node, basis>(
          pt, Xpts, vars, fn, d, Xxi, n0, T, XdinvT, XdinvzT, u0x, u1x);

      // Evaluate the tying components of the strain
      TacsScalar gty[6];  // The symmetric components of the tying strain
      basis::interpTyingStrain(pt, ety, gty);

      // Compute the symmetric parts of the tying strain
      TacsScalar e0ty[6];  // e0ty = XdinvT^{T}*gty*XdinvT
      mat3x3SymmTransformTranspose(XdinvT, gty, e0ty);

      // Compute the set of strain components
      TacsScalar e[9];  // The components of the strain
      model::evalStrain(u0x, u1x, e0ty, e);
      e[8] = et;

      // Compute the corresponding stresses
      TacsScalar s[9];
      con->evalStress(elemIndex, pt, X, e, s);

      if (write_flag & TACS_OUTPUT_NODES) {
        data[0] = X[0];
        data[1] = X[1];
        data[2] = X[2];
        data += 3;
      }
      if (write_flag & TACS_OUTPUT_DISPLACEMENTS) {
        int len = vars_per_node;
        if (len > 6) {
          len = 6;
        }
        for (int i = 0; i < len; i++) {
          data[i] = vars[i + vars_per_node * index];
        }
        for (int i = len; i < 6; i++) {
          data[i] = 0.0;
        }
        data += 6;
      }
      if (write_flag & TACS_OUTPUT_STRAINS) {
        for (int i = 0; i < 9; i++) {
          data[i] = e[i];
        }
        data += 9;
      }
      if (write_flag & TACS_OUTPUT_STRESSES) {
        for (int i = 0; i < 9; i++) {
          data[i] = s[i];
        }
        data += 9;
      }
      if (write_flag & TACS_OUTPUT_EXTRAS) {
        data[0] = con->evalFailureFieldValue(elemIndex, pt, X, e, 0);
        data[1] = con->evalFailureFieldValue(elemIndex, pt, X, e, 1);
        data[2] = con->evalFailureFieldValue(elemIndex, pt, X, e, 2);
        data[3] = con->evalFailureFieldValue(elemIndex, pt, X, e, 3);
        data[4] = con->evalFailureFieldValue(elemIndex, pt, X, e, 4);
        data[5] = con->evalFailureFieldValue(elemIndex, pt, X, e, 5);
        data[6] = con->evalDesignFieldValue(elemIndex, pt, X, 0);
        data[7] = con->evalDesignFieldValue(elemIndex, pt, X, 1);
        data[8] = con->evalDesignFieldValue(elemIndex, pt, X, 2);
        data[9] = con->evalDesignFieldValue(elemIndex, pt, X, 3);
        data[10] = con->evalDesignFieldValue(elemIndex, pt, X, 4);
        data[11] = con->evalDesignFieldValue(elemIndex, pt, X, 5);
        data[12] = con->evalDesignFieldValue(elemIndex, pt, X, 6);
        data += 13;
      }
    }
  }
}

template <int vars_per_node, class basis, class model>
int TacsTestShellTyingStrain(double dh = 1e-7, int test_print_level = 2,
                             double test_fail_atol = 1e-5,
                             double test_fail_rtol = 1e-5) {
  const int size = vars_per_node * basis::NUM_NODES;
  const int usize = 3 * basis::NUM_NODES;
  const int dsize = 3 * basis::NUM_NODES;

  TacsScalar Xpts[3 * basis::NUM_NODES], fn[3 * basis::NUM_NODES];
  TacsGenerateRandomArray(Xpts, 3 * basis::NUM_NODES);
  TacsGenerateRandomArray(fn, 3 * basis::NUM_NODES);

  TacsScalar d[dsize], vars[size];
  TacsGenerateRandomArray(d, dsize);
  TacsGenerateRandomArray(vars, size);

  TacsScalar XdinvT[9];
  TacsGenerateRandomArray(XdinvT, 9);

  TacsScalar de0ty[6], d2e0ty[36];
  TacsGenerateRandomArray(de0ty, 6);
  TacsGenerateRandomArray(d2e0ty, 36);

  double pt[2];
  TacsGenerateRandomArray(pt, 2);

  TacsScalar dety[basis::NUM_TYING_POINTS];
  TacsScalar d2ety[basis::NUM_TYING_POINTS * basis::NUM_TYING_POINTS];
  TacsScalar d2etyu[basis::NUM_TYING_POINTS * usize];
  TacsScalar d2etyd[basis::NUM_TYING_POINTS * dsize];
  memset(dety, 0, basis::NUM_TYING_POINTS * sizeof(TacsScalar));
  memset(
      d2ety, 0,
      basis::NUM_TYING_POINTS * basis::NUM_TYING_POINTS * sizeof(TacsScalar));
  memset(d2etyu, 0, basis::NUM_TYING_POINTS * usize * sizeof(TacsScalar));
  memset(d2etyd, 0, basis::NUM_TYING_POINTS * dsize * sizeof(TacsScalar));

  TacsScalar ety[basis::NUM_TYING_POINTS];
  model::template computeTyingStrain<TacsScalar,vars_per_node, basis>(Xpts, fn, vars, d,
                                                           ety);

  // Evaluate the tying components of the strain
  TacsScalar gty[6];  // The symmetric components of the tying strain
  basis::interpTyingStrain(pt, ety, gty);

  // Compute the symmetric parts of the tying strain
  TacsScalar e0ty[6];  // e0ty = XdinvT^{T}*gty*XdinvT
  mat3x3SymmTransformTranspose(XdinvT, gty, e0ty);

  // Compute the of the tying strain w.r.t. derivative w.r.t. the coefficients
  TacsScalar dgty[6], d2gty[36];
  mat3x3SymmTransformTransSens(XdinvT, de0ty, dgty);
  mat3x3SymmTransformTransHessian(XdinvT, d2e0ty, d2gty);

  // Evaluate the tying strain
  basis::addInterpTyingStrainTranspose(pt, dgty, dety);
  basis::addInterpTyingStrainHessian(pt, d2gty, d2ety);

  TacsScalar res[size], dd[dsize];
  memset(res, 0, size * sizeof(TacsScalar));
  memset(dd, 0, dsize * sizeof(TacsScalar));

  TacsScalar mat[size * size], d2d[dsize * dsize], d2du[dsize * usize];
  memset(mat, 0, size * size * sizeof(TacsScalar));
  memset(d2d, 0, dsize * dsize * sizeof(TacsScalar));
  memset(d2du, 0, dsize * usize * sizeof(TacsScalar));

  // Set the total number of tying points needed for this element
  model::template addComputeTyingStrainTranspose<TacsScalar,vars_per_node, basis>(
      Xpts, fn, vars, d, dety, res, dd);
  model::template addComputeTyingStrainHessian<vars_per_node, basis>(
      1.0, Xpts, fn, vars, d, dety, d2ety, d2etyu, d2etyd, mat, d2d, d2du);

  TacsScalar fdmat[size * size], fdd2du[dsize * usize];
  for (int i = 0; i < size; i++) {
    TacsScalar varst[size];
    memcpy(varst, vars, size * sizeof(TacsScalar));

#ifdef TACS_USE_COMPLEX
    varst[i] = vars[i] + TacsScalar(0.0, dh);
#else
    varst[i] = vars[i] + dh;
#endif  // TACS_USE_COMPLEX

    // Perturb the variables
    TacsScalar etyt[basis::NUM_TYING_POINTS];
    model::template computeTyingStrain<TacsScalar,vars_per_node, basis>(Xpts, fn, varst, d,
                                                             etyt);

    // Evaluate the tying components of the strain
    TacsScalar gtyt[6];  // The symmetric components of the tying strain
    basis::interpTyingStrain(pt, etyt, gtyt);

    // Compute the symmetric parts of the tying strain
    TacsScalar e0tyt[6];
    mat3x3SymmTransformTranspose(XdinvT, gtyt, e0tyt);

    TacsScalar de0tyt[6];
    for (int j = 0; j < 6; j++) {
      de0tyt[j] = de0ty[j];
      for (int k = 0; k < 6; k++) {
        de0tyt[j] += d2e0ty[6 * j + k] * (e0tyt[k] - e0ty[k]);
      }
    }

    // Compute the of the tying strain w.r.t. derivative w.r.t. the coefficients
    TacsScalar dgtyt[6];
    mat3x3SymmTransformTransSens(XdinvT, de0tyt, dgtyt);

    TacsScalar detyt[basis::NUM_TYING_POINTS];
    memset(detyt, 0, basis::NUM_TYING_POINTS * sizeof(TacsScalar));

    // Evaluate the tying strain
    basis::addInterpTyingStrainTranspose(pt, dgtyt, detyt);

    TacsScalar rest[size], ddt[dsize];
    memset(rest, 0, size * sizeof(TacsScalar));
    memset(ddt, 0, dsize * sizeof(TacsScalar));

    // Set the total number of tying points needed for this element
    model::template addComputeTyingStrainTranspose<TacsScalar,vars_per_node, basis>(
        Xpts, fn, varst, d, detyt, rest, ddt);

    for (int j = 0; j < size; j++) {
#ifdef TACS_USE_COMPLEX
      fdmat[size * j + i] = TacsImagPart(rest[j]) / dh;
#else
      fdmat[size * j + i] = (rest[j] - res[j]) / dh;
#endif  // TACS_USE_COMPLEX
    }

    if (i % vars_per_node < 3) {
      int index = 3 * (i / vars_per_node) + i % vars_per_node;
      for (int j = 0; j < dsize; j++) {
#ifdef TACS_USE_COMPLEX
        fdd2du[usize * j + index] = TacsImagPart(ddt[j]) / dh;
#else
        fdd2du[usize * j + index] = (ddt[j] - dd[j]) / dh;
#endif  // TACS_USE_COMPLEX
      }
    }
  }

  int fail = 0;
  double max_err, max_rel;
  int max_err_index, max_rel_index;

  // Compute the error
  max_err = TacsGetMaxError(mat, fdmat, size * size, &max_err_index);
  max_rel = TacsGetMaxRelError(mat, fdmat, size * size, &max_rel_index);

  if (test_print_level > 0) {
    fprintf(stderr, "Testing the second derivative w.r.t. vars\n");
    fprintf(stderr, "Max Err: %10.4e in component %d.\n", max_err,
            max_err_index);
    fprintf(stderr, "Max REr: %10.4e in component %d.\n", max_rel,
            max_rel_index);
  }
  // Print the error if required
  if (test_print_level > 1) {
    TacsPrintErrorComponents(stderr, "mat", mat, fdmat, size * size);
  }
  if (test_print_level) {
    fprintf(stderr, "\n");
  }

  fail = (max_err > test_fail_atol || max_rel > test_fail_rtol);

  // Compute the error
  max_err = TacsGetMaxError(d2du, fdd2du, dsize * usize, &max_err_index);
  max_rel = TacsGetMaxRelError(d2du, fdd2du, dsize * usize, &max_rel_index);

  if (test_print_level > 0) {
    fprintf(stderr, "Testing the second derivative w.r.t. vars and d\n");
    fprintf(stderr, "Max Err: %10.4e in component %d.\n", max_err,
            max_err_index);
    fprintf(stderr, "Max REr: %10.4e in component %d.\n", max_rel,
            max_rel_index);
  }
  // Print the error if required
  if (test_print_level > 1) {
    TacsPrintErrorComponents(stderr, "d2du", d2du, fdd2du, dsize * usize);
  }
  if (test_print_level) {
    fprintf(stderr, "\n");
  }

  fail = (max_err > test_fail_atol || max_rel > test_fail_rtol);

  TacsScalar fdd2d[dsize * dsize];
  for (int i = 0; i < dsize; i++) {
    TacsScalar dt[size];
    memcpy(dt, d, dsize * sizeof(TacsScalar));

#ifdef TACS_USE_COMPLEX
    dt[i] = d[i] + TacsScalar(0.0, dh);
#else
    dt[i] = d[i] + dh;
#endif  // TACS_USE_COMPLEX

    // Perturb the variables
    TacsScalar etyt[basis::NUM_TYING_POINTS];
    model::template computeTyingStrain<TacsScalar,vars_per_node, basis>(Xpts, fn, vars, dt,
                                                             etyt);

    // Evaluate the tying components of the strain
    TacsScalar gtyt[6];  // The symmetric components of the tying strain
    basis::interpTyingStrain(pt, etyt, gtyt);

    // Compute the symmetric parts of the tying strain
    TacsScalar e0tyt[6];
    mat3x3SymmTransformTranspose(XdinvT, gtyt, e0tyt);

    TacsScalar de0tyt[6];
    for (int j = 0; j < 6; j++) {
      de0tyt[j] = de0ty[j];
      for (int k = 0; k < 6; k++) {
        de0tyt[j] += d2e0ty[6 * j + k] * (e0tyt[k] - e0ty[k]);
      }
    }

    // Compute the of the tying strain w.r.t. derivative w.r.t. the coefficients
    TacsScalar dgtyt[6];
    mat3x3SymmTransformTransSens(XdinvT, de0tyt, dgtyt);

    TacsScalar detyt[basis::NUM_TYING_POINTS];
    memset(detyt, 0, basis::NUM_TYING_POINTS * sizeof(TacsScalar));

    // Evaluate the tying strain
    basis::addInterpTyingStrainTranspose(pt, dgtyt, detyt);

    TacsScalar rest[size], ddt[dsize];
    memset(rest, 0, size * sizeof(TacsScalar));
    memset(ddt, 0, dsize * sizeof(TacsScalar));

    // Set the total number of tying points needed for this element
    model::template addComputeTyingStrainTranspose<TacsScalar,vars_per_node, basis>(
        Xpts, fn, vars, dt, detyt, rest, ddt);

    for (int j = 0; j < dsize; j++) {
#ifdef TACS_USE_COMPLEX
      fdd2d[dsize * j + i] = TacsImagPart(ddt[j]) / dh;
#else
      fdd2d[dsize * j + i] = (ddt[j] - dd[j]) / dh;
#endif  // TACS_USE_COMPLEX
    }
  }

  // Compute the error
  max_err = TacsGetMaxError(d2d, fdd2d, dsize * dsize, &max_err_index);
  max_rel = TacsGetMaxRelError(d2d, fdd2d, dsize * dsize, &max_rel_index);

  if (test_print_level > 0) {
    fprintf(stderr, "Testing the second derivative w.r.t. d\n");
    fprintf(stderr, "Max Err: %10.4e in component %d.\n", max_err,
            max_err_index);
    fprintf(stderr, "Max REr: %10.4e in component %d.\n", max_rel,
            max_rel_index);
  }
  // Print the error if required
  if (test_print_level > 1) {
    TacsPrintErrorComponents(stderr, "d2d", d2d, fdd2d, dsize * dsize);
  }
  if (test_print_level) {
    fprintf(stderr, "\n");
  }

  fail = (max_err > test_fail_atol || max_rel > test_fail_rtol);

  return fail;
}

// #endif  // TACS_SHELL_ELEMENT_H