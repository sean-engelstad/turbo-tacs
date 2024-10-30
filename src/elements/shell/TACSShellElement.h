#ifndef TACS_SHELL_ELEMENT_H
#define TACS_SHELL_ELEMENT_H

#include "TACSDirector.h"
#include "TACSElement.h"
#include "TACSElementAlgebra.h"
#include "TACSElementTypes.h"
#include "TACSElementVerification.h"
#include "TACSShellConstitutive.h"
#include "TACSShellElementModel.h"
#include "TACSShellElementTransform.h"
#include "TACSShellUtilities.h"
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
  TacsShellComputeNodeNormals<basis>(Xpts, fn, Xdn);

  // Store information about the transformation and derivatives at each node for
  // the drilling degrees of freedom
  TacsScalar etn[num_nodes];
  TacsScalar XdinvTn[9 * num_nodes], Tn[9 * num_nodes];
  TacsScalar u0xn[9 * num_nodes], Ctn[csize];
  TacsShellComputeDrillStrain<vars_per_node, offset, basis, director, model>(
      transform, Xdn, fn, vars, XdinvTn, Tn, u0xn, Ctn, etn);

  // Compute the director rates
  TacsScalar d[dsize], ddot[dsize];
  director::template computeDirectorRates<vars_per_node, offset, num_nodes>(
      vars, dvars, fn, d, ddot);

  // Set the total number of tying points needed for this element
  TacsScalar ety[basis::NUM_TYING_POINTS];
  model::template computeTyingStrain<vars_per_node, basis>(Xpts, fn, vars, d,
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

    basis::template interpFields<3, 3>(pt, Xpts, X.get_data());
    basis::template interpFieldsGrad<3, 3>(pt, Xpts, Xxi.get_data());
    basis::template interpFields<3, 3>(pt, fn, n0.get_data());
    basis::template interpFields<1, 1>(pt, etn, &et);

    // Compute the transformation at the quadrature point
    transform->computeTransform(Xxi.get_data(), n0.get_data(), T.get_data());

    // Evaluate the displacement gradient at the point
    A2D::Mat<TacsScalar, 3, 3> XdinvT, XdinvzT, u0x, u1x;
    TacsScalar detXd; // can we setup scalars like this in new vs A2D?
    detXd = TacsShellComputeDispGrad<vars_per_node, basis>(
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
    basis::template interpFields<vars_per_node, 3>(pt, dvars, u0dot.get_data());
    basis::template interpFields<3, 3>(pt, ddot, d0dot.get_data());

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
  TacsShellComputeNodeNormals<basis>(Xpts, fn, Xdn);

  // Compute the drill strain penalty at each node
  TacsScalar etn[num_nodes], detn[num_nodes];
  memset(detn, 0, num_nodes * sizeof(TacsScalar));

  // Store information about the transformation and derivatives at each node for
  // the drilling degrees of freedom
  TacsScalar XdinvTn[9 * num_nodes], Tn[9 * num_nodes];
  TacsScalar u0xn[9 * num_nodes], Ctn[csize];
  TacsShellComputeDrillStrain<vars_per_node, offset, basis, director, model>(
      transform, Xdn, fn, vars, XdinvTn, Tn, u0xn, Ctn, etn);

  TacsScalar d[dsize], ddot[dsize], dddot[dsize];
  director::template computeDirectorRates<vars_per_node, offset, num_nodes>(
      vars, dvars, ddvars, fn, d, ddot, dddot);

  // Compute the tying strain values
  TacsScalar ety[basis::NUM_TYING_POINTS], dety[basis::NUM_TYING_POINTS];
  memset(dety, 0, basis::NUM_TYING_POINTS * sizeof(TacsScalar));
  model::template computeTyingStrain<vars_per_node, basis>(Xpts, fn, vars, d,
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
    basis::template interpFields<3, 3>(pt, Xpts, X.get_data());
    basis::template interpFields<3, 3>(pt, fn, n0.get_data());
    basis::template interpFields<1, 1>(pt, etn, et.value().get_data());
    basis::template interpFields<3, 3>(pt, d, d0.value().get_data());

    basis::template interpFieldsGrad<3, 3>(pt, Xpts, Xxi.get_data());
    basis::template interpFieldsGrad<3, 3>(pt, fn, nxi.get_data());
    basis::template interpFieldsGrad<3, 3>(pt, d, d0xi.value().get_data());
    basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, u0xi.value().get_data());

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
    basis::template addInterpFieldsTranspose<1, 1>(pt, et.bvalue().get_data(), detn);
    basis::template addInterpFieldsTranspose<3, 3>(pt, d0.bvalue().get_data(), dd);

    basis::template addInterpFieldsGradTranspose<3, 3>(pt, d0xi.bvalue().get_data(), dd);
    basis::template addInterpFieldsGradTranspose<vars_per_node, 3>(pt, u0xi.bvalue().get_data(), res);

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
    basis::template interpFields<vars_per_node, 3>(pt, dvars, u0dot.value().get_data());
    basis::template interpFields<3, 3>(pt, ddot, d0dot.value().get_data());
    // interpolate second time derivatives
    basis::template interpFields<vars_per_node, 3>(pt, ddvars, u0ddot.get_data());
    basis::template interpFields<3, 3>(pt, dddot, d0ddot.get_data());

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
    basis::template addInterpFieldsTranspose<vars_per_node, 3>(pt, u0dot.bvalue().get_data(), res);
    basis::template addInterpFieldsTranspose<3, 3>(pt, d0dot.value().get_data(), dd);
  }

  // Add the contribution to the residual from the drill strain
  TacsShellAddDrillStrainSens<vars_per_node, offset, basis, director, model>(
      Xdn, fn, vars, XdinvTn, Tn, u0xn, Ctn, detn, res);

  // Add the contributions from the tying strain
  model::template addComputeTyingStrainTranspose<vars_per_node, basis>(
      Xpts, fn, vars, d, dety, res, dd);

  // Add the contributions to the director field
  director::template addDirectorResidual<vars_per_node, offset, num_nodes>(
      vars, dvars, ddvars, fn, dd, res);

  // Add the contribution from the rotation constraint (defined by the
  // rotational parametrization) - if any
  director::template addRotationConstraint<vars_per_node, offset, num_nodes>(
      vars, res);
}

// 
template <class quadrature, class basis, class director, class model>
void TACSShellElement<quadrature, basis, director, model>::addJacobian(
    int elemIndex, double time, TacsScalar alpha, TacsScalar beta,
    TacsScalar gamma, const TacsScalar Xpts[], const TacsScalar vars[],
    const TacsScalar dvars[], const TacsScalar ddvars[], TacsScalar res[],
    TacsScalar mat[]) {
  // Compute the number of quadrature points
  const int nquad = quadrature::getNumQuadraturePoints();

  // printf("Begin addJacobian.. \n");

  // Derivative of the director field
  TacsScalar dd[dsize];
  memset(dd, 0, dsize * sizeof(TacsScalar));

  // Second derivatives required for the director
  TacsScalar d2d[dsize * dsize], d2du[usize * dsize];
  TacsScalar d2Tdotd[dsize * dsize], d2Tdotu[usize * dsize];
  memset(d2d, 0, dsize * dsize * sizeof(TacsScalar));
  memset(d2du, 0, usize * dsize * sizeof(TacsScalar));
  memset(d2Tdotd, 0, dsize * dsize * sizeof(TacsScalar));
  memset(d2Tdotu, 0, usize * dsize * sizeof(TacsScalar));

  // Zero the contributions to the tying strain derivatives
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

  // Compute the node normal directions
  TacsScalar fn[3 * num_nodes], Xdn[9 * num_nodes];
  TacsShellComputeNodeNormals<basis>(Xpts, fn, Xdn);

  // Compute the drill strain penalty at each node
  TacsScalar etn[num_nodes], detn[num_nodes];
  TacsScalar d2etn[num_nodes * num_nodes];
  memset(detn, 0, num_nodes * sizeof(TacsScalar));
  memset(d2etn, 0, num_nodes * num_nodes * sizeof(TacsScalar));

  // Store information about the transformation and derivatives at each node for
  // the drilling degrees of freedom
  TacsScalar XdinvTn[9 * num_nodes], Tn[9 * num_nodes];
  TacsScalar u0xn[9 * num_nodes], Ctn[csize];
  TacsShellComputeDrillStrain<vars_per_node, offset, basis, director, model>(
      transform, Xdn, fn, vars, XdinvTn, Tn, u0xn, Ctn, etn);

  TacsScalar d[dsize], ddot[dsize], dddot[dsize];
  director::template computeDirectorRates<vars_per_node, offset, num_nodes>(
      vars, dvars, ddvars, fn, d, ddot, dddot);

  // Set the total number of tying points needed for this element
  TacsScalar ety[basis::NUM_TYING_POINTS];
  model::template computeTyingStrain<vars_per_node, basis>(Xpts, fn, vars, d, ety);

  // Loop over each quadrature point and add the residual contribution
  for (int quad_index = 0; quad_index < nquad; quad_index++) {
    
    // Get the quadrature weight
    double pt[3];
    double weight = quadrature::getQuadraturePoint(quad_index, pt);

    // printf("iquad = %d\n", quad_index);

    // interpolation section
    // ----------------------------------------

    // passive A2D Objs used in interpolation
    A2D::Vec<TacsScalar,3> X, n0;
    A2D::Mat<TacsScalar,3,2> Xxi, nxi;
    A2D::Mat<TacsScalar,3,3> T;
    
    // active A2D objs used in interpolation
    A2D::A2DObj<A2D::Vec<TacsScalar,1>> et;
    A2D::A2DObj<A2D::Vec<TacsScalar,3>> d0;
    A2D::A2DObj<A2D::Mat<TacsScalar,3,2>> d0xi, u0xi;
    A2D::A2DObj<A2D::SymMat<TacsScalar,3>> e0ty, e0ty_tmp, gty;    

    // interpolate coordinates, director, midplane displacements with the basis
    // tried interpolating U,d => d0, d0xi, u0xi in main stack, but this doesn't help much because need d2gtyu0xi somewhere else
    basis::template interpFields<3, 3>(pt, Xpts, X.get_data());
    basis::template interpFields<3, 3>(pt, fn, n0.get_data());
    basis::template interpFields<1, 1>(pt, etn, et.value().get_data());
    basis::template interpFields<3, 3>(pt, d, d0.value().get_data());

    basis::template interpFieldsGrad<3, 3>(pt, Xpts, Xxi.get_data());
    basis::template interpFieldsGrad<3, 3>(pt, fn, nxi.get_data());
    basis::template interpFieldsGrad<3, 3>(pt, d, d0xi.value().get_data());
    basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, u0xi.value().get_data());

    // too hard to interpolate since different # of tying points for each gij entry
    basis::interpTyingStrain(pt, ety, gty.value().get_data());

    // debug print out intermediate states for interpolations up to this point
    // printf("et = %.8e\n", et.value().get_data()[0]);
    // for (int i = 0; i < 3; i++) {
    //   printf("X[%d] = %.8e\n", i, X.get_data()[i]);
    //   printf("n0[%d] = %.8e\n", i, n0.get_data()[i]);
    // }
    // for (int j = 0; j < 6; j++) {
    //   printf("Xxi[%d] = %.8e\n", j, Xxi.get_data()[j]);
    //   printf("gty[%d] = %.8e\n", j, gty.value().get_data()[j]);
    // }
    // for (int i = 0; i < 3; i++) {
    //   printf("d0[%d] = %.8e\n", i, d0.value().get_data()[i]);
    // }
    // for (int j = 0; j < 6; j++) {
    //   printf("nxi[%d] = %.8e\n", j, nxi.get_data()[j]);
    //   printf("d0xi[%d] = %.8e\n", j, d0xi.value().get_data()[j]);
    //   printf("u0xi[%d] = %.8e\n", j, u0xi.value().get_data()[j]);
    // }

    // setup before A2D strain energy stack
    // ------------------------------------

    // Compute the transformation at the quadrature point
    transform->computeTransform(Xxi.get_data(), n0.get_data(), T.get_data()); 

    // compute ABD matrix from shell theory (prospective)
    // A2D::SymMat<TacsScalar,9> ABD; // normally ABD is 6x6, but this one includes transverse shear and drill strains
    A2D::Mat<TacsScalar,9,9> ABD; // A2D doesn't handle mixed symmat * vec well right now..
    con->getABDmatrix(0, pt, X.get_data(), ABD.get_data()); // TODO make this routine

    // passive variables for strain energy stack
    A2D::Vec<TacsScalar, 3> zero;
    // A2D::Mat<TacsScalar, 3, 3> Xd, Xdz, Xdinv, XdinvT;
    // TacsScalar *detXd; // should be able to make this not an A2DObj

    // active variables for strain energy stack
    A2D::A2DObj<TacsScalar> detXd, ES_dot, Uelem;
    
    A2D::A2DObj<A2D::Mat<TacsScalar, 3, 3>> Xd, Xdz, Xdinv, XdinvT;

    A2D::A2DObj<A2D::Vec<TacsScalar,9>> E, S;
    A2D::A2DObj<A2D::Mat<TacsScalar, 3, 3>> u0x_tmp, u1x_tmp1, u1x_tmp2, u1x_tmp3, u1x_term1, u1x_term2, u1x_sum; // temp variables
    A2D::A2DObj<A2D::Mat<TacsScalar, 3, 3>> u0xi_frame, u1xi_frame, u0x, u1x;

    const A2D::MatOp NORMAL = A2D::MatOp::NORMAL, TRANSPOSE = A2D::MatOp::TRANSPOSE;
    const A2D::ShellStrainType STRAIN_TYPE = A2D::ShellStrainType::LINEAR; // if condition on type of model here..

    // printf("Pre strain energy stack\n");

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

    // for (int j = 0; j < 6; j++) {
    //   printf("e0ty[%d] = %.8e\n", j, e0ty.value().get_data()[j]);
    // }
    // for (int i = 0; i < 9; i++) {
    //   printf("u0x[%d] = %.8e\n", i, u0x.value().get_data()[i]);
    //   printf("u1x[%d] = %.8e\n", i, u1x.value().get_data()[i]);
    //   printf("E[%d] = %.8e\n", i, E.value().get_data()[i]);
    //   printf("S[%d] = %.8e\n", i, S.value().get_data()[i]);
    // }
    printf("Uelem = %.8e\n", Uelem.value());
    // for (int i = 0; i < 9; i++) {
    //   printf("Xd[%d] = %.8e\n", i, Xd.value().get_data()[i]);
    //   printf("Xdz[%d] = %.8e\n", i, Xdz.value().get_data()[i]);
    //   printf("Xdinv[%d] = %.8e\n", i, Xdinv.value().get_data()[i]);
    //   printf("XdinvT[%d] = %.8e\n", i, XdinvT.value().get_data()[i]);     
    // }

    // printf("Post strain energy stack defn\n");

    // reverse mode 1st order AD for the strain energy stack
    // -------------------------------------------------
    Uelem.bvalue() = 1.0;
    Uelem.hvalue() = 0.0;
    // strain_energy_stack.reverse(); // don't want to call this twice (called in hextract)

    // reverse mode 1st + 2nd order AD for the strain energy stack
    // -----------------------------------------------------

    // create submatrix Hessian matrices
    A2D::Mat<TacsScalar,1,1> d2et;
    A2D::Mat<TacsScalar, 3, 3> d2d0;
    A2D::Mat<TacsScalar, 3, 6> d2d0d0xi, d2d0u0xi;
    A2D::Mat<TacsScalar, 6, 3> d2gtyd0;
    A2D::Mat<TacsScalar, 6, 6> d2d0xi, d2d0xiu0xi, d2u0xi, d2gty, d2gtyd0xi, d2gtyu0xi;

    // debugging
    A2D::Mat<TacsScalar, 6, 6> d2e0ty;

    // d0, d0xi, u0xi, et, gty
    const int ncomp = 22;
    A2D::SymMat<TacsScalar, ncomp> hess;
    // auto in = A2D::MakeTieTuple<TacsScalar, A2D::ADseed::p>(d0, d0xi, u0xi, gty, et);
    // auto out = A2D::MakeTieTuple<TacsScalar, A2D::ADseed::h>(d0, d0xi, u0xi, gty, et);

    // debugging get e0ty instead of gty
    auto in = A2D::MakeTieTuple<TacsScalar, A2D::ADseed::p>(d0, d0xi, u0xi, e0ty, et);
    auto out = A2D::MakeTieTuple<TacsScalar, A2D::ADseed::h>(d0, d0xi, u0xi, e0ty, et);
    // strain_energy_stack.hextract(in, out, hess);

    // try my own custom hextract since some hvalues are not getting re-zeroed correctly and build up magnitude
    // maybe can fix this internally in a2d.. but not yet, try this first (d2gty was not right with regular hextract)
    strain_energy_stack.reverse();
    for (A2D::index_t icol = 0; icol < 22; icol++) {
      in.zero();
      Uelem.pvalue() = 0.0; // should compute nonzero hvalue and multiply by this or not? prob doesn't matter

      in[icol] = 1.0;
      strain_energy_stack.hforward();

      // clear all input hvalues (this is what is not in original hextract that somehow needs to be)
      out.zero();
      strain_energy_stack.hreverse();

      // extract hessian at input level
      for (A2D::index_t irow = 0; irow < 22; irow++) {
        hess(irow,icol) = out[irow];
      }
    }

    // try for debugging
    // // strain_energy_stack.hextract(d0.pvalue(), d0.hvalue(), d2d0);
    // // strain_energy_stack.hextract(d0.pvalue(), d0xi.hvalue(), d2d0d0xi);
    // // strain_energy_stack.hextract(d0.pvalue(), u0xi.hvalue(), d2d0u0xi);
    // strain_energy_stack.hextract(d0xi.pvalue(), d0xi.hvalue(), d2d0xi);
    // // strain_energy_stack.hextract(d0xi.pvalue(), u0xi.hvalue(), d2d0xiu0xi);
    // // strain_energy_stack.hextract(u0xi.pvalue(), u0xi.hvalue(), d2u0xi);

    // printf("Post strain energy stack.reverse\n");

    // debug first order derivs
    // printf("det %.8e\n", et.bvalue()[0]);
    // for (int i = 0; i < 3; i++) {
    //   printf("dd0[%d] %.8e\n", i, d0.bvalue().get_data()[i]);
    // }
    // for (int j = 0; j < 6; j++) {
    //   printf("dd0xi[%d] %.8e\n", j, d0xi.bvalue()[j]);
    //   printf("du0xi[%d] %.8e\n", j, u0xi.bvalue()[j]);
    // }
    // for (int k = 0; k < 6; k++) {
    //   printf("de0ty[%d] %.8e\n", k, e0ty.bvalue().get_data()[k]);
    //   printf("dgty[%d] %.8e\n", k, gty.bvalue().get_data()[k]);
    // }
    
    // reverse through the basis back to the director class, drill strain, tying strain
    basis::template addInterpFieldsTranspose<1, 1>(pt, et.bvalue().get_data(), detn);
    basis::template addInterpFieldsTranspose<3, 3>(pt, d0.bvalue().get_data(), dd);

    basis::template addInterpFieldsGradTranspose<3, 3>(pt, d0xi.bvalue().get_data(), dd);
    basis::template addInterpFieldsGradTranspose<vars_per_node, 3>(pt, u0xi.bvalue().get_data(), res);

    basis::addInterpTyingStrainTranspose(pt, gty.bvalue().get_data(), dety);

    // copy values from full hessian into submatrix-Hessians
    // could add A2D routines to extract submatrices in the future using upper, lower bounds maybe?
    for (int irow = 0; irow < 22; irow++) {
      for (int icol = 0; icol < 22; icol++) {
        // start of large irow if block
        if (irow < 3) { // d0 rows
          if (icol < 3) { // d0 cols
            d2d0(irow, icol) = hess(irow,icol);
          } else if (3 <= icol && icol < 9) { // d0xi cols
            d2d0d0xi(irow, icol-3) = hess(irow,icol);
          } else if (9 <= icol && icol < 15) { // u0xi cols
            d2d0xiu0xi(irow, icol-9) = hess(irow, icol);
          }
        } else if (3 <= irow && irow < 9) { // d0xi rows
          if (3 <= icol && icol < 9) { // d0xi cols
            d2d0xi(irow-3, icol-3) = hess(irow,icol);
          } else if (9 <= icol && icol < 15) { // u0xi calls
            d2d0xiu0xi(irow-3, icol-9) = hess(irow,icol);
          }
        } else if (9 <= irow && irow < 15) { // u0xi rows
          if (9 <= icol && icol < 15) { // u0xi cols
            d2u0xi(irow-9, icol-9) = hess(irow,icol);
          }
        } else if (15 <= irow && irow < 21) { // gty rows
          if (0 <= icol && icol < 3) { // d0 cols
            d2gtyd0(irow, icol-15) = hess(irow,icol);
          } else if (3 <= icol && icol < 9) { // d0xi cols
            d2gtyd0xi(irow-15, icol-3) = hess(irow,icol);
          } else if (9 <= icol && icol < 15) { // u0xi cols
            d2gtyu0xi(irow-15, icol-9) = hess(irow,icol);
          } else if (15 <= icol && icol < 21) { // gty cols
            // d2gty(irow-15, icol-15) = hess(irow, icol);
            d2e0ty(irow-15, icol-15) = hess(irow, icol);
          }
        } // done with large irow if block 
      } // end of icol for loop
    } // end of icol for loop
    d2et(0,0) = hess(21, 21);

    // debug compare the 2nd derivatives with orig shell element via printout
    // in main addJacobian of orig
    printf("d2et = %.8e\n", d2et.get_data()[0]);

    // in TacsShellAddDispGradHessian of orig
    for (int i1 = 0; i1 < 9; i1++) {
      printf("d2d0[%d] = %.8e\n", i1, d2d0.get_data()[i1]);
    }
    for (int i2 = 0; i2 < 18; i2++) {
      printf("d2d0d0xi[%d] = %.8e\n", i2, d2d0d0xi.get_data()[i2]);
    }
    for (int i3 = 0; i3 < 18; i3++) {
      printf("d2d0u0xi[%d] = %.8e\n", i3, d2d0u0xi.get_data()[i3]);
    }
    for (int i5 = 0; i5 < 36; i5++) {
      printf("d2d0xi[%d] = %.8e\n", i5, d2d0xi.get_data()[i5]);
    }
    for (int i6 = 0; i6 < 36; i6++) {
      printf("d2d0xiu0xi[%d] = %.8e\n", i6, d2d0xiu0xi.get_data()[i6]);
    }
    for (int i7 = 0; i7 < 36; i7++) {
      printf("d2u0xi[%d] = %.8e\n", i7, d2u0xi.get_data()[i7]);
    }

    for (int i8 = 0; i8 < 36; i8++) {
      printf("d2e0ty[%d] = %.8e\n", i8, d2e0ty.get_data()[i8]);
    }

    // in main addJacobian of orig
    // for (int i8 = 0; i8 < 36; i8++) {
    //   printf("d2gty[%d] = %.8e\n", i8, d2gty.get_data()[i8]);
    // }

    // matches TacsShellAddTyingDispCoupling
    // for (int i4 = 0; i4 < 18; i4++) {
    //   printf("d2gtyd0[%d] = %.8e\n", i4, d2gtyd0.get_data()[i4]);
    // }
    // for (int i9 = 0; i9 < 36; i9++) {
    //   printf("d2gtyd0xi[%d] = %.8e\n", i9, d2gtyd0xi.get_data()[i9]);
    // }
    // for (int i10 = 0; i10 < 36; i10++) {
    //   printf("d2gtyu0xi[%d] = %.8e\n", i10, d2gtyu0xi.get_data()[i10]);
    // }

    // Hessian backprop from quad level to nodes level
    basis::template addInterpFieldsOuterProduct<1, 1, 1, 1>(pt, d2et.get_data(), d2etn);
    basis::template addInterpFieldsOuterProduct<3, 3, 3, 3>(pt, d2d0.get_data(), d2d);
    basis::template addInterpGradOuterProduct<3, 3, 3, 3>(pt, d2d0xi.get_data(), d2d);
    basis::template addInterpGradMixedOuterProduct<3, 3, 3, 3>(pt, d2d0d0xi.get_data(), d2d0d0xi.get_data(), d2d);
    basis::template addInterpGradMixedOuterProduct<3, 3, 3, 3>(pt, d2d0u0xi.get_data(), NULL, d2du);
    basis::template addInterpGradOuterProduct<3, 3, 3, 3>(pt, d2d0xiu0xi.get_data(), d2du);
    if (mat) {
      basis::template addInterpGradOuterProduct<vars_per_node, vars_per_node, 3, 3>(pt, d2u0xi.get_data(), mat);
    }
    basis::addInterpTyingStrainHessian(pt, d2gty.get_data(), d2ety);  

    // double check this new routine.. => not sure if this part is right..
    TacsShellAddTyingDispCouplingPostStack<basis>(pt, 
      d2gtyd0.get_data(), d2gtyd0xi.get_data(), d2gtyu0xi.get_data(), 
      d2etyu, d2etyd);

    // setup before kinetic energy stack
    // ------------------------------------

    // passive variables
    A2D::Vec<TacsScalar,3> moments, u0ddot, d0ddot;

    // active variables
    A2D::A2DObj<TacsScalar> uu_term, ud_term1, ud_term2, dd_term, dTelem_dt;
    A2D::A2DObj<A2D::Vec<TacsScalar,3>> u0dot, d0dot;

    // evaluate mass moments
    con->evalMassMoments(elemIndex, pt, X.get_data(), moments.get_data());
    // interpolate first time derivatives
    basis::template interpFields<vars_per_node, 3>(pt, dvars, u0dot.value().get_data());
    basis::template interpFields<3, 3>(pt, ddot, d0dot.value().get_data());
    // interpolate second time derivatives
    basis::template interpFields<vars_per_node, 3>(pt, ddvars, u0ddot.get_data());
    basis::template interpFields<3, 3>(pt, dddot, d0ddot.get_data());

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
    dTelem_dt.hvalue() = 0.0;
    kinetic_energy_stack.reverse();

    // backpropagate the time derivatives to the residual
    basis::template addInterpFieldsTranspose<vars_per_node, 3>(pt, u0dot.bvalue().get_data(), res);
    basis::template addInterpFieldsTranspose<3, 3>(pt, d0dot.value().get_data(), dd);

    // now get the Hessians (but slightly different formulation here..)
    A2D::Mat<TacsScalar, 3, 3> d2u0dot, d2u0dotd0dot, d2d0dot, d2u0dot_scaled;
    kinetic_energy_stack.hextract(u0dot.pvalue(), u0dot.hvalue(), d2u0dot);
    kinetic_energy_stack.hextract(u0dot.pvalue(), d0dot.hvalue(), d2u0dotd0dot);
    kinetic_energy_stack.hextract(d0dot.pvalue(), d0dot.hvalue(), d2d0dot);
    // how to scale d2u0dot by gamma easily?
    for (int i = 0; i < 9; i++) {
      d2u0dot_scaled.get_data()[i] = gamma * d2u0dot.get_data()[i];
    }
    // A2D::MatScale<TacsScalar, 3, 3>(gamma_, d2u0dot.get_data(), d2u0dot_scaled.get_data());

    basis::template addInterpFieldsOuterProduct<vars_per_node, vars_per_node, 3, 3>(pt, d2u0dot_scaled.get_data(), mat);
    basis::template addInterpFieldsOuterProduct<3, 3, 3, 3>(pt, d2u0dotd0dot.get_data(), d2Tdotu);
    basis::template addInterpFieldsOuterProduct<3, 3, 3, 3>(pt, d2d0dot.get_data(), d2Tdotd);
  }

  // printf("Done with quad loop\n");

  // is it possible to A2D the nodal steps? => maybe too hard?

  // Add the contribution to the residual from the drill strain
  TacsShellAddDrillStrainHessian<vars_per_node, offset, basis, director, model>(
      Xdn, fn, vars, XdinvTn, Tn, u0xn, Ctn, detn, d2etn, res, mat);

  // Add the residual from the tying strain
  model::template addComputeTyingStrainTranspose<vars_per_node, basis>(
      Xpts, fn, vars, d, dety, res, dd);

  // Add the second order terms from the tying strain
  model::template addComputeTyingStrainHessian<vars_per_node, basis>(
      alpha, Xpts, fn, vars, d, dety, d2ety, d2etyu, d2etyd, mat, d2d, d2du);

  // Add the contributions to the stiffness matrix
  director::template addDirectorJacobian<vars_per_node, offset, num_nodes>(
      alpha, beta, gamma, vars, dvars, ddvars, fn, dd, d2Tdotd, d2Tdotu, d2d,
      d2du, res, mat);

  // Add the constraint associated with the rotational parametrization (if any)
  director::template addRotationConstrJacobian<vars_per_node, offset,
                                               num_nodes>(alpha, vars, res,
                                                          mat);

  // check the values in the matrix (compare for debug) 
  int index = 0;
  for (int irow = 0; irow < 24; irow++) {
    for (int icol = 0; icol < 24; icol++, ++index) {
      printf("Kelem[%d,%d] = %.8e\n", irow, icol, mat[index]);
    }
  }

  // printf("Done with addJacobian on elem %d\n", elemIndex);
}

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
  TacsShellComputeNodeNormals<basis>(Xpts, fn, Xdn);

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
    basis::template interpFields<3, 3>(pt, Xpts, X);
    basis::template interpFieldsGrad<3, 3>(pt, Xpts, Xxi);
    basis::template interpFields<3, 3>(pt, fn, n0);
    basis::template interpFields<1, 1>(pt, etn, &et);
    basis::template interpFields<1, 1>(pt, etnd, &etd);

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
    basis::template interpFields<vars_per_node, 3>(pt, ddvars, u0ddot);
    basis::template interpFields<3, 3>(pt, dddot, d0ddot);

    TacsScalar du0ddot[3], dd0ddot[3];
    basis::template interpFields<vars_per_node, 3>(pt, psi, du0ddot);
    basis::template interpFields<3, 3>(pt, dd, dd0ddot);

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
  TacsShellComputeNodeNormals<basis>(Xpts, fn);

  if (quantityType == TACS_FAILURE_INDEX ||
      quantityType == TACS_STRAIN_ENERGY_DENSITY) {
    if (quantity) {
      // Compute the director rates
      TacsScalar d[dsize], ddot[dsize];
      director::template computeDirectorRates<vars_per_node, offset, num_nodes>(
          vars, dvars, fn, d, ddot);

      // Set the total number of tying points needed for this element
      TacsScalar ety[basis::NUM_TYING_POINTS];
      model::template computeTyingStrain<vars_per_node, basis>(Xpts, fn, vars,
                                                               d, ety);

      // Compute X, X,xi and the interpolated normal n0
      TacsScalar X[3], Xxi[6], n0[3], T[9];
      basis::template interpFields<3, 3>(pt, Xpts, X);
      basis::template interpFieldsGrad<3, 3>(pt, Xpts, Xxi);
      basis::template interpFields<3, 3>(pt, fn, n0);

      // Compute the transformation at the quadrature point
      transform->computeTransform(Xxi, n0, T);

      // Evaluate the displacement gradient at the point
      TacsScalar XdinvT[9], XdinvzT[9];
      TacsScalar u0x[9], u1x[9];
      *detXd = TacsShellComputeDispGrad<vars_per_node, basis>(
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
      basis::template interpFields<3, 3>(pt, Xpts, X);
      basis::template interpFieldsGrad<3, 3>(pt, Xpts, Xxi);
      basis::template interpFields<3, 3>(pt, fn, n0);

      TacsScalar Xd[9];
      TacsShellAssembleFrame(Xxi, n0, Xd);
      *detXd = det3x3(Xd);
      *quantity = con->evalDensity(elemIndex, pt, X);
    }

    return 1;
  } else if (quantityType == TACS_ELEMENT_DISPLACEMENT) {
    if (quantity) {
      TacsScalar Xxi[6], n0[3], X[3];
      basis::template interpFields<3, 3>(pt, Xpts, X);
      basis::template interpFieldsGrad<3, 3>(pt, Xpts, Xxi);
      basis::template interpFields<3, 3>(pt, fn, n0);

      TacsScalar Xd[9];
      TacsShellAssembleFrame(Xxi, n0, Xd);
      *detXd = det3x3(Xd);

      // Compute the interpolated displacements
      basis::template interpFields<vars_per_node, 3>(pt, vars, quantity);
    }

    return 3;
  } else if (quantityType == TACS_ELEMENT_DENSITY_MOMENT) {
    if (quantity) {
      TacsScalar Xxi[6], n0[3], X[3];
      basis::template interpFields<3, 3>(pt, Xpts, X);
      basis::template interpFieldsGrad<3, 3>(pt, Xpts, Xxi);
      basis::template interpFields<3, 3>(pt, fn, n0);

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
      basis::template interpFields<3, 3>(pt, Xpts, X);
      basis::template interpFieldsGrad<3, 3>(pt, Xpts, Xxi);
      basis::template interpFields<3, 3>(pt, fn, n0);

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
      basis::template interpFields<3, 3>(pt, Xpts, X);
      basis::template interpFieldsGrad<3, 3>(pt, Xpts, Xxi);
      basis::template interpFields<3, 3>(pt, fn, n0);

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
    TacsShellComputeNodeNormals<basis>(Xpts, fn);

    // Compute the director rates
    TacsScalar d[dsize], ddot[dsize];
    director::template computeDirectorRates<vars_per_node, offset, num_nodes>(
        vars, dvars, fn, d, ddot);

    // Set the total number of tying points needed for this element
    TacsScalar ety[basis::NUM_TYING_POINTS];
    model::template computeTyingStrain<vars_per_node, basis>(Xpts, fn, vars, d,
                                                             ety);

    // Compute X, X,xi and the interpolated normal n0
    TacsScalar X[3], Xxi[6], n0[3], T[9];
    basis::template interpFields<3, 3>(pt, Xpts, X);
    basis::template interpFieldsGrad<3, 3>(pt, Xpts, Xxi);
    basis::template interpFields<3, 3>(pt, fn, n0);

    // Compute the transformation at the quadrature point
    transform->computeTransform(Xxi, n0, T);

    // Evaluate the displacement gradient at the point
    TacsScalar XdinvT[9], XdinvzT[9];
    TacsScalar u0x[9], u1x[9];
    TacsShellComputeDispGrad<vars_per_node, basis>(
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
    basis::template interpFields<3, 3>(pt, Xpts, X);

    con->addDensityDVSens(elemIndex, scale * dfdq[0], pt, X, dvLen, dfdx);
  } else if (quantityType == TACS_ELEMENT_DENSITY_MOMENT) {
    // Compute the node normal directions
    TacsScalar fn[3 * num_nodes];
    TacsShellComputeNodeNormals<basis>(Xpts, fn);

    // Compute X and the interpolated normal n0
    TacsScalar X[3], n0[3];
    basis::template interpFields<3, 3>(pt, Xpts, X);
    basis::template interpFields<3, 3>(pt, fn, n0);

    TacsScalar dfdmoments[3] = {0.0};

    for (int i = 0; i < 3; i++) {
      dfdmoments[0] += scale * dfdq[i] * X[i];
      dfdmoments[1] += scale * dfdq[i] * n0[i];
    }

    con->addMassMomentsDVSens(elemIndex, pt, X, dfdmoments, dvLen, dfdx);
  } else if (quantityType == TACS_ELEMENT_MOMENT_OF_INERTIA) {
    // Compute the node normal directions
    TacsScalar fn[3 * num_nodes];
    TacsShellComputeNodeNormals<basis>(Xpts, fn);

    // Compute X, X,xi and the interpolated normal n0
    TacsScalar X[3], Xxi[6], n0[3], T[9];
    basis::template interpFields<3, 3>(pt, Xpts, X);
    basis::template interpFieldsGrad<3, 3>(pt, Xpts, Xxi);
    basis::template interpFields<3, 3>(pt, fn, n0);

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
    TacsShellComputeNodeNormals<basis>(Xpts, fn, Xdn);

    TacsScalar d[dsize], ddot[dsize], dddot[dsize];
    director::template computeDirectorRates<vars_per_node, offset, num_nodes>(
        vars, dvars, ddvars, fn, d, ddot, dddot);

    // Set the total number of tying points needed for this element
    TacsScalar ety[basis::NUM_TYING_POINTS];
    model::template computeTyingStrain<vars_per_node, basis>(Xpts, fn, vars, d,
                                                             ety);

    // Compute X, X,xi and the interpolated normal n0
    TacsScalar X[3], Xxi[6], n0[3], T[9];
    basis::template interpFields<3, 3>(pt, Xpts, X);
    basis::template interpFieldsGrad<3, 3>(pt, Xpts, Xxi);
    basis::template interpFields<3, 3>(pt, fn, n0);

    // Compute the transformation at the quadrature point
    transform->computeTransform(Xxi, n0, T);

    // Evaluate the displacement gradient at the point
    TacsScalar XdinvT[9], XdinvzT[9];
    TacsScalar u0x[9], u1x[9];
    TacsShellComputeDispGrad<vars_per_node, basis>(
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
    model::evalStrainSens(alpha * dfdq[0], sens, u0x, u1x, du0x, du1x, de0ty);

    // Add the contributions to the residual from du0x, du1x and dCt
    TacsShellAddDispGradSens<vars_per_node, basis>(pt, T, XdinvT, XdinvzT, du0x,
                                                   du1x, dfdu, dd);

    // Compute the of the tying strain w.r.t. derivative w.r.t. the coefficients
    TacsScalar dgty[6];
    mat3x3SymmTransformTransSens(XdinvT, de0ty, dgty);

    // Evaluate the tying strain
    basis::addInterpTyingStrainTranspose(pt, dgty, dety);

    // Set the total number of tying points needed for this element
    model::template addComputeTyingStrainTranspose<vars_per_node, basis>(
        Xpts, fn, vars, d, dety, dfdu, dd);

    // Add the contributions to the director field
    director::template addDirectorResidual<vars_per_node, offset, num_nodes>(
        vars, dvars, ddvars, fn, dd, dfdu);
  } else if (quantityType == TACS_ELEMENT_DISPLACEMENT) {
    // Compute the interpolated displacements
    basis::template addInterpFieldsTranspose<vars_per_node, 3>(pt, dfdq, dfdu);
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
    TacsShellComputeNodeNormals<basis>(Xpts, fn, Xdn);

    // Store information about the transformation and derivatives at each node
    // for the drilling degrees of freedom
    TacsScalar etn[num_nodes];
    TacsScalar XdinvTn[9 * num_nodes], Tn[9 * num_nodes];
    TacsScalar u0xn[9 * num_nodes], Ctn[csize];
    TacsShellComputeDrillStrain<vars_per_node, offset, basis, director, model>(
        transform, Xdn, fn, vars, XdinvTn, Tn, u0xn, Ctn, etn);

    TacsScalar d[dsize], ddot[dsize], dddot[dsize];
    director::template computeDirectorRates<vars_per_node, offset, num_nodes>(
        vars, dvars, ddvars, fn, d, ddot, dddot);

    // Set the total number of tying points needed for this element
    TacsScalar ety[basis::NUM_TYING_POINTS];
    model::template computeTyingStrain<vars_per_node, basis>(Xpts, fn, vars, d,
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
      basis::template interpFields<3, 3>(pt, Xpts, X);
      basis::template interpFieldsGrad<3, 3>(pt, Xpts, Xxi);
      basis::template interpFields<3, 3>(pt, fn, n0);
      basis::template interpFields<1, 1>(pt, etn, &et);

      // Compute the transformation at the quadrature point
      transform->computeTransform(Xxi, n0, T);

      // Evaluate the displacement gradient at the point
      TacsScalar XdinvT[9], XdinvzT[9];
      TacsScalar u0x[9], u1x[9];
      TacsShellComputeDispGrad<vars_per_node, basis>(
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
    TacsShellComputeNodeNormals<basis>(Xpts, fn, Xdn);

    // Store information about the transformation and derivatives at each node
    // for the drilling degrees of freedom
    TacsScalar etn[num_nodes];
    TacsScalar XdinvTn[9 * num_nodes], Tn[9 * num_nodes];
    TacsScalar u0xn[9 * num_nodes], Ctn[csize];
    TacsShellComputeDrillStrain<vars_per_node, offset, basis, director, model>(
        transform, Xdn, fn, vars, XdinvTn, Tn, u0xn, Ctn, etn);

    TacsScalar d[dsize], ddot[dsize], dddot[dsize];
    director::template computeDirectorRates<vars_per_node, offset, num_nodes>(
        vars, dvars, ddvars, fn, d, ddot, dddot);

    // Set the total number of tying points needed for this element
    TacsScalar ety[basis::NUM_TYING_POINTS];
    model::template computeTyingStrain<vars_per_node, basis>(Xpts, fn, vars, d,
                                                             ety);

    // Loop over each quadrature point and add the residual contribution
    for (int index = 0; index < num_vis_nodes; index++) {
      // Get the quadrature weight
      double pt[3];
      basis::getNodePoint(index, pt);

      // Compute X, X,xi and the interpolated normal n0
      TacsScalar X[3], Xxi[6], n0[3], T[9], et;
      basis::template interpFields<3, 3>(pt, Xpts, X);
      basis::template interpFieldsGrad<3, 3>(pt, Xpts, Xxi);
      basis::template interpFields<3, 3>(pt, fn, n0);
      basis::template interpFields<1, 1>(pt, etn, &et);

      // Compute the transformation at the quadrature point
      transform->computeTransform(Xxi, n0, T);

      // Evaluate the displacement gradient at the point
      TacsScalar XdinvT[9], XdinvzT[9];
      TacsScalar u0x[9], u1x[9];
      TacsShellComputeDispGrad<vars_per_node, basis>(
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
  model::template computeTyingStrain<vars_per_node, basis>(Xpts, fn, vars, d,
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
  model::template addComputeTyingStrainTranspose<vars_per_node, basis>(
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
    model::template computeTyingStrain<vars_per_node, basis>(Xpts, fn, varst, d,
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
    model::template addComputeTyingStrainTranspose<vars_per_node, basis>(
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
    model::template computeTyingStrain<vars_per_node, basis>(Xpts, fn, vars, dt,
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
    model::template addComputeTyingStrainTranspose<vars_per_node, basis>(
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

#endif  // TACS_SHELL_ELEMENT_H