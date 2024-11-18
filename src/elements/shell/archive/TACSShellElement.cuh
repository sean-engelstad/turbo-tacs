// currently discarded or archived kernel code with A2D
// not sure if we can directly use A2D in CUDA right now
// since no __device__ specifiers on A2D code.. TBD on that

template <class quadrature, class basis, class director, class model>
__device__
void TACSShellElement<quadrature, basis, director, model>::addStaticJacobian_kernel(
  int ideriv, int igauss,
  double time, TacsScalar alpha, TacsScalar beta, TacsScalar gamma,
  const TacsScalar Xpts[], const TacsScalar vars[],
  TacsScalar res[], TacsScalar mat[]
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
  // comment out kinetic energy section for now (since too many doubles per thread used maybe)
  // T dU[N], ddU[N];

  // TODO : check Kinetic energy part works too
  for (int ivar = 0; ivar < N; ivar++) {
    U[ivar].value = vars[ivar];
  }
  // instead of if statement in for loop, do this to prevent branch statement problems which would
  // slow down the thread
  U[ideriv].deriv[0] = 1.0;
  
  for (int inode = 0; inode < 3 * num_nodes; inode++) {
    XptsAD[inode] = T(Xpts[inode]);
  }

  // Derivative of the director field and matrix at each 
  T dd[dsize];

  // TODO : make some of these TacsScalar and not T type to remove amount of data on the thread later
  // TODO : check if we can directly mix between T and TacsScalar type

  // TODO : need to make all of these __host__ __device__ functions in TacsUtilities, etc.

  // Compute the node normal directions
  T fn[3 * num_nodes], Xdn[9 * num_nodes];
  TacsShellComputeNodeNormals<T, basis>(XptsAD, fn, Xdn);

  // Compute the drill strain penalty at each node
  T etn[num_nodes], detn[num_nodes];

  // Store information about the transformation and derivatives at each node for
  // the drilling degrees of freedom
  // TODO : may want to also do interpolation in this method as well to reduce load on each thread..
  T XdinvTn[9 * num_nodes], Tn[9 * num_nodes];
  T u0xn[9 * num_nodes], Ctn[csize];
  TacsShellComputeDrillStrain<T, vars_per_node, offset, basis, director, model>(
      transform, Xdn, fn, U, XdinvTn, Tn, u0xn, Ctn, etn);

  T d[dsize];
  // T ddot[dsize], dddot[dsize]; // comment out for just static part to reduce load on each thread
  director::template computeDirectorRates<T, vars_per_node, offset, num_nodes>(U, fn, d);

  // Compute the tying strain values
  T ety[basis::NUM_TYING_POINTS], dety[basis::NUM_TYING_POINTS];
  model::template computeTyingStrain<T, vars_per_node, basis>(XptsAD, fn, U, d, ety);

  // beginning of what was gauss pt loop
  // ------------------------------------
  // now done outside the kernel call for lighter weight kernel

  // TODO : can this be changed back to TacsScalar pt[3] type here? and fix the templates for that
  // Get the quadrature weight
  T pt[3];
  T weight = quadrature::template getQuadraturePoint<T>(igauss, pt);

  // interpolation section
  // ----------------------------------------

  // TODO : can this be changed back to TacsScalar pt[3] type here? and fix the templates for that
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

  // TODO : can this be changed back to TacsScalar/double type here not T? and fix the templates
  // Compute the transformation at the quadrature point
  transform->computeTransform<T>(Xxi.get_data(), n0.get_data(), Tmat.get_data()); 

  // TODO : can this be changed back to TacsScalar/double type here not T? and fix the templates
  // compute ABD matrix from shell theory (prospective)
  A2D::Mat<T,9,9> ABD;
  con->getABDmatrix<T>(0, pt, X.get_data(), ABD.get_data()); // TODO make this routine

  // TODO : can this be changed back to TacsScalar/double type here not T? and fix the 
  // TODO : this is a lot of intermediate objects, if we have 8000 floats we can store on one thread
  // is this too heavyweight?
  // maybe we can free up some data after we are done with it?

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

  // TODO : maybe we can free up the static strain energy objects here?

  // commenting out to reduce load on each thread (bc too many doubles per thread maybe)
  // // setup before kinetic energy stack
  // // ------------------------------------

  // // passive variables
  // A2D::Vec<T,3> moments;
  // A2D::ADObj<A2D::Vec<T,3>> u0ddot, d0ddot; // had to make no longer passive just for forward scalar..

  // // active variables
  // A2D::ADObj<T> uu_term, ud_term1, ud_term2, dd_term, dTelem_dt;
  // A2D::ADObj<A2D::Vec<T,3>> u0dot, d0dot;

  // // evaluate mass moments
  // // don't need elemIndex input (doesn't use it..)
  // con->evalMassMoments<T>(0, pt, X.get_data(), moments.get_data());

  // // interpolate first time derivatives
  // basis::template interpFields<T, vars_per_node, 3>(pt, dU, u0dot.value().get_data());
  // basis::template interpFields<T, 3, 3>(pt, ddot, d0dot.value().get_data());

  // // interpolate second time derivatives
  // basis::template interpFields<T, vars_per_node, 3>(pt, ddU, u0ddot.value().get_data());
  // basis::template interpFields<T, 3, 3>(pt, dddot, d0ddot.value().get_data());

  // // due to integration by parts, residual is based on dT/dt, time derivative of KE so 1st and 2nd time derivatives used
  // //   double check: but Jacobian should be obtained with a cross Hessian d^2(dT/dt)/du0dot/du0ddot (and same for directors d0)
  // auto kinetic_energy_stack = A2D::MakeStack(
  //   A2D::VecDot(u0dot, u0ddot, uu_term),
  //   A2D::VecDot(u0dot, d0ddot, ud_term1),
  //   A2D::VecDot(u0ddot, d0dot, ud_term2),
  //   A2D::VecDot(d0dot, d0ddot, dd_term),
  //   Eval(detXd * (Trev(moments[0]) * uu_term + Trev(moments[1]) * (ud_term1 + ud_term2) + Trev(moments[2]) * dd_term), dTelem_dt)
  // );

  // // now reverse to from dTelem_dt => u0dot, d0dot sensitivities
  // dTelem_dt.bvalue() = T(1.0);
  // kinetic_energy_stack.reverse();

  // // backpropagate the time derivatives to the residual
  // basis::template addInterpFieldsTranspose<T, vars_per_node, 3>(pt, u0dot.bvalue().get_data(), resA2D);
  // basis::template addInterpFieldsTranspose<T, 3, 3>(pt, d0dot.bvalue().get_data(), dd);
  
  // end of what was the gauss pt loop call (now done outside kernel)

  // Add the contribution to the residual from the drill strain
  TacsShellAddDrillStrainSens<T, vars_per_node, offset, basis, director, model>(
      Xdn, fn, U, XdinvTn, Tn, u0xn, Ctn, detn, resA2D);

  // Add the contributions from the tying strain
  model::template addComputeTyingStrainTranspose<T, vars_per_node, basis>(
      XptsAD, fn, U, d, dety, resA2D, dd);

  // Add the contributions to the director field
  // director::template addDirectorResidual<T, vars_per_node, offset, num_nodes>(
  //     U, dU, ddU, fn, dd, resA2D);
  director::template addDirectorResidual<T, vars_per_node, offset, num_nodes>(
      U, fn, dd, resA2D);

  // Add the contribution from the rotation constraint (defined by the
  // rotational parametrization) - if any
  director::template addRotationConstraint<T, vars_per_node, offset, num_nodes>(
      U, resA2D);

  // use forward AD to compute Kelem = dres/dU (TODO : transient case and kinetic energy matrix also)
  if (mat) {
    for (int ivar = 0; ivar < N; ivar++) {
      res[ivar] = resA2D[ivar].value;
      
      // write in the ideriv column of the matrix
      mat[N * ivar + ideriv] = resA2D[ivar].deriv[ideriv];
    }
  }
}