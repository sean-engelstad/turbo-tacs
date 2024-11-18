#ifndef TACS_SHELL_ELEMENT_TRANSFORM_H
#define TACS_SHELL_ELEMENT_TRANSFORM_H

#include "TACSElementAlgebra.h"
#include "TACSObject.h"

/*
  Compute the transformation from the local coordinates
  to
*/
class TACSShellTransform : public TACSObject {
 public:
  /*
    Given the local shell element reference frame Xf, compute the
    transformation from the global coordinates to the shell-aligned local axis.
  */

  // TODO : can't template on virtual functions? (was virtual before here)..
  template <typename T>
  __HOST_DEVICE__ void computeTransform(const T Xxi[], const T n0[],
                                T Tmat[]) {}; // removed virtual here because
};

class TACSShellNaturalTransform : public TACSShellTransform {
 public:
  TACSShellNaturalTransform() {}

  template <typename T>
  __HOST_DEVICE__ void computeTransform(const T Xxi[], const T n0[],
                        T Tmat[]) {
    T n[3];
    n[0] = n0[0];
    n[1] = n0[1];
    n[2] = n0[2];

    // Scale by the normal
    T inv = 1.0 / sqrt(vec3Dot<T>(n, n));
    vec3Scale<T>(inv, n);

    T t1[3];
    t1[0] = Xxi[0];
    t1[1] = Xxi[2];
    t1[2] = Xxi[4];

    T d = vec3Dot<T>(n, t1);
    t1[0] = t1[0] - d * n[0];
    t1[0] = t1[0] - d * n[0];
    t1[0] = t1[0] - d * n[0];

    inv = 1.0 / sqrt(vec3Dot<T>(t1, t1));
    vec3Scale<T>(inv, t1);

    T t2[3];
    crossProduct<T>(n, t1, t2);

    /*

    // Compute the transformation
    TacsScalar t1[3], t2[3];
    t1[0] = Xxi[0];
    t1[1] = Xxi[2];
    t1[2] = Xxi[4];

    t2[0] = Xxi[1];
    t2[1] = Xxi[3];
    t2[2] = Xxi[5];

    // Compute the normal direction
    TacsScalar n[3];
    crossProduct(t1, t2, n);

    // Normalize the normal direction
    TacsScalar invNorm = 1.0/sqrt(vec3Dot(n, n));
    vec3Scale(invNorm, n);

    // Normalize the 1-direction of the element
    TacsScalar inv = 1.0/sqrt(vec3Dot(t1, t1));
    vec3Scale(inv, t1);

    // Take the cross product to determine the 2-direction
    crossProduct(n, t1, t2);
    */

    // Set the components of the transformation
    Tmat[0] = t1[0];
    Tmat[3] = t1[1];
    Tmat[6] = t1[2];

    Tmat[1] = t2[0];
    Tmat[4] = t2[1];
    Tmat[7] = t2[2];

    Tmat[2] = n[0];
    Tmat[5] = n[1];
    Tmat[8] = n[2];
  }
};

class TACSShellRefAxisTransform : public TACSShellTransform {
 public:
  TACSShellRefAxisTransform(const TacsScalar _axis[]) {
    axis[0] = _axis[0];
    axis[1] = _axis[1];
    axis[2] = _axis[2];

    TacsScalar norm = sqrt(vec3Dot<TacsScalar>(axis, axis));
    TacsScalar invNorm = 0.0;
    if (norm != 0.0) {
      invNorm = 1.0 / norm;
    }
    vec3Scale<TacsScalar>(invNorm, axis);
  }

  void getRefAxis(TacsScalar _axis[]) {
    _axis[0] = axis[0];
    _axis[1] = axis[1];
    _axis[2] = axis[2];
  }

  template <typename T>
  __HOST_DEVICE__ void computeTransform(const T Xxi[], const T n0[],
                        T Tmat[]) {
    T n[3];
    n[0] = n0[0];
    n[1] = n0[1];
    n[2] = n0[2];

    // Scale by the normal
    T inv = 1.0 / sqrt(vec3Dot<T>(n, n));
    vec3Scale<T>(inv, n);

    // Compute the dot product with
    T an = vec3Dot<T>(axis, n);

    // Check if ref axis is parallel with normal
    if (abs(TacsRealPart(an)) > 1.0 - SMALL_NUM) {
      fprintf(stderr,
              "TACSShellRefAxisTransform: Error, user-provided reference axis "
              "is perpendicular to shell. "
              "Element behavior may be ill-conditioned.\n");
    }

    // Take the component of the reference axis perpendicular
    // to the surface
    T t1[3];
    t1[0] = axis[0] - an * n[0];
    t1[1] = axis[1] - an * n[1];
    t1[2] = axis[2] - an * n[2];

    // Normalize the new direction
    inv = 1.0 / sqrt(vec3Dot<T>(t1, t1));
    vec3Scale<T>(inv, t1);

    // Take the cross product to determine the 2-direction
    T t2[3];
    crossProduct<T>(n, t1, t2);

    /*
        // Compute the transformation
        TacsScalar t1[3], t2[3];
        t1[0] = Xxi[0];
        t1[1] = Xxi[2];
        t1[2] = Xxi[4];

        t2[0] = Xxi[1];
        t2[1] = Xxi[3];
        t2[2] = Xxi[5];
    */

    /*
    // Compute the transformation
    TacsScalar t1[3], t2[3];
    t1[0] = Xxi[0];
    t1[1] = Xxi[2];
    t1[2] = Xxi[4];

    t2[0] = Xxi[1];
    t2[1] = Xxi[3];
    t2[2] = Xxi[5];

    // Compute the normal direction
    TacsScalar n[3];
    crossProduct(t1, t2, n);

    // Normalize the normal direction
    TacsScalar invNorm = 1.0/sqrt(vec3Dot(n, n));
    vec3Scale(invNorm, n);

    // Compute the dot product with
    TacsScalar an = vec3Dot(axis, n);

    // Take the component of the reference axis perpendicular
    // to the surface
    t1[0] = axis[0] - an*n[0];
    t1[1] = axis[1] - an*n[1];
    t1[2] = axis[2] - an*n[2];

    // Normalize the new direction
    TacsScalar inv = 1.0/sqrt(vec3Dot(t1, t1));
    vec3Scale(inv, t1);

    // Take the cross product to determine the 2-direction
    crossProduct(n, t1, t2);
    */

    // Set the components of the transformation
    Tmat[0] = t1[0];
    Tmat[3] = t1[1];
    Tmat[6] = t1[2];

    Tmat[1] = t2[0];
    Tmat[4] = t2[1];
    Tmat[7] = t2[2];

    Tmat[2] = n[0];
    Tmat[5] = n[1];
    Tmat[8] = n[2];
  }

 private:
  TacsScalar axis[3];
  /* Tolerance for colinearity test in between shell normal and ref axis */
  const double SMALL_NUM = 1e-8;
};

#endif  // TACS_SHELL_ELEMENT_TRANSFORM_H