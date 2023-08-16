//===- Barvinok.h - MLIR Barvinok's Algorithm -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Functions and classes for Barvinok's algorithm in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_BARVINOK_H
#define MLIR_ANALYSIS_PRESBURGER_BARVINOK_H

#include "mlir/Analysis/Presburger/Fraction.h"
#include "mlir/Analysis/Presburger/Matrix.h"
#include "mlir/Analysis/Presburger/PresburgerSpace.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/Utils.h"
#include "mlir/Support/LogicalResult.h"
#include <optional>

namespace mlir {
namespace presburger {

using PolyhedronH = IntegerRelation;
using PolyhedronV = Matrix<MPInt>;
using ConeH = PolyhedronH;
using ConeV = PolyhedronV;
using Point = MutableArrayRef<Fraction>;

// Describe the type of generating function
// used to enumerate the integer points in a polytope.
// Consists of a set of terms, each having
// * a sign, ±1
// * a numerator, of the form x^{n}
// * a denominator, of the form (1 - x^{d1})...(1 - x^{dn})
class GeneratingFunction
{
public:
    GeneratingFunction(SmallVector<int, 16> s, std::vector<Point> nums, std::vector<std::vector<Point>> dens)
        : signs(s), numerators(nums), denominators(dens) {};

private:
    SmallVector<int, 16> signs;
    std::vector<Point> numerators;
    std::vector<std::vector<Point>> denominators;
};

inline ConeH defineHRep(int num_ineqs, int num_vars)
{
    // We don't distinguish between domain and range variables, so
    // we set the number of domain variables as 0 and the number of
    // range variables as the number of actual variables.
    // There are no symbols (non-parametric for now) and no local
    // (existentially quantified) variables.
    ConeH cone(PresburgerSpace::getRelationSpace(0, num_vars, 0, 0));
    return cone;
}

// Get the index of a cone.
// If it has more rays than the dimension, return 0.
MPInt getIndex(ConeV);

// Get the smallest vector in the basis described by the inverse of the 
// rays of the cone, and the coefficients needed to express it in that basis.
std::pair<Point, SmallVector<MPInt, 16>> getSamplePoint(ConeV, Fraction);

// Get the dual of a cone in H-representation, returning the V-representation of it.
ConeV getDual(ConeH);
// Get the dual of a cone in V-representation, returning the H-representation of it.
ConeH getDual(ConeV);

// Decompose a cone into unimodular cones,
// triangulating it first if it is not simplicial.
SmallVector<std::pair<int, ConeH>, 16> unimodularDecomposition(ConeH);

// Decompose a simplicial cone into unimodular cones.
SmallVector<std::pair<int, ConeV>, 16> unimodularDecompositionSimplicial(int, ConeV);

// Convert a cone in V-representation to a list of the inner normals of its faces.
Matrix<MPInt> generatorsToNormals(ConeV);

// Triangulate a non-simplicial cone into a simplicial cones.
SmallVector<ConeV, 16> triangulate(ConeV);

// Compute the generating function for a unimodular cone.
GeneratingFunction unimodularConeGeneratingFunction(ConeH);

// Compute the generating function for a polytope,
// as the sum of generating functions of its tangent cones.
GeneratingFunction polytopeGeneratingFunction(PolyhedronH);

// Substitute the generating function with the unit vector
// to find the number of terms.
MPInt substituteWithUnitVector(GeneratingFunction);

// Count the number of integer points in a polytope,
// by chaining together `polytopeGeneratingFunction`
// and `substituteWithUnitVector`.
MPInt countIntegerPoints(PolyhedronH);

} // namespace presburger
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_BARVINOK_H