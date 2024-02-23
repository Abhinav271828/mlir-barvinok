//===- GeneratingFunction.h - Generating Functions over Q^d -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definition of the GeneratingFunction class for Barvinok's algorithm,
// which represents a function over Q^n, parameterized by d parameters.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_GENERATINGFUNCTION_H
#define MLIR_ANALYSIS_PRESBURGER_GENERATINGFUNCTION_H

#include "mlir/Analysis/Presburger/Fraction.h"
#include "mlir/Analysis/Presburger/Matrix.h"
#include "mlir/Analysis/Presburger/PresburgerRelation.h"
#include "mlir/Analysis/Presburger/QuasiPolynomial.h"
#include "llvm/ADT/Sequence.h"

namespace mlir {
namespace presburger {
namespace detail {

// A parametric point is a vector, each of whose elements
// is an affine function of n parameters. Each column
// in the matrix represents the affine function and
// has n+1 elements.
using ParamPoint = FracMatrix;

// A point is simply a vector.
using Point = SmallVector<Fraction>;

// A class to describe the type of generating function
// used to enumerate the integer points in a polytope.
// Consists of a set of terms, where the ith term has
// * a sign, ±1, stored in `signs[i]`
// * a numerator, of the form x^{n},
//      where n, stored in `numerators[i]`,
//      is a parametric point.
// * a denominator, of the form (1 - x^{d1})...(1 - x^{dn}),
//      where each dj, stored in `denominators[i][j]`,
//      is a vector.
//
// Represents functions f_p : Q^n -> Q of the form
//
// f_p(x) = \sum_i s_i * (x^n_i(p)) / (\prod_j (1 - x^d_{ij})
//
// where s_i is ±1,
// n_i \in Q^d -> Q^n is an n-vector of affine functions on d parameters, and
// g_{ij} \in Q^n are vectors.
class GeneratingFunction {
public:
  GeneratingFunction(unsigned numSymbols, SmallVector<int> signs,
                     std::vector<ParamPoint> nums,
                     std::vector<std::vector<Point>> dens)
      : numSymbols(numSymbols), signs(signs), numerators(nums),
        denominators(dens) {
#ifndef NDEBUG
    for (const ParamPoint &term : numerators)
      assert(term.getNumRows() == numSymbols + 1 &&
             "dimensionality of numerator exponents does not match number of "
             "parameters!");
#endif // NDEBUG
  }

  unsigned getNumSymbols() const { return numSymbols; }

  SmallVector<int> getSigns() const { return signs; }

  std::vector<ParamPoint> getNumerators() const { return numerators; }

  std::vector<std::vector<Point>> getDenominators() const {
    return denominators;
  }

  GeneratingFunction operator+(const GeneratingFunction &gf) const {
    assert(numSymbols == gf.getNumSymbols() &&
           "two generating functions with different numbers of parameters "
           "cannot be added!");
    SmallVector<int> sumSigns = signs;
    sumSigns.append(gf.signs);

    std::vector<ParamPoint> sumNumerators = numerators;
    sumNumerators.insert(sumNumerators.end(), gf.numerators.begin(),
                         gf.numerators.end());

    std::vector<std::vector<Point>> sumDenominators = denominators;
    sumDenominators.insert(sumDenominators.end(), gf.denominators.begin(),
                           gf.denominators.end());
    return GeneratingFunction(numSymbols, sumSigns, sumNumerators,
                              sumDenominators);
  }

  /// Find the number of terms in the generating function, as
  /// a quasipolynomial in the parameter space of the input function.
  /// The generating function must be such that for all values of the
  /// parameters, the number of terms is finite.
  QuasiPolynomial computeNumTerms() const;

  /// Find a vector that is not orthogonal to any of the exponents in the
  /// terms' denominators, i.e., has nonzero dot product with those of the
  /// denominator exponents that are not null.
  /// If any of the vectors is null, it is ignored.
  Point getNonOrthogonalVector() const;

  llvm::raw_ostream &print(llvm::raw_ostream &os) const {
    for (int i : llvm::seq<int>(0, signs.size())) {
      if (i == 0) {
        if (signs[i] == -1)
          os << "- ";
      } else {
        if (signs[i] == 1)
          os << " + ";
        else
          os << " - ";
      }

      os << "x^[";
      unsigned r = numerators[i].getNumRows();
      for (unsigned j = 0; j < r - 1; ++j) {
        os << "[";
        for (int k : llvm::seq<int>(0, numerators[i].getNumColumns() - 1))
          os << numerators[i].at(j, k) << ",";
        os << numerators[i].getRow(j).back() << "],";
      }
      os << "[";
      for (int k : llvm::seq<int>(0, numerators[i].getNumColumns() - 1))
        os << numerators[i].at(r - 1, k) << ",";
      os << numerators[i].getRow(r - 1).back() << "]]/";

      for (const Point &den : denominators[i]) {
        os << "(x^[";
        for (int j : llvm::seq<int>(0, den.size() - 1))
          os << den[j] << ",";
        os << den.back() << "])";
      }
    }
    return os;
  }

private:
  unsigned numSymbols;
  SmallVector<int> signs;
  std::vector<ParamPoint> numerators;
  std::vector<std::vector<Point>> denominators;
};

// A class that encodes the count of lattice points in a rational parametric
// polyhedron.
// It maintains a partition of the parameter space into regions (chambers), in
// each of which the count is given by a specific generating function.
// This partition satisfies the properties that
// * the union of all chambers is the complete parameter space.
// * the intersection of any two chambers is not full-dimensional.
class PiecewiseGF {
public:
  // The initial value is simply the universe, associated with an empty
  // generating function.
  PiecewiseGF(unsigned numSymbols)
      : chambers({{PresburgerSet::getUniverse(
                       PresburgerSpace::getSetSpace(numSymbols)),
                   GeneratingFunction(numSymbols, {}, {}, {})}}){};

  unsigned size() { return chambers.size(); };

  PresburgerSet getRegion(unsigned index) { return chambers[index].first; };

  GeneratingFunction getGeneratingFunction(unsigned index) {
    return chambers[index].second;
  };

  // Given a generating function and its corresponding region of activity,
  // update the chamber-GF list by finding where the region intersects the
  // current partitions and partitioning them further if needed.
  void updateWithGF(const PresburgerSet &region, const GeneratingFunction &gf);

  Fraction evaluateAt(SmallVector<Fraction> parameters);

private:
  std::vector<std::pair<PresburgerSet, GeneratingFunction>> chambers;
};

} // namespace detail
} // namespace presburger
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_GENERATINGFUNCTION_H
