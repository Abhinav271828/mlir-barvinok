//===- Barvinok.cpp - Barvinok's Algorithm ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/Barvinok.h"
#include "mlir/Analysis/Presburger/Utils.h"
#include "llvm/ADT/Sequence.h"
#include <algorithm>
#include <bitset>

using namespace mlir;
using namespace presburger;
using namespace mlir::presburger::detail;

/// Assuming that the input cone is pointed at the origin,
/// converts it to its dual in V-representation.
/// Essentially we just remove the all-zeroes constant column.
ConeV mlir::presburger::detail::getDual(ConeH cone) {
  unsigned numIneq = cone.getNumInequalities();
  unsigned numVar = cone.getNumCols() - 1;
  ConeV dual(numIneq, numVar, 0, 0);
  // Assuming that an inequality of the form
  // a1*x1 + ... + an*xn + b ≥ 0
  // is represented as a row [a1, ..., an, b]
  // and that b = 0.

  for (int i : llvm::seq<int>(0, numIneq)) {
    assert(cone.atIneq(i, numVar) == 0 &&
           "H-representation of cone is not centred at the origin!");
    dual.setRow(i, cone.getInequality(i).take_front(numVar));
  }

  // Now dual is of the form [ [a1, ..., an] , ... ]
  // which is the V-representation of the dual.
  return dual;
}

/// Converts a cone in V-representation to the H-representation
/// of its dual, pointed at the origin (not at the original vertex).
/// Essentially adds a column consisting only of zeroes to the end.
ConeH mlir::presburger::detail::getDual(ConeV cone) {
  unsigned rows = cone.getNumRows();
  unsigned columns = cone.getNumColumns();
  ConeH dual = defineHRep(columns);
  // Add a new column (for constants) at the end.
  // This will be initialized to zero.
  cone.insertColumn(columns);

  for (unsigned i = 0; i < rows; ++i)
    dual.addInequality(cone.getRow(i));

  // Now dual is of the form [ [a1, ..., an, 0] , ... ]
  // which is the H-representation of the dual.
  return dual;
}

/// Find the index of a cone in V-representation.
MPInt mlir::presburger::detail::getIndex(ConeV cone) {
  if (cone.getNumRows() > cone.getNumColumns())
    return MPInt(0);

  return cone.determinant();
}

/// Compute the generating function for a unimodular cone.
/// This consists of a single term of the form
/// sign * x^num / prod_j (1 - x^den_j)
///
/// sign is either +1 or -1.
/// den_j is defined as the set of generators of the cone.
/// num is computed by expressing the vertex as a weighted
/// sum of the generators, and then taking the floor of the
/// coefficients.
GeneratingFunction
mlir::presburger::detail::computeUnimodularConeGeneratingFunction(
    ParamPoint vertex, int sign, ConeH cone) {
  // Consider a cone with H-representation [0  -1].
  //                                       [-1 -2]
  // Let the vertex be given by the matrix [ 2  2   0], with 2 params.
  //                                       [-1 -1/2 1]

  // `cone` must be unimodular.
  assert(abs(getIndex(getDual(cone))) == 1 && "input cone is not unimodular!");

  unsigned numVar = cone.getNumVars();
  unsigned numIneq = cone.getNumInequalities();

  // Thus its ray matrix, U, is the inverse of the
  // transpose of its inequality matrix, `cone`.
  // The last column of the inequality matrix is null,
  // so we remove it to obtain a square matrix.
  FracMatrix transp = FracMatrix(cone.getInequalities()).transpose();
  transp.removeRow(numVar);

  FracMatrix generators(numVar, numIneq);
  transp.determinant(/*inverse=*/&generators); // This is the U-matrix.
  // Thus the generators are given by U = [2  -1].
  //                                      [-1  0]

  // The powers in the denominator of the generating
  // function are given by the generators of the cone,
  // i.e., the rows of the matrix U.
  std::vector<Point> denominator(numIneq);
  ArrayRef<Fraction> row;
  for (auto i : llvm::seq<int>(0, numVar)) {
    row = generators.getRow(i);
    denominator[i] = Point(row);
  }

  // The vertex is v \in Z^{d x (n+1)}
  // We need to find affine functions of parameters λ_i(p)
  // such that v = Σ λ_i(p)*u_i,
  // where u_i are the rows of U (generators)
  // The λ_i are given by the columns of Λ = v^T U^{-1}, and
  // we have transp = U^{-1}.
  // Then the exponent in the numerator will be
  // Σ -floor(-λ_i(p))*u_i.
  // Thus we store the (exponent of the) numerator as the affine function -Λ,
  // since the generators u_i are already stored as the exponent of the
  // denominator. Note that the outer -1 will have to be accounted for, as it is
  // not stored. See end for an example.

  unsigned numColumns = vertex.getNumColumns();
  unsigned numRows = vertex.getNumRows();
  ParamPoint numerator(numColumns, numRows);
  SmallVector<Fraction> ithCol(numRows);
  for (int i : llvm::seq<int>(0, numColumns)) {
    for (int j : llvm::seq<int>(0, numRows))
      ithCol[j] = vertex(j, i);
    numerator.setRow(i, transp.preMultiplyWithRow(ithCol));
    numerator.negateRow(i);
  }
  // Therefore Λ will be given by [ 1    0 ] and the negation of this will be
  //                              [ 1/2 -1 ]
  //                              [ -1  -2 ]
  // stored as the numerator.
  // Algebraically, the numerator exponent is
  // [ -2 ⌊ - N - M/2 + 1 ⌋ + 1 ⌊ 0 + M + 2 ⌋ ] -> first  COLUMN of U is [2, -1]
  // [  1 ⌊ - N - M/2 + 1 ⌋ + 0 ⌊ 0 + M + 2 ⌋ ] -> second COLUMN of U is [-1, 0]

  return GeneratingFunction(numColumns - 1, SmallVector<int>(1, sign),
                            std::vector({numerator}),
                            std::vector({denominator}));
}

/// We use Gaussian elimination to find the solution to a set of d equations
/// of the form
/// a_1 x_1 + ... + a_d x_d + b_1 m_1 + ... + b_p m_p + c = 0
/// where x_i are variables,
/// m_i are parameters and
/// a_i, b_i, c are rational coefficients.
///
/// The solution expresses each x_i as an affine function of the m_i, and is
/// therefore represented as a matrix of size d x (p+1).
/// If there is no solution, we return null.
std::optional<ParamPoint>
mlir::presburger::detail::solveParametricEquations(FracMatrix equations) {
  // equations is a d x (d + p + 1) matrix.
  // Each row represents an equation.
  unsigned d = equations.getNumRows();
  unsigned numCols = equations.getNumColumns();

  // If the determinant is zero, there is no unique solution.
  // Thus we return null.
  if (FracMatrix(equations.getSubMatrix(/*fromRow=*/0, /*toRow=*/d - 1,
                                        /*fromColumn=*/0,
                                        /*toColumn=*/d - 1))
          .determinant() == 0)
    return std::nullopt;

  // Perform row operations to make each column all zeros except for the
  // diagonal element, which is made to be one.
  for (int i : llvm::seq<int>(0, d)) {
    // First ensure that the diagonal element is nonzero, by swapping
    // it with a row that is non-zero at column i.
    if (equations(i, i) != 0)
      continue;
    for (int j : llvm::seq<int>(i + 1, d)) {
      if (equations(j, i) == 0)
        continue;
      equations.swapRows(j, i);
      break;
    }

    Fraction diagElement = equations(i, i);

    // Apply row operations to make all elements except the diagonal to zero.
    for (int j : llvm::seq<int>(0, d)) {
      if (i == j)
        continue;
      if (equations(j, i) == 0)
        continue;
      // Apply row operations to make element (j, i) zero by subtracting the
      // ith row, appropriately scaled.
      Fraction currentElement = equations(j, i);
      equations.addToRow(/*sourceRow=*/i, /*targetRow=*/j,
                         /*scale=*/-currentElement / diagElement);
    }
  }

  // Rescale diagonal elements to 1.
  for (int i : llvm::seq<int>(0, d))
    equations.scaleRow(i, 1 / equations(i, i));

  // Now we have reduced the equations to the form
  // x_i + b_1' m_1 + ... + b_p' m_p + c' = 0
  // i.e. each variable appears exactly once in the system, and has coefficient
  // one.
  //
  // Thus we have
  // x_i = - b_1' m_1 - ... - b_p' m_p - c
  // and so we return the negation of the last p + 1 columns of the matrix.
  //
  // We copy these columns and return them.
  ParamPoint vertex =
      equations.getSubMatrix(/*fromRow=*/0, /*toRow=*/d - 1,
                             /*fromColumn=*/d, /*toColumn=*/numCols - 1);
  vertex.negateMatrix();
  return vertex;
}

/// This is an implementation of the Clauss-Loechner algorithm for chamber
/// decomposition.
///
/// We maintain a list of pairwise disjoint chambers and the generating
/// functions corresponding to each one. We iterate over the list of regions,
/// each time adding the current region's generating function to the chambers
/// where it is active and separating the chambers where it is not.
///
/// Given the region each generating function is active in, for each subset of
/// generating functions the region that (the sum of) precisely this subset is
/// in, is the intersection of the regions that these are active in,
/// intersected with the complements of the remaining regions.
///
/// If the parameter values lie in the intersection of two chambers with
/// differing vertex sets, the answer is given by considering either one.
/// Assume that the chambers differ in that one has k+1 vertices and the other
/// only k – we can then show that the same vertex is counted twice in the
/// former, and so the sets are in fact the same. This is proved below.
/// Consider a parametric polyhedron D in n variables and m parameters. We want
/// to show that if two vertices' chambers intersect, then the vertices collapse
/// in this region.
///
/// Let D' be the polyhedron in combined data-and-parameter space (R^{n+m}).
/// Then we have from [1] that:
///
/// * a vertex is a 0-dimensional face of D, which is equivalent to an m-face
/// F_i^m of D'.
/// * a vertex v_i(p) is obtained from F_i^m(D') by projecting down to n-space
/// (see Fig. 1).
/// * the region a vertex exists in is given by projecting F_i^m(D') down to
/// m-space.
///
/// Thus, if two chambers intersect, this means that the corresponding faces
/// (say F_i^m(D') and F_j^m(D')) also have some intersection. Since the
/// vertices v_i and v_j correspond to the projections of F_i^m(D') and
/// F_j^m(D'), they must collapse in the intersection.
///
/// [1]: Parameterized Polyhedra and Their Vertices (Loechner & Wilde, 1997)
/// https://link.springer.com/article/10.1023/A:1025117523902

PiecewiseGF mlir::presburger::detail::computeChamberDecomposition(
    unsigned numSymbols, ArrayRef<std::pair<PresburgerSet, GeneratingFunction>>
                             regionsAndGeneratingFunctions) {
  assert(!regionsAndGeneratingFunctions.empty() &&
         "there must be at least one chamber!");
  // We maintain a list of regions and their associated generating function
  // initialized with the universe and the empty generating function.
  PiecewiseGF chambers(numSymbols);

  // We iterate over the region list, updating the partition according to the
  // current generating function's activity region.
  for (const auto &[region, generatingFunction] :
       regionsAndGeneratingFunctions) {
    chambers.updateWithGF(region, generatingFunction);
  }

  return chambers;
}

/// For a polytope expressed as a set of n inequalities, compute the generating
/// function corresponding to the lattice points included in the polytope. This
/// algorithm has three main steps:
/// 1. Enumerate the vertices, by iterating over subsets of inequalities and
///    checking for satisfiability. For each d-subset of inequalities (where d
///    is the number of variables), we solve to obtain the vertex in terms of
///    the parameters, and then check for the region in parameter space where
///    this vertex satisfies the remaining (n - d) inequalities.
/// 2. For each vertex, identify the tangent cone and compute the generating
///    function corresponding to it. The generating function depends on the
///    parametric expression of the vertex and the (non-parametric) generators
///    of the tangent cone.
/// 3. [Clauss-Loechner decomposition] Identify the regions in parameter space
///    (chambers) where each vertex is active, and accordingly compute the
///    GF of the polytope in each chamber.
///
/// Verdoolaege, Sven, et al. "Counting integer points in parametric
/// polytopes using Barvinok's rational functions." Algorithmica 48 (2007):
/// 37-66.
PiecewiseGF mlir::presburger::detail::computePolytopeGeneratingFunction(
    const PolyhedronH &poly) {
  unsigned numVars = poly.getNumRangeVars();
  unsigned numSymbols = poly.getNumSymbolVars();
  unsigned numIneqs = poly.getNumInequalities();

  // We store a list of the computed vertices.
  std::vector<ParamPoint> vertices;
  // For each vertex, we store the corresponding active region and the
  // generating functions of the tangent cone, in order.
  std::vector<std::pair<PresburgerSet, GeneratingFunction>>
      regionsAndGeneratingFunctions;

  // We iterate over all subsets of inequalities with cardinality numVars,
  // using permutations of numVars 1's and (numIneqs - numVars) 0's.
  //
  // For a given permutation, we consider a subset which contains
  // the i'th inequality if the i'th bit in the bitset is 1.
  //
  // We start with the permutation that takes the last numVars inequalities.
  SmallVector<int> indicator(numIneqs);
  for (int i : llvm::seq<int>(numIneqs - numVars, numIneqs))
    indicator[i] = 1;

  do {
    // Collect the inequalities corresponding to the bits which are set
    // and the remaining ones.
    auto [subset, remainder] = poly.getInequalities().splitByBitset(indicator);
    // All other inequalities are stored in a2 and b2c2.
    //
    // These are column-wise splits of the inequalities;
    // a2 stores the coefficients of the variables, and
    // b2c2 stores the coefficients of the parameters and the constant term.
    FracMatrix a2(numIneqs - numVars, numVars);
    FracMatrix b2c2(numIneqs - numVars, numSymbols + 1);
    a2 = FracMatrix(
        remainder.getSubMatrix(0, numIneqs - numVars - 1, 0, numVars - 1));
    b2c2 = FracMatrix(remainder.getSubMatrix(0, numIneqs - numVars - 1, numVars,
                                             numVars + numSymbols));

    // Find the vertex, if any, corresponding to the current subset of
    // inequalities.
    std::optional<ParamPoint> vertex =
        solveParametricEquations(FracMatrix(subset)); // d x (p+1)

    if (!vertex)
      continue;
    if (std::find(vertices.begin(), vertices.end(), vertex) != vertices.end())
      continue;
    // If this subset corresponds to a vertex that has not been considered,
    // store it.
    vertices.push_back(*vertex);

    // If a vertex is formed by the intersection of more than d facets, we
    // assume that any d-subset of these facets can be solved to obtain its
    // expression. This assumption is valid because, if the vertex has two
    // distinct parametric expressions, then a nontrivial equality among the
    // parameters holds, which is a contradiction as we know the parameter
    // space to be full-dimensional.

    // Let the current vertex be [X | y], where
    // X represents the coefficients of the parameters and
    // y represents the constant term.
    //
    // The region (in parameter space) where this vertex is active is given
    // by substituting the vertex into the *remaining* inequalities of the
    // polytope (those which were not collected into `subset`), i.e., into the
    // inequalities [A2 | B2 | c2].
    //
    // Thus, the coefficients of the parameters after substitution become
    // (A2 • X + B2)
    // and the constant terms become
    // (A2 • y + c2).
    //
    // The region is therefore given by
    // (A2 • X + B2) p + (A2 • y + c2) ≥ 0
    //
    // This is equivalent to A2 • [X | y] + [B2 | c2].
    //
    // Thus we premultiply [X | y] with each row of A2
    // and add each row of [B2 | c2].
    FracMatrix activeRegion(numIneqs - numVars, numSymbols + 1);
    for (int i : llvm::seq<int>(0, numIneqs - numVars)) {
      activeRegion.setRow(i, vertex->preMultiplyWithRow(a2.getRow(i)));
      activeRegion.addToRow(i, b2c2.getRow(i), 1);
    }

    // We convert the representation of the active region to an integers-only
    // form so as to store it as a PresburgerSet.
    IntegerPolyhedron activeRegionRel(
        PresburgerSpace::getRelationSpace(0, numSymbols, 0, 0), activeRegion);

    // Now, we compute the generating function at this vertex.
    // We collect the inequalities corresponding to each vertex to compute
    // the tangent cone at that vertex.
    //
    // We only need the coefficients of the variables (NOT the parameters)
    // as the generating function only depends on these.
    // We translate the cones to be pointed at the origin by making the
    // constant terms zero.
    ConeH tangentCone = defineHRep(numVars);
    for (int j : llvm::seq<int>(0, subset.getNumRows())) {
      SmallVector<MPInt> ineq(numVars + 1);
      for (int k : llvm::seq<int>(0, numVars))
        ineq[k] = subset(j, k);
      tangentCone.addInequality(ineq);
    }
    // We assume that the tangent cone is unimodular, so there is no need
    // to decompose it.
    //
    // In the general case, the unimodular decomposition may have several
    // cones.
    GeneratingFunction vertexGf(numSymbols, {}, {}, {});
    SmallVector<std::pair<int, ConeH>, 4> unimodCones = {{1, tangentCone}};
    for (std::pair<int, ConeH> signedCone : unimodCones) {
      auto [sign, cone] = signedCone;
      vertexGf = vertexGf +
                 computeUnimodularConeGeneratingFunction(*vertex, sign, cone);
    }
    // We store the vertex we computed with the generating function of its
    // tangent cone.
    regionsAndGeneratingFunctions.emplace_back(PresburgerSet(activeRegionRel),
                                               vertexGf);
  } while (std::next_permutation(indicator.begin(), indicator.end()));

  // Now, we use Clauss-Loechner decomposition to identify regions in parameter
  // space where each vertex is active. These regions (chambers) have the
  // property that no two of them have a full-dimensional intersection, i.e.,
  // they may share "facets" or "edges", but their intersection can only have
  // up to numVars - 1 dimensions.
  //
  // In each chamber, we sum up the generating functions of the active vertices
  // to find the generating function of the polytope.
  return computeChamberDecomposition(numSymbols, regionsAndGeneratingFunctions);
}
