//===- Barvinok.cpp - Barvinok's Algorithm -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/Barvinok.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/Matrix.h"
#include "mlir/Analysis/Presburger/LinearTransform.h"
#include "mlir/Analysis/Presburger/PWMAFunction.h"
#include "mlir/Analysis/Presburger/PresburgerRelation.h"
#include "mlir/Analysis/Presburger/Simplex.h"
#include "mlir/Analysis/Presburger/Utils.h"
#include <numeric>
#include <optional>
#include <bitset>

using namespace mlir;
using namespace presburger;

// Assuming that the input cone is pointed at the origin,
// converts it to its dual in V-representation.
// Essentially we just remove the all-zeroes constant column.
ConeV mlir::presburger::getDual(ConeH cone)
{
    ConeV dual(cone.getNumInequalities(), cone.getNumCols()-1, 0, 0);
    // Assuming that an inequality of the form
    // a1*x1 + ... + an*xn + b ≥ 0
    // is represented as a row [a1, ..., an, b]
    // and that b = 0.

    for (unsigned i = 0; i < cone.getNumInequalities(); i++)
    {
        for (unsigned j = 0; j < cone.getNumCols()-1; j++)
        {
            // Like line 99 in Matrix.cpp
            dual.at(i, j) = cone.atIneq(i, j);
        }
    }

    // Now dual is of the form [ [a1, ..., an] , ... ]
    // which is the V-representation of the dual.
    return dual;
}

// Converts a cone in V-representation to the H-representation
// of its dual, pointed at the origin (not at the original vertex).
// Essentially adds a column consisting only of zeroes to the end.
ConeH mlir::presburger::getDual(ConeV cone)
{
    ConeH dual = defineHRep(cone.getNumRows(), cone.getNumColumns());
    cone.insertColumn(cone.getNumColumns());

    for (unsigned i = 0; i < cone.getNumRows(); i++)
        dual.addInequality(cone.getRow(i));

    // Now dual is of the form [ [a1, ..., an, 0] , ... ]
    // which is the H-representation of the dual.
    return dual;
}

// Find the index of a cone in V-representation.
// If there are more rays than variables, return 0.
MPInt mlir::presburger::getIndex(ConeV cone)
{
    unsigned rows = cone.getNumRows();
    unsigned cols = cone.getNumColumns();
    if (rows > cols)
    {
        return MPInt(0);
    }

    return cone.determinant();
}

// Find the shortest point in the lattice spanned by the rows
// of the cone, and the coefficients needed to express it in
// that basis.
std::pair<Point, SmallVector<MPInt, 16>> mlir::presburger::getSamplePoint(ConeV cone, Fraction delta)
{
    unsigned r = cone.getNumRows();
    unsigned c = cone.getNumColumns();
    Matrix<Fraction> rayMatrix(r, c);
    for (unsigned i = 0; i < r; i++)
        for (unsigned j = 0; j < c; j++)
            rayMatrix(i, j) = Fraction(cone(i, j), 1);
    
    // We now have a basis formed by the rows of A^{-1},
    // which we reduce.
    Matrix<Fraction> reducedBasis = rayMatrix.inverse();
    reducedBasis.LLL(delta);

    // We now have to find the smallest vector in this
    // basis by ∞-norm.
    Fraction min_norm(1, 0);
    unsigned min_i;
    Fraction norm, absVal;
    for (unsigned i = 0; i < r; i++)
    {
        norm = Fraction(0, 1);
        for (unsigned j = 0; j < c; j++)
        {
            absVal = abs(reducedBasis(i, j));
            norm = norm > absVal ? norm : absVal;
        }

        // We now have the norm of the i'th row.
        if (min_norm > norm)
        {
            min_norm = norm;
            min_i = i;
        }
    }

    // Now we have the smallest vector in the lattice spanned
    // by A^{-1} and the coefficients to express it in this basis.
    Point lambda = SmallVector<Fraction>(reducedBasis.getRow(min_i));
    SmallVector<Fraction> zFrac = rayMatrix.preMultiplyWithRow(lambda);

    SmallVector<MPInt> z(r);
    for (unsigned i = 0; i < r; i++)
        z[i] = zFrac[i].getAsInteger();
    
    unsigned allNeg = 1;
    for (unsigned i = 0; i < r; i++)
        if (lambda[i] > Fraction(0, 1))
        {
            allNeg = 0;
            break;
        }

    if (allNeg == 1)
    {
        for (unsigned i = 0; i < r; i++)
        {
            lambda[i] = - lambda[i];
            z[i] = - z[i];
        }
    }

    return std::make_pair(lambda, z);
}

SmallVector<std::pair<int, ConeV>, 16> mlir::presburger::unimodularDecompositionSimplicial(int sign, ConeV cone)
{
    MPInt index = getIndex(cone);
    if (index == 1 || index == -1)
    {
        return SmallVector<std::pair<int, ConeV>, 1>(1, std::make_pair(sign, cone));
    }
    std::pair<Point, SmallVector<MPInt>> samplePoint = getSamplePoint(cone, Fraction(3, 4));
    Point lambda = samplePoint.first;
    SmallVector<MPInt> z = samplePoint.second;

    SmallVector<std::pair<int, ConeV>, 2> decomposed, finalDecomposed;
    ConeV rays(cone.getNumRows(), cone.getNumColumns());
    for (unsigned i = 0; i < lambda.size(); i++)
    {
        if (lambda[i] == 0)
            continue;
        
        rays = cone;
        rays.setRow(i, z);
        assert(abs(getIndex(rays)) < abs(getIndex(cone)));
        decomposed = unimodularDecompositionSimplicial(lambda[i] > 0 ? sign : - sign, rays);
        finalDecomposed.append(decomposed);
    }
    return finalDecomposed;
}

// Decomposes a (not necessarily either unimodular or simplicial) cone
// pointed at the origin. Before passing to this function, the constant
// term should be eliminated from the cone.
SmallVector<std::pair<int, ConeH>, 16> mlir::presburger::unimodularDecomposition(ConeH cone)
{
    ConeV dualCone = getDual(cone);
    SmallVector<std::pair<int, ConeV>, 16> dualDecomposed;
    SmallVector<std::pair<int, ConeH>, 16> decomposed;

    MPInt index = getIndex(dualCone);
    if (index == 0)
    {
        SmallVector<ConeV, 16> simplicialCones = triangulate(dualCone);
        for (ConeV simplicialCone : simplicialCones)
        {
            SmallVector<std::pair<int, ConeV>, 16> unimodularCones = unimodularDecompositionSimplicial(1, simplicialCone);
            dualDecomposed.append(unimodularCones);
        }
    }
    else
    {
        dualDecomposed = unimodularDecompositionSimplicial(1, dualCone);
    }
    
    for (std::pair<int, ConeV> dualComponent : dualDecomposed)
    {
        decomposed.append(1, std::make_pair(dualComponent.first, getDual(dualComponent.second)));
    }

    return decomposed;
    
}

// Convert a cone in V-representation to H-representation, by iterating
// over (d-1)-subsets of the rays, finding the nullspace of their span
// and including it if it is spanned by one vector.
// The order in which the normals are returned is that formed by iterating
// a bitset starting at 2^n-1 and ending with 0.
// All the normals returned are inner normals.
Matrix<MPInt> mlir::presburger::generatorsToNormals(ConeV cone)
{
    unsigned n = cone.getNumRows();
    unsigned d = cone.getNumColumns();

    Matrix<MPInt> subset = Matrix<MPInt>(d-1, d);
    Matrix<MPInt> normals = Matrix<MPInt>(0, d);
    SmallVector<MPInt, 4> result;
    int allNonneg, allNonpos;

    // We need to iterate over all subsets of n with
    // d-1 elements.
    for (std::bitset<16> indicator(((1ul << (d-1))-1ul) << (n-d+1));
        indicator.to_ulong() <= ((1ul << (d-1))-1ul) << (n-d+1);              // (d-1) 1's followed by n-d+1 0's
        indicator = std::bitset<16>(indicator.to_ulong() - 1))
    {
        if (indicator.count() != d-1)
            continue;

        unsigned j = 0;
        for (unsigned i = 0; i < n; i++)
            if (indicator.test(i))
                subset.setRow(j++, cone.getRow(i));
        
        // Now subset is a (d-1)-subset of generators (d-1xd matrix).
        // We need to check if its null space is 1-dimensional,
        // and find the normal v.
        // If this normal belongs to the cone, it is the inner normal;
        // otherwise it's the outer normal.
        // To check if v belongs to the cone, we compute Av, and
        // if it is nonnegative, it belongs to it.

        Matrix<MPInt> nullspace = subset.nullSpace();

        if (nullspace.getNumRows() != 1u)
            continue;
        ArrayRef<MPInt> normalToFace = nullspace.getRow(0);

        result = cone.postMultiplyWithColumn(normalToFace);

        allNonneg = 1; allNonpos = 1;
        for (MPInt i : result)
            if (i < 0)
                allNonneg = 0;
            else if (i > 0)
                allNonpos = 0;

        if (allNonneg == 1)
            normals.appendExtraRow(normalToFace);
        else if (allNonpos == 1)
        {
            SmallVector<MPInt> innerNormal(normalToFace.begin(), normalToFace.end());
            for (unsigned i = 0; i < normalToFace.size(); i++)
                innerNormal[i] = -normalToFace[i];
            normals.appendExtraRow(innerNormal);
        }
    }

    return normals;
}

// Decompose a non-simplicial cone into a list of simplicial cones.
// Uses Delaunay's method of projecting up the rays and projecting
// down the lower-facing facets.
SmallVector<ConeV, 16> mlir::presburger::triangulate(ConeV cone)
{
    if (cone.getNumRows() == cone.getNumColumns())
    {
        return SmallVector<ConeV, 2>(1, cone);
    }

    unsigned n = cone.getNumRows();
    unsigned d = cone.getNumColumns();
    ConeV higherDimCone(n, d+1);
    MPInt sum;
    for (unsigned i = 0; i < n; i++)
    {
        sum = MPInt(0);
        for (unsigned j = 0; j < d; j++)
        {
            higherDimCone(i, j) = cone(i, j);
            sum += cone(i, j) * cone(i, j);
        }
        higherDimCone(i, d) = sum;
    }

    d = d + 1;

    SmallVector<ConeV, 4> decomposition = SmallVector<ConeV, 4>(0, Matrix<MPInt>(d-1, d-1));
    Matrix<MPInt> normals = generatorsToNormals(higherDimCone);
    Matrix<MPInt> subset = Matrix<MPInt>(d-1, d);

    unsigned i = 0;
    for (std::bitset<16> indicator(((1ul << (d-1))-1ul) << (n-d+1));
        indicator.to_ulong() <= ((1ul << (d-1))-1ul) << (n-d+1);              // (d-1) 1's followed by n-d+1 0's
        indicator = std::bitset<16>(indicator.to_ulong() - 1))
    {
        if (indicator.count() != d-1)
            continue;

        if (normals(i++, d-1) >= 0)
        {
            unsigned j = 0;
            for (unsigned k = 0; k < n; k++)
            if (indicator.test(k))
                subset.setRow(j++, higherDimCone.getRow(k));

            Matrix<MPInt> simplicial(d-1, d-1);
            for (unsigned k = 0; k < d-1; k++)
                for (unsigned j = 0; j < d-1; j++)
                    simplicial(k, j) = subset(k, j);
 
            decomposition.push_back(simplicial);
        }
    }

    return decomposition;
}

// Compute the generating function for a unimodular cone.
GeneratingFunction mlir::presburger::unimodularConeGeneratingFunction(Point vertex, int sign, ConeH cone)
{
    Matrix<Fraction> transp(cone.getNumVars(), cone.getNumInequalities());
    for (unsigned i = 0; i < cone.getNumInequalities(); i++)
        for (unsigned j = 0; j < cone.getNumVars(); j++)
            transp(j, i) = Fraction(cone.atIneq(i, j), MPInt(1));

    Matrix<Fraction> generators = transp.inverse();

    std::vector<Point> denominator(generators.getNumRows());
    ArrayRef<Fraction> row;
    for (unsigned i = 0; i < generators.getNumRows(); i++)
    {
        row = generators.getRow(i);
        denominator[i] = Point(row);
    }

    Fraction element = Fraction(0, 1);
    Point numerator;
    int flag = 1;
    for (unsigned i = 0; i < vertex.size(); i++)
        if (vertex[i].den != 1)
        {
            flag = 0;
            break;
        }
    if (flag == 1)
        numerator = vertex;
    else
    {
        // `cone` is assumed to be unimodular. Thus its ray matrix
        // is the inverse of its transpose.
        // We need to find c such that v = c @ rays = c @ (cone^{-1})^T.
        // Thus c = v @ cone^T.
        SmallVector<Fraction> coefficients = transp.preMultiplyWithRow(vertex);
        for (unsigned i = 0; i < coefficients.size(); i++)
            coefficients[i] = Fraction(ceil(coefficients[i]), MPInt(1));

        // The numerator is ceil(c) @ rays.
        numerator = generators.preMultiplyWithRow(coefficients);
    }

    GeneratingFunction gf(SmallVector<int>(1, sign),
              std::vector({numerator}),
              std::vector({denominator}));
 
    return gf;
}

std::optional<ParamPoint> mlir::presburger::findVertex(Matrix<MPInt> equations)
{
    // `equalities` is a d x (d + p + 1) matrix.

    unsigned r = equations.getNumRows();
    unsigned c = equations.getNumColumns();

    Matrix<MPInt> coeffs(r, r);
    for (unsigned i = 0; i < r; i++)
        for (unsigned j = 0; j < r; j++)
            coeffs(i, j) = equations(i, j), 1;
    
    if (coeffs.determinant() == MPInt(0))
        return std::nullopt;

    Matrix<Fraction> equationsF(r, c);
    for (unsigned i = 0; i < r; i++)
        for (unsigned j = 0; j < c; j++)
            equationsF(i, j) = Fraction(equations(i, j), 1);
    
    Fraction a, b;
    for (unsigned i = 0; i < r; i++)
    {
        if (equationsF(i, i) == Fraction(0, 1))
            for (unsigned j = i+1; j < r; j++)
                if (equationsF(j, i) != 0)
                {
                    equationsF.addToRow(i, equationsF.getRow(j), Fraction(1, 1));
                    break;
                }
        b = equationsF(i, i);

        for (unsigned j = 0; j < r; j++)
        {
            if (equationsF(j, i) == 0 || j == i) continue;
            a = equationsF(j, i);
            equationsF.addToRow(j, equationsF.getRow(i), - a / b);
        }
    }

    for (unsigned i = 0; i < r; i++)
    {
        a = equationsF(i, i);
        for (unsigned j = 0; j < c; j++)
            equationsF(i, j) = equationsF(i, j) / a;
    }

    ParamPoint vertex(r, c-r); // d x p+1
    for (unsigned i = 0; i < r; i++)
        for (unsigned j = 0; j < c-r; j++)
            vertex(i, j) = -equationsF(i, r+j);
    
    return vertex;
}

GeneratingFunction mlir::presburger::polytopeGeneratingFunction(PolyhedronH poly)
{
    unsigned d = poly.getNumRangeVars(); 
    unsigned p = poly.getNumSymbolVars();
    unsigned n = poly.getNumInequalities();

    SmallVector<std::pair<int, ConeH>, 4> unimodCones;
    GeneratingFunction gf({}, {}, {});
    ConeH tgtCone = defineHRep(d, d);

    Matrix<MPInt> subset(d, d+p+1);
    std::vector<Matrix<MPInt>> subsets; // Stores the inequality subsets corresponding to each vertex.
    Matrix<Fraction> remaining(n-d, d+p+1);

    std::optional<ParamPoint> vertex;
    std::vector<ParamPoint> vertices;

    Matrix<Fraction> a2(n-d, d);
    Matrix<Fraction> b2c2(n-d, p+1);

    Matrix<Fraction> activeRegion(n-d, p+1);
    Matrix<MPInt> activeRegionNorm(n-d, p+1);
    MPInt lcmDenoms;
    IntegerRelation activeRegionRel(PresburgerSpace::getRelationSpace(0, p, 0, 0));
    // The active region will be defined as activeRegionCoeffs @ p + activeRegionConstant ≥ 0.
    // The active region is a polyhedron in parameter space.
    std::vector<PresburgerRelation> activeRegions;
    

    for (std::bitset<16> indicator(((1ul << d)-1ul) << (n-d));
        indicator.to_ulong() <= ((1ul << d)-1ul) << (n-d);              // d 1's followed by n-d 0's
        indicator = std::bitset<16>(indicator.to_ulong() - 1))
    {
        if (indicator.count() != d)
            continue;

        subset = Matrix<MPInt>(d, d+p+1);
        remaining = Matrix<Fraction>(n-d, d+p+1);
        unsigned j1 = 0, j2 = 0;
        for (unsigned i = 0; i < n; i++)
            if (indicator.test(i))
                subset.setRow(j1++, poly.getInequality(i));
                // [A1 | B1 | c1]
            else
            {
                for (unsigned k = 0; k < d; k++)
                    a2(j2, k) = Fraction(poly.atIneq(i, k), 1);
                for (unsigned k = d; k < d+p+1; k++)
                    b2c2(j2, k-d) = Fraction(poly.atIneq(i, k), 1);
                j2++;
                // [A2 | B2 | c2]
            }

        vertex = findVertex(subset);

        if (vertex == std::nullopt) continue;
        vertices.push_back(*vertex);
        subsets.push_back(subset);

        // Region is given by (A2 @ X + B2) p + (A2 @ y + c2) ≥ 0
        // This is equivt to A2 @ [X | y] + [B2 | c2]
        // We premultiply [X | y] with each row of A2 and add each row of [B2 | c2].
        for (unsigned i = 0; i < n-d; i++)
        {
            activeRegion.setRow(i, (*vertex).preMultiplyWithRow(a2.getRow(i)));
            activeRegion.addToRow(i, b2c2.getRow(i), Fraction(1, 1));
        }

        activeRegionNorm = Matrix<MPInt>(n-d, p+1);
        activeRegionRel = IntegerRelation(PresburgerSpace::getRelationSpace(0, p, 0, 0));
        lcmDenoms = 1;
        for (unsigned i = 0; i < n-d; i++)
        {
            for (unsigned j = 0; j < p+1; j++)
                lcmDenoms = lcm(lcmDenoms, activeRegion(i, j).den);
            for (unsigned j = 0; j < p+1; j++)
                activeRegionNorm(i, j) = (activeRegion(i, j) * Fraction(lcmDenoms, 1)).getAsInteger();

            activeRegionRel.addInequality(activeRegionNorm.getRow(i));
        }

        activeRegions.push_back(PresburgerRelation(activeRegionRel));
    }

    // Clauss-Loechner chamber decomposition
    std::vector<std::pair<PresburgerRelation, std::vector<unsigned>>> chambers = 
        {std::make_pair(activeRegions[0], std::vector({0u}))};
    std::vector<std::pair<PresburgerRelation, std::vector<unsigned>>> newChambers;
    for (unsigned j = 1; j < vertices.size(); j++)
    {
        newChambers.clear();
        PresburgerRelation r_j = activeRegions[j];
        ParamPoint v_j = vertices[j];
        for (unsigned i = 0; i < chambers.size(); i++)
        {
            auto [r_i, v_i] = chambers[i];

            PresburgerRelation intersection = r_i.intersect(r_j);
            bool isFullDim = false;
            for (auto disjunct : intersection.getAllDisjuncts())
                if (disjunct.isFullDim())
                {
                    isFullDim = true;
                    break;
                }
            if (!isFullDim) newChambers.push_back(chambers[i]);
            else
            {
                PresburgerRelation subtraction = r_i.subtract(r_j);
                newChambers.push_back(std::make_pair(subtraction, v_i));

                v_i.push_back(j);
                newChambers.push_back(std::make_pair(intersection, v_i));
            }

        }
        for (auto chamber : newChambers)
            r_j = r_j.subtract(chamber.first);

        newChambers.push_back(std::make_pair(r_j, std::vector({j})));

        chambers.clear();
        for (auto chamber : newChambers)
        {
            bool empty = true;
            for (auto disjunct : chamber.first.getAllDisjuncts())
                if (!disjunct.isEmpty())
                {
                    empty = false;
                    break;
                }
            if (!empty)
                chambers.push_back(chamber);
        }
    }

    SmallVector<MPInt> ineq(d+1);
    for (auto chamber : chambers)
    {
        for (unsigned i : chamber.second)
        {
            tgtCone = defineHRep(d, d);
            for (unsigned j = 0; j < d; j++)
            {
                for (unsigned k = 0; k < d; k++)
                    ineq[k] = subsets[i](j, k);
                ineq[d] = subsets[i](j, d+p);
                tgtCone.addInequality(ineq);
            }
            unimodCones = unimodularDecomposition(tgtCone);
            // TODO
            // Define generating function computation for parametric cone.
        }
    }
    return gf;
}

Point getNonOrthogonalVector(std::vector<Point> vectors)
{
    // We use a recursive procedure. Let the inputs be {x1, ..., xk}, all vectors of length n.

    // Suppose we have a vector vs which is not orthogonal to
    // any of {x1[:n-1], ..., xk[:n-1]}.
    // Then we need v s.t. <xi, vs++[v]> != 0
    // => <xi[:n-1], v> + xi[-1]*v != 0
    // => v != - <xi[:n-1], v> / xi[-1]
    // We compute this value for all i, and then
    // set v to be the max of this set + 1. Thus
    // v is outside the set as desired, and we append it to vs.

    // The base case is given in one dimension,
    // where the vector [1] is not orthogonal to any
    // of the input vectors (since they are all nonzero).
    unsigned dim = vectors[0].size();
    SmallVector<Fraction> newPoint = {Fraction(1, 1)};
    std::vector<Fraction> lowerDimDotProducts;
    Fraction dotP = Fraction(0, 1);
    Fraction maxDisallowedValue = Fraction(-1, 0), disallowedValue = Fraction(0, 1);
    Fraction newValue;
    for (unsigned d = 2; d <= dim; d++)
    {
        lowerDimDotProducts.clear();
        for (Point vector : vectors)
        {
            dotP = Fraction(0, 1);
            for (unsigned i = 0; i < d-1; i++)
                dotP = dotP + vector[i] * newPoint[i];
            lowerDimDotProducts.push_back(dotP);
        }
        for (unsigned i = 0; i < vectors.size(); i++)
        {
            if (vectors[i][d-1] == 0) continue;
            disallowedValue = - lowerDimDotProducts[i] / vectors[i][d-1];
            if (maxDisallowedValue < disallowedValue)
                maxDisallowedValue = disallowedValue;
        }
        newValue = Fraction(ceil(maxDisallowedValue + Fraction(1, 1)), 1);
        newPoint.append(1, newValue);
    }
    return newPoint;
}

Fraction mlir::presburger::getCoefficientInRationalFunction(int power, std::vector<Fraction> num, std::vector<Fraction> den)
{
    // Let P[i] denote the coefficient of s^i in the L. polynomial P(s).
    // (P/Q)[r] =
    // if (r == 0) then P[0]/Q[0]
    // else
    //   (P[r] - {Σ_{i=1}^r (P/Q)[r-i])}/(Q[0])

    if (power == 0)
        return (num[0] / den[0]);

    Fraction t;
    if (power < num.size()) t = num[power];
    else t = Fraction(0, 1);
    for (int i = 1; (unsigned)i < (power+1 < den.size() ? power+1 : den.size()); i++)
        t = t - den[i] * getCoefficientInRationalFunction(power-i, num, den);
    return (t / den[0]);
}

// Substitute the generating function with the unit vector
// to find the number of terms.
Fraction mlir::presburger::substituteWithUnitVector(GeneratingFunction gf)
{
    std::vector<Point> allDenominators;
    for (std::vector<Point> den : gf.denominators)
        allDenominators.insert(allDenominators.end(), den.begin(), den.end());
    Point mu = getNonOrthogonalVector(allDenominators);

    Fraction term;
    int sign; Point v; std::vector<Point> ds;
    Fraction num; std::vector<Fraction> dens;
    int numNegExps; Fraction sumNegExps;
    std::vector<Fraction> numeratorCoefficients, singleTermDenCoefficients, denominatorCoefficients;
    std::vector<std::vector<Fraction>> eachTermDenCoefficients;
    std::vector<Fraction> convolution;
    unsigned convlen = 0; Fraction sum;
    unsigned r;

    Fraction totalTerm = Fraction(0, 1);
    for (unsigned i = 0; i < gf.signs.size(); i++)
    {
        sign = gf.signs[i]; v = gf.numerators[i]; ds = gf.denominators[i];

        // Substitute x_i = (s+1)^μ_i
        num = Fraction(0, 1);
        for (unsigned j = 0; j < v.size(); j++)
            num = num + v[j] * mu[j];
        // Now the numerator is (s+1)^num

        dens.clear();
        for (Point d : ds)
        {
            dens.push_back(Fraction(0, 1));
            for (unsigned k = 0; k < d.size(); k++)
                dens.back() = dens.back() + d[k] * mu[k];
            // This term in the denominator is
            // (1 - (s+1)^dens.back())
        }

        numNegExps = 0;
        sumNegExps = Fraction(0, 1);
        for (unsigned j = 0; j < dens.size(); j++)
        {
            if (dens[j] < Fraction(0, 1))
            {
                numNegExps += 1;
                sumNegExps = sumNegExps + dens[j];
            }
            // All exponents will be made positive (see line 480); then
            // reduce (1 - (s+1)^x) to (-s)*(Σ_{x-1} (s+1)^j) because x > 0
            dens[j] = abs(dens[j])-1;
        }

        // If we have (1 - (s+1)^-c) in the denominator,
        // multiply and divide by (s+1)^c to get
        // -(1 - (s+1)^c) in the denominator and
        // increase the numerator by c.
        if (numNegExps % 2 == 1) sign = - sign;
        num = num - sumNegExps;

        // Take all the (-s) out, from line 495
        r = dens.size();
        if (r % 2 == 1) sign = - sign;

        // Now the expression is
        // (s+1)^num /
        // s^r * Π_r (Σ_{d_i} (s+1)^j)
        
        // Letting P(s) = (s+1)^num and Q(s) = Π_r (...),
        // we need to find the coefficient of s^r in
        // P(s)/Q(s).

        // First, the coefficients of P(s), which are binomial coefficients.
        numeratorCoefficients.clear();
        numeratorCoefficients.push_back(1);
        for (int j = 1; j <= num; j++)
            numeratorCoefficients.push_back(numeratorCoefficients[j-1] * (num + 1 - j) / j);
        
        // Then the coefficients of each individual term in Q(s),
        // which are (di+1) C (k+1) for 0 ≤ k ≤ di
        eachTermDenCoefficients.clear();
        for (Fraction den : dens)
        {
            singleTermDenCoefficients.clear();
            singleTermDenCoefficients.push_back(den+Fraction(1, 1));
            for (unsigned j = 1; j <= den; j++)
                singleTermDenCoefficients.push_back(singleTermDenCoefficients[j-1] * (den + 1 - j) / (j + 1));

            eachTermDenCoefficients.push_back(singleTermDenCoefficients);
        }

        // Now we find the coefficients in Q(s) itself
        // by taking the convolution of the coefficients
        // of all the terms.
        denominatorCoefficients.clear();
        denominatorCoefficients = eachTermDenCoefficients[0];
        for (unsigned j = 1; j < eachTermDenCoefficients.size(); j++)
        {
            convlen = denominatorCoefficients.size() > eachTermDenCoefficients[j].size() ?
                      denominatorCoefficients.size() : eachTermDenCoefficients[j].size();
            for (unsigned k = denominatorCoefficients.size(); k < convlen; k++)
                denominatorCoefficients.push_back(Fraction(0, 1));
            for (unsigned k = eachTermDenCoefficients[j].size(); k < convlen; k++)
                eachTermDenCoefficients[j].push_back(Fraction(0, 1));

            convolution.clear();
            for (unsigned k = 0; k < convlen; k++)
            {
                sum = Fraction(0, 1);
                for (unsigned l = 0; l <= k; l++)
                    sum = sum + denominatorCoefficients[l] * eachTermDenCoefficients[j][k-l];
                convolution.push_back(sum);
            }
            denominatorCoefficients = convolution;
        }

        term = getCoefficientInRationalFunction(r, numeratorCoefficients, denominatorCoefficients);
        totalTerm = totalTerm + Fraction(sign, 1) * term;
    }

    return totalTerm;

}

MPInt mlir::presburger::countIntegerPoints(PolyhedronH poly)
{
    GeneratingFunction gf = polytopeGeneratingFunction(poly);
    Fraction f = substituteWithUnitVector(gf);
    return f.getAsInteger();
}