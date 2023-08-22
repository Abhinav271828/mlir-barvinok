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
    Point lambda = reducedBasis.getRow(min_i);
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
    Point row;
    for (unsigned i = 0; i < generators.getNumRows(); i++)
    {
        row = generators.getRow(i);
        denominator[i] = MutableArrayRef(row.begin(), row.end());
    }

    Fraction element = Fraction(0, 1);
    int flag = 1;
    for (unsigned i = 0; i < vertex.size(); i++)
        if (vertex[i].den != 1)
        {
            flag = 0;
            break;
        }
    if (flag == 1)
    {
        GeneratingFunction gf(SmallVector<int, 1>(1, sign),
                          std::vector({vertex}),
                          std::vector({denominator}));
        return gf;
    }
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
        SmallVector<Fraction> firstIntegerPoint = generators.preMultiplyWithRow(coefficients);
        Point numerator;
        numerator = MutableArrayRef(firstIntegerPoint.begin(), firstIntegerPoint.end());

        GeneratingFunction gf(SmallVector<int, 1>(1, sign),
                  std::vector({numerator}),
                  std::vector({denominator}));
 
        return gf;
    }

}