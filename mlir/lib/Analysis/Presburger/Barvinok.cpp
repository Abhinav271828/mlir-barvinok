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
    // We don't distinguish between domain and range variables, so
    // we set the number of domain variables as 0 and the number of
    // range variables as the number of actual variables.
    // There are no symbols (non-parametric for now) and no local
    // (existentially quantified) variables.
    PresburgerSpace space = PresburgerSpace::getRelationSpace(0, cone.getNumColumns(), 0, 0);

    // We leave an extra column for the constant term. This is
    // checked in the assert in the source of IntegerRelation.
    ConeH dual(cone.getNumRows(), 0, cone.getNumColumns()+1, space);

    for (unsigned i = 0; i < cone.getNumRows(); i++)
    {
        for (unsigned j = 0; j < cone.getNumColumns(); j++)
        {
            dual.atIneq(i, j) = cone.at(i, j);
        }
        dual.atIneq(i, cone.getNumColumns()) = 0;
    }

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
std::pair<Point, SmallVector<MPInt, 16>> mlir::presburger::getSamplePoint(ConeV cone)
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
    reducedBasis.LLL(Fraction(3, 4));

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
    std::pair<Point, SmallVector<MPInt>> samplePoint = getSamplePoint(cone);
    Point lambda = samplePoint.first;
    SmallVector<MPInt> z = samplePoint.second;

    SmallVector<std::pair<int, ConeV>, 2> decomposed, finalDecomposed;
    ConeV rays = cone;
    for (unsigned i = 0; i < lambda.size(); i++)
    {
        if (lambda[i] == 0)
            continue;
        
        rays = cone;
        rays.setRow(i, z);
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