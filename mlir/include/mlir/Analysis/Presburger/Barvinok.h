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
using ParamPoint = Matrix<Fraction>;
using Point = SmallVector<Fraction>;

// Describe the type of generating function
// used to enumerate the integer points in a polytope.
// Consists of a set of terms, each having
// * a sign, ±1
// * a numerator, of the form x^{n}
// * a denominator, of the form (1 - x^{d1})...(1 - x^{dn})
class GeneratingFunction
{
public:
    GeneratingFunction(SmallVector<int, 16> s, std::vector<ParamPoint> nums, std::vector<std::vector<Point>> dens)
        : signs(s), numerators(nums), denominators(dens) {};

    bool operator==(const GeneratingFunction &gf) const
    {
        if (signs != gf.signs || numerators != gf.numerators || denominators != gf.denominators)
            return false;
        return true;
    }

    GeneratingFunction operator+(const GeneratingFunction &gf)
    {
        signs.insert(signs.end(), gf.signs.begin(), gf.signs.end());
        numerators.insert(numerators.end(), gf.numerators.begin(), gf.numerators.end());
        denominators.insert(denominators.end(), gf.denominators.begin(), gf.denominators.end());
        return *this;
    }

    llvm::raw_ostream &print(llvm::raw_ostream &os) const {
        if (signs[0] == 1) os << "+";
        else os << "-";

        os << "*(x^[paramVec])/";
        //os << "*(x^[";
        //for (unsigned i = 0; i < numerators[0].size()-1; i++)
        //    os << numerators[0][i] << ",";
        //os << numerators[0][numerators[0].size()-1] << "])/";

        for (Point den : denominators[0])
        {
            os << "(x^[";
            for (unsigned i = 0; i < den.size()-1; i++)
                os << den[i] << ",";
            os << den[den.size()-1] << "])";
        }
        
        for (unsigned i = 1; i < signs.size(); i++)
        {
            if (signs[i] == 1) os << "+";
            else os << "-";

            os << "*(x^[paramVec])/";
            //os << "*(x^[";
            //for (unsigned j = 0; j < numerators[i].size()-1; j++)
            //    os << numerators[i][j] << ",";
            //os << numerators[i][numerators[i].size()-1] << "])/";

            for (Point den : denominators[i])
            {
                os << "(x^[";
                for (unsigned j = 0; j < den.size(); j++)
                    os << den[j] << ",";
                os << den[den.size()-1] << "])";
            }
        }
        return os;
    }

    SmallVector<int, 16> signs;
    std::vector<ParamPoint> numerators;
    std::vector<std::vector<Point>> denominators;
};

class QuasiPolynomial
{
public:
    QuasiPolynomial(int par = 0,
                    SmallVector<Fraction> coeffs = {},
                    std::vector<std::vector<SmallVector<Fraction>>> aff = {}) :
        params(par), coefficients(coeffs), affine(aff) {};
    
    QuasiPolynomial(Fraction cons) :
        params(0), coefficients({cons}), affine({{}}) {};

    QuasiPolynomial(QuasiPolynomial const&) = default; //:
        //params(qp.params), coefficients(qp.coefficients), affine(qp.affine), constant(qp.constant) {};

    int params;
    SmallVector<Fraction> coefficients;
    std::vector<std::vector<SmallVector<Fraction>>> affine;

    // All the operations assume that the number of parameters is equal.
    QuasiPolynomial operator+(const QuasiPolynomial &x)
    {
        coefficients.append(x.coefficients);
        affine.insert(affine.end(), x.affine.begin(), x.affine.end());
        //constant = constant + x.constant;
        return *this;
    }
    QuasiPolynomial operator-(const QuasiPolynomial &x)
    {
        QuasiPolynomial qp(x.params, x.coefficients, x.affine);
        for (unsigned i = 0; i < x.coefficients.size(); i++)
            qp.coefficients[i] = - qp.coefficients[i];
        return (*this + qp);
    };
    QuasiPolynomial operator*(const QuasiPolynomial &x)
    {
        QuasiPolynomial qp(params);
        std::vector<SmallVector<Fraction>> product;
        for (unsigned i = 0; i < coefficients.size(); i++)
        {
            for (unsigned j = 0; j < x.coefficients.size(); j++)
            {
                qp.coefficients.append({coefficients[i] * x.coefficients[j]});
                product.clear();
                product.insert(product.end(), affine[i].begin(), affine[i].end());
                product.insert(product.end(), x.affine[j].begin(), x.affine[j].end());
                qp.affine.push_back(product);
            }
            //qp.coefficients.append({coefficients[i] * x.constant});
            //qp.affine.push_back(affine[i]);
        }
        //for (unsigned j = 0; j < x.coefficients.size(); j++)
        //{
        //    qp.coefficients.append({constant * x.coefficients[j]});
        //    qp.affine.push_back(x.affine[j]);
        //}
        //qp.constant = constant * x.constant;

        return qp;
    };
    QuasiPolynomial operator/(Fraction x)
    {
        for (unsigned i = 0; i < coefficients.size(); i++)
            coefficients[i] = coefficients[i] / x;
        //constant = constant / x;
        return *this;
    };

};

inline ConeH defineHRep(int num_ineqs, int num_vars, int num_params = 0)
{
    // We don't distinguish between domain and range variables, so
    // we set the number of domain variables as 0 and the number of
    // range variables as the number of actual variables.
    // There are no symbols (non-parametric for now) and no local
    // (existentially quantified) variables.
    ConeH cone(PresburgerSpace::getRelationSpace(0, num_vars, num_params, 0));
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
GeneratingFunction unimodularConeGeneratingFunction(ParamPoint, int, ConeH);

// Compute the (parametric) vertex from a subset of inequalities, if any such exists.
std::optional<ParamPoint> findVertex(Matrix<MPInt>);

// Compute the generating function for a polytope,
// as the sum of generating functions of its tangent cones.
std::vector<std::pair<PresburgerRelation, GeneratingFunction>> polytopeGeneratingFunction(PolyhedronH);

// Find the coefficient of a given power of s
// in a rational function given by P(s)/Q(s).
QuasiPolynomial getCoefficientInRationalFunction(int, std::vector<QuasiPolynomial>, std::vector<Fraction>);

// Substitute the generating function with the unit vector
// to find the number of terms.
QuasiPolynomial substituteWithUnitVector(GeneratingFunction);

// Count the number of integer points in a polytope,
// by chaining together `polytopeGeneratingFunction`
// and `substituteWithUnitVector`.
std::vector<std::pair<PresburgerRelation, QuasiPolynomial>> countIntegerPoints(PolyhedronH);

} // namespace presburger
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_BARVINOK_H