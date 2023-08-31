#include "mlir/Analysis/Presburger/Barvinok.h"
#include "mlir/Analysis/Presburger/Matrix.h"
#include "mlir/Analysis/Presburger/LinearTransform.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/PresburgerRelation.h"
#include "./Utils.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace mlir;
using namespace presburger;

TEST(BarvinokTest, samplePoint) {
    ConeV cone = makeIntMatrix(2, 2, {{2, 7}, {1, 0}});
    std::pair<Point, SmallVector<MPInt>> p = getSamplePoint(cone, Fraction(3, 4));

    Point shortest = {Fraction(1, 7), Fraction(-2, 7)};
    //Point shortest = MutableArrayRef(s);
    EXPECT_EQ(p.first, shortest);

    SmallVector<MPInt, 16> coeffs = {MPInt(0), MPInt(1)};
    EXPECT_EQ(p.second, coeffs);

 }

TEST(BarvinokTest, unimodularDecompositionSimplicial) {
    ConeV cone = makeIntMatrix(2, 2, {{1, 0}, {1, 10}});
    SmallVector<std::pair<int, ConeV>, 1> r = unimodularDecompositionSimplicial(1, cone);

    EXPECT_EQ(r.size(), 2u);

    ConeV mat = makeIntMatrix(2, 2, {{0, 1}, {1, 10}});
    EXPECT_EQ(r[0].first, -1);
    for (unsigned i = 0; i < 2; i++)
        for (unsigned j = 0; j < 2; j++)
            EXPECT_EQ(r[0].second(i, j), mat(i, j));

    mat = makeIntMatrix(2, 2, {{1, 0}, {0, 1}});
    EXPECT_EQ(r[1].first, 1);
    for (unsigned i = 0; i < 2; i++)
        for (unsigned j = 0; j < 2; j++)
            EXPECT_EQ(r[1].second(i, j), mat(i, j));

    cone = makeIntMatrix(2, 2, {{2, 0}, {0, 2}});
    r = unimodularDecompositionSimplicial(1, cone);

    EXPECT_EQ(r.size(), 1u);

    mat = makeIntMatrix(2, 2, {{1, 0}, {0, 1}});
    EXPECT_EQ(r[0].first, 1);
    for (unsigned i = 0; i < 2; i++)
        for (unsigned j = 0; j < 2; j++)
            EXPECT_EQ(r[0].second(i, j), mat(i, j));

    cone = makeIntMatrix(3, 3, {{4, 5, 0},
                                {0, 3, 5},
                                {0, 0, 3}});
    r = unimodularDecompositionSimplicial(1, cone);
    EXPECT_EQ(r.size(), 8u);
}

TEST(BarvinokTest, genToIneq) {
    ConeV cone = makeIntMatrix(4, 4, {{ 4, 14, 11,  3},
                                      {13,  5, 14, 12},
                                      {13,  9,  7, 14},
                                      { 2,  3, 12,  7}});
    Matrix<MPInt> normals = cone.integerInverse().transpose();
    normals.normalizeByRows();

    Matrix<MPInt> coneH = generatorsToNormals(cone);

    for (unsigned i = 0; i < 4u; i++)
        for (unsigned j = 0; j < 4u; j++)
            EXPECT_EQ(coneH(i, j), normals(i, j));

    cone = makeIntMatrix(7, 7, {{ 4,  9,  3,  4, 13,  4, 11},
                                { 6,  5,  5, 13,  8, 14,  8},
                                { 3,  6,  6,  7,  7, 14,  9},
                                { 9,  7,  7,  3,  9,  6,  5},
                                {14, 13,  4, 10, 10,  7, 11},
                                { 4,  7,  3,  5,  2, 11,  4},
                                { 7,  8,  4, 11,  5, 11, 12}});
    normals = cone.integerInverse().transpose();
    normals.normalizeByRows();

    coneH = generatorsToNormals(cone);

    for (unsigned i = 0; i < 7u; i++)
        for (unsigned j = 0; j < 7u; j++)
            EXPECT_EQ(coneH(i, j), normals(i, j));

    cone = makeIntMatrix(10, 10, {{14, 11,  8,  3,  3,  4, 13, 14, 12, 11},
                                  {14,  3, 12,  2, 12, 11, 10,  9,  9, 14},
                                  { 5,  5,  6,  5, 13,  7, 14, 10, 10,  3},
                                  { 9, 13,  9, 10,  2,  3,  4, 12, 14, 11},
                                  { 4,  7, 13,  9, 14, 12,  2,  7,  8, 12},
                                  { 5, 10,  8, 11, 11, 11,  9,  2,  2,  9},
                                  {11, 11,  7,  9,  7, 12,  3,  7,  4, 11},
                                  { 4,  3, 10, 12,  3,  2,  2,  3,  8,  5},
                                  { 5,  4,  6,  8,  3,  3,  5,  6, 13,  9},
                                  { 6,  2, 12,  5,  4, 12, 14,  9,  4,  7}});
    normals = cone.integerInverse().transpose();
    normals.normalizeByRows();

    coneH = generatorsToNormals(cone);

    for (unsigned i = 0; i < 7u; i++)
        for (unsigned j = 0; j < 7u; j++)
            EXPECT_EQ(coneH(i, j), normals(i, j));

}

TEST(BarvinokTest, triangulate) {
    ConeV cone = makeIntMatrix(4, 3, {{3, 0, 4},
                                      {4, 5, 0},
                                      {0, 3, 5},
                                      {0, 0, 3}});
    
    SmallVector<ConeV, 2> decomp = triangulate(cone);
    EXPECT_EQ(decomp.size(), 2u);

    for (unsigned i = 0; i < 3u; i++)
        for (unsigned j = 0; j < 3u; j++)
            EXPECT_EQ(decomp[1](i, j), makeIntMatrix(3, 3, {{3, 0, 4},
                                                            {4, 5, 0},
                                                            {0, 0, 3}})(i, j));
    for (unsigned i = 0; i < 3u; i++)
        for (unsigned j = 0; j < 3u; j++)
            EXPECT_EQ(decomp[0](i, j), makeIntMatrix(3, 3, {{4, 5, 0},
                                                            {0, 3, 5},
                                                            {0, 0, 3}})(i, j));
}

TEST(BarvinokTest, unimodDecomp) {
    ConeH cone = defineHRep(2, 2);
    Matrix<MPInt> ineqs = makeIntMatrix(2, 3, {{0,  1, 0},
                                               {3, -1, 0}});
    for (unsigned i = 0; i < 2; i++)
        cone.addInequality(ineqs.getRow(i));

    // /_ = /  + |_ + |
    //     |          |
    SmallVector<std::pair<int, ConeH>, 2> decomp = unimodularDecomposition(cone);

    // This decomposition is modulo cones with lines. In fact
    // case, we need to subtract the cone defined by (x >= 0).
    ConeH coneWithLine = defineHRep(1, 2);
    coneWithLine.addInequality(ArrayRef({MPInt(1), MPInt(0), MPInt(0)}));
    decomp.append(1, std::make_pair(-1, coneWithLine));

    int flag = 1;
    for (int64_t x = -5; x < 5; x++)
        for (int64_t y = -5; y < 5; y++)
        {
            // Count 1 if the point belongs to a cone,
            // and 0 if it does not. Multiply this by
            // the sign of the cone.
            flag = 0;
            for (std::pair<int, ConeH> component : decomp)
                if (component.second.containsPoint(ArrayRef({MPInt(x), MPInt(y)})))
                    flag += component.first;

            // The result should indicate if the point
            // belongs to the original cone.
            EXPECT_EQ(cone.containsPoint(ArrayRef({MPInt(x), MPInt(y)})), (bool)flag);
        }

    cone = defineHRep(3, 3);
    // This is the cone whose V-rep is [[3, 0, 4], [4, 5, 0], [0, 5, 3]]
    ineqs = makeIntMatrix(3, 4, {{-20,  16, 15, 0},
                                 {20, 9, -15, 0},
                                 {15, -12, 20, 0}});
    for (unsigned i = 0; i < 3; i++)
        cone.addInequality(ineqs.getRow(i));

    decomp = unimodularDecomposition(cone);
    EXPECT_EQ(decomp.size(), 18u);

    EXPECT_EQ(decomp[0].first, 1);
    for (unsigned i = 0; i < 3; i++)
        for (unsigned j = 0; j < 3; j++)
            EXPECT_EQ(decomp[0].second.atIneq(i, j),
                      makeIntMatrix(3, 3, {{5, -4, 7}, {4, -3, 5}, {15, -12, 20}})(i, j));
    EXPECT_EQ(decomp[1].first, 1);
    for (unsigned i = 0; i < 3; i++)
        for (unsigned j = 0; j < 3; j++)
            EXPECT_EQ(decomp[1].second.atIneq(i, j),
                      makeIntMatrix(3, 3, {{0, 0, 1}, {4, -3, 5}, {5, -4, 7}})(i, j));
    EXPECT_EQ(decomp[2].first, 1);
    for (unsigned i = 0; i < 3; i++)
        for (unsigned j = 0; j < 3; j++)
            EXPECT_EQ(decomp[2].second.atIneq(i, j),
                      makeIntMatrix(3, 3, {{1, -1, 2}, {1, -0, 0}, {4, -3, 5}})(i, j));
    EXPECT_EQ(decomp[3].first, -1);
    for (unsigned i = 0; i < 3; i++)
        for (unsigned j = 0; j < 3; j++)
            EXPECT_EQ(decomp[3].second.atIneq(i, j),
                      makeIntMatrix(3, 3, {{0, 0, 1}, {1, -1, 2}, {4, -3, 5}})(i, j));
    EXPECT_EQ(decomp[4].first, 1);
    for (unsigned i = 0; i < 3; i++)
        for (unsigned j = 0; j < 3; j++)
            EXPECT_EQ(decomp[4].second.atIneq(i, j),
                      makeIntMatrix(3, 3, {{0, 0, 1}, {1, 0, 0}, {1, -1, 2}})(i, j));
    EXPECT_EQ(decomp[5].first, 1);
    for (unsigned i = 0; i < 3; i++)
        for (unsigned j = 0; j < 3; j++)
            EXPECT_EQ(decomp[5].second.atIneq(i, j),
                      makeIntMatrix(3, 3, {{-1, 1, 1}, {1, 0, 0}, {0, 0, 1}})(i, j));
    EXPECT_EQ(decomp[6].first, 1);
    for (unsigned i = 0; i < 3; i++)
        for (unsigned j = 0; j < 3; j++)
            EXPECT_EQ(decomp[6].second.atIneq(i, j),
                      makeIntMatrix(3, 3, {{-5, 4, 4}, {-1, 1, 1}, {0, 0, 1}})(i, j));
    EXPECT_EQ(decomp[7].first, 1);
    for (unsigned i = 0; i < 3; i++)
        for (unsigned j = 0; j < 3; j++)
            EXPECT_EQ(decomp[7].second.atIneq(i, j),
                      makeIntMatrix(3, 3, {{-20, 16, 15}, {-1, 1, 1}, {-5, 4, 4}})(i, j));
    EXPECT_EQ(decomp[8].first, 1);
    for (unsigned i = 0; i < 3; i++)
        for (unsigned j = 0; j < 3; j++)
            EXPECT_EQ(decomp[8].second.atIneq(i, j),
                      makeIntMatrix(3, 3, {{-20, 16, 15}, {1, 0, 0}, {-1, 1, 1}})(i, j));
    EXPECT_EQ(decomp[9].first, 1);
    for (unsigned i = 0; i < 3; i++)
        for (unsigned j = 0; j < 3; j++)
            EXPECT_EQ(decomp[9].second.atIneq(i, j),
                      makeIntMatrix(3, 3, {{4, 2, -3}, {7, 3, -5}, {1, 0, 0}})(i, j));
    EXPECT_EQ(decomp[10].first, 1);
    for (unsigned i = 0; i < 3; i++)
        for (unsigned j = 0; j < 3; j++)
            EXPECT_EQ(decomp[10].second.atIneq(i, j),
                      makeIntMatrix(3, 3, {{4, 2, -3}, {20, 9, -15}, {7, 3, -5}})(i, j));
    EXPECT_EQ(decomp[11].first, -1);
    for (unsigned i = 0; i < 3; i++)
        for (unsigned j = 0; j < 3; j++)
            EXPECT_EQ(decomp[11].second.atIneq(i, j),
                      makeIntMatrix(3, 3, {{-1, -1, 1}, {4, 2, -3}, {1, 0, 0}})(i, j));
    EXPECT_EQ(decomp[12].first, -1);
    for (unsigned i = 0; i < 3; i++)
        for (unsigned j = 0; j < 3; j++)
            EXPECT_EQ(decomp[12].second.atIneq(i, j),
                      makeIntMatrix(3, 3, {{0, 1, 0}, {-1, -1, 1}, {1, 0, 0}})(i, j));
    EXPECT_EQ(decomp[13].first, 1);
    for (unsigned i = 0; i < 3; i++)
        for (unsigned j = 0; j < 3; j++)
            EXPECT_EQ(decomp[13].second.atIneq(i, j),
                      makeIntMatrix(3, 3, {{0, 1, 0}, {4, 2, -3}, {-1, -1, 1}})(i, j));
    EXPECT_EQ(decomp[14].first, 1);
    for (unsigned i = 0; i < 3; i++)
        for (unsigned j = 0; j < 3; j++)
            EXPECT_EQ(decomp[14].second.atIneq(i, j),
                      makeIntMatrix(3, 3, {{-1, 1, 1}, {0, 1, 0}, {1, 0, 0}})(i, j));
    EXPECT_EQ(decomp[15].first, -1);
    for (unsigned i = 0; i < 3; i++)
        for (unsigned j = 0; j < 3; j++)
            EXPECT_EQ(decomp[15].second.atIneq(i, j),
                      makeIntMatrix(3, 3, {{4, -3, -3}, {0, 1, 0}, {-1, 1, 1}})(i, j));
    EXPECT_EQ(decomp[16].first, -1);
    for (unsigned i = 0; i < 3; i++)
        for (unsigned j = 0; j < 3; j++)
            EXPECT_EQ(decomp[16].second.atIneq(i, j),
                      makeIntMatrix(3, 3, {{-20, 16, 15}, {-1, 1, 1}, {1, 0, 0}})(i, j));
    EXPECT_EQ(decomp[17].first, 1);
    for (unsigned i = 0; i < 3; i++)
        for (unsigned j = 0; j < 3; j++)
            EXPECT_EQ(decomp[17].second.atIneq(i, j),
                      makeIntMatrix(3, 3, {{-20, 16, 15}, {4, -3, -3}, {-1, 1, 1}})(i, j));
}

TEST(BarvinokTest, unimodGenFunc) {
    ConeH cone = defineHRep(2, 2);
    Matrix<MPInt> ineqs = makeIntMatrix(2, 3, {{1, -10, 0}, {0, 1, 0}});
    for (unsigned i = 0; i < 2u; i++)
        cone.addInequality(ineqs.getRow(i));
    
    SmallVector<Fraction> vertex = {Fraction(3, 4), Fraction(5, 3)};
    GeneratingFunction gf = unimodularConeGeneratingFunction(vertex, 1, cone);

    SmallVector<Fraction, 2> nums = SmallVector<Fraction>({Fraction(5, 1), Fraction(2, 1)});
    std::vector dens = std::vector({SmallVector<Fraction>({Fraction(1, 1), Fraction(0, 1)}),
                                    SmallVector<Fraction>({Fraction(10, 1), Fraction(1, 1)})});

    EXPECT_EQ(gf.signs, SmallVector<int>(1, 1));

    EXPECT_EQ(gf.numerators.size(), 1u);
    EXPECT_EQ(gf.numerators[0], nums);

    EXPECT_EQ(gf.denominators.size(), 1u);
    for (unsigned i = 0; i < 2; i++)
        for (unsigned j = 0; j < 2; j++)
            EXPECT_EQ(gf.denominators[0][i][j], dens[i][j]);
}

TEST(BarvinokTest, getCoefficientInRationalFunction) {
    std::vector<Fraction> numeratorCoefficients, singleTermDenCoefficients, denominatorCoefficients, convolution;
    int convlen; Fraction sum;
    std::vector<std::vector<Fraction>> eachTermDenCoefficients;

    Fraction num = Fraction(20, 1);
    std::vector<Fraction> dens = {Fraction(8, 1), Fraction(6, 1), Fraction(1, 1)};

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

    Fraction coeff = getCoefficientInRationalFunction(3, numeratorCoefficients, denominatorCoefficients);
    EXPECT_EQ(coeff, Fraction(4531, 3024));
}

TEST(BarvinokTest, substituteWithUnitVector) {
    GeneratingFunction gf(SmallVector<int>({1, 1, 1}),
                          std::vector({SmallVector<Fraction>({Fraction(2, 1), Fraction(0, 1)}),
                                       SmallVector<Fraction>({Fraction(0, 1), Fraction(2, 1)}),
                                       SmallVector<Fraction>({Fraction(0, 1), Fraction(0, 1)})}),
                          std::vector({std::vector({SmallVector<Fraction>({-Fraction(1, 1), Fraction(0, 1)}),
                                                    SmallVector<Fraction>({-Fraction(1, 1), Fraction(1, 1)})}),
                                       std::vector({SmallVector<Fraction>({Fraction(0, 1), -Fraction(1, 1)}),
                                                    SmallVector<Fraction>({Fraction(1, 1), -Fraction(1, 1)})}),
                                       std::vector({SmallVector<Fraction>({Fraction(1, 1), Fraction(0, 1)}),
                                                    SmallVector<Fraction>({Fraction(0, 1), Fraction(1, 1)})})}));

    Fraction numPoints = substituteWithUnitVector(gf);
    EXPECT_EQ(numPoints, Fraction(6, 1));
}

TEST(BarvinokTest, polytopeGeneratingFunction) {
    Matrix<MPInt> ineqs = makeIntMatrix(6, 4, {{1, 0, 0, 0},
                                               {0, 1, 0, 0},
                                               {0, 0, 1, 0},
                                               {-1, 0, 0, 1},
                                               {0, -1, 0, 1},
                                               {0, 0, -1, 1}});
    PolyhedronH cube = defineHRep(6, 3);
    for (unsigned i = 0; i < 6; i++)
        cube.addInequality(ineqs.getRow(i));
    
    GeneratingFunction gf = polytopeGeneratingFunction(cube);

    EXPECT_EQ(gf.signs, SmallVector<int>({1, 1, 1, 1, 1, 1, 1, 1}));
    EXPECT_EQ(gf.numerators, std::vector<Point>({SmallVector<Fraction>({1, 1, 1}),
                                                 SmallVector<Fraction>({0, 1, 1}),
                                                 SmallVector<Fraction>({1, 0, 1}),
                                                 SmallVector<Fraction>({0, 0, 1}),
                                                 SmallVector<Fraction>({1, 1, 0}),
                                                 SmallVector<Fraction>({0, 1, 0}),
                                                 SmallVector<Fraction>({1, 0, 0}),
                                                 SmallVector<Fraction>({0, 0, 0})}));
    EXPECT_EQ(gf.denominators, std::vector<std::vector<Point>>({std::vector<Point>({SmallVector<Fraction>({-1, 0, 0}),
                                                                                    SmallVector<Fraction>({0, -1, 0}),
                                                                                    SmallVector<Fraction>({0, 0, -1})}),
                                                                std::vector<Point>({SmallVector<Fraction>({1, 0, 0}),
                                                                                    SmallVector<Fraction>({0, -1, 0}),
                                                                                    SmallVector<Fraction>({0, 0, -1})}),
                                                                std::vector<Point>({SmallVector<Fraction>({0, 1, 0}),
                                                                                    SmallVector<Fraction>({-1, 0, 0}),
                                                                                    SmallVector<Fraction>({0, 0, -1})}),
                                                                std::vector<Point>({SmallVector<Fraction>({1, 0, 0}),
                                                                                    SmallVector<Fraction>({0, 1, 0}),
                                                                                    SmallVector<Fraction>({0, 0, -1})}),
                                                                std::vector<Point>({SmallVector<Fraction>({0, 0, 1}),
                                                                                    SmallVector<Fraction>({-1, 0, 0}),
                                                                                    SmallVector<Fraction>({0, -1, 0})}),
                                                                std::vector<Point>({SmallVector<Fraction>({1, 0, 0}),
                                                                                    SmallVector<Fraction>({0, 0, 1}),
                                                                                    SmallVector<Fraction>({0, -1, 0})}),
                                                                std::vector<Point>({SmallVector<Fraction>({0, 1, 0}),
                                                                                    SmallVector<Fraction>({0, 0, 1}),
                                                                                    SmallVector<Fraction>({-1, 0, 0})}),
                                                                std::vector<Point>({SmallVector<Fraction>({1, 0, 0}),
                                                                                    SmallVector<Fraction>({0, 1, 0}),
                                                                                    SmallVector<Fraction>({0, 0, 1})})}));
}