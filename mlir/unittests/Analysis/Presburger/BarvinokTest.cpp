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
    ConeV cone = makeMatrix<MPInt>(2, 2, {{MPInt(2), MPInt(7)}, {MPInt(1), MPInt(0)}});
    std::pair<Point, SmallVector<MPInt>> p = getSamplePoint(cone, Fraction(3, 4));

    std::vector s = {Fraction(1, 7), Fraction(-2, 7)};
    Point shortest = MutableArrayRef(s);
    EXPECT_EQ(p.first, shortest);

    SmallVector<MPInt, 16> coeffs = {MPInt(0), MPInt(1)};
    EXPECT_EQ(p.second, coeffs);

 }

TEST(BarvinokTest, unimodularDecompositionSimplicial) {
    ConeV cone = makeMatrix<MPInt>(2, 2, {{MPInt(1), MPInt(0)}, {MPInt(1), MPInt(10)}});
    SmallVector<std::pair<int, ConeV>, 1> r = unimodularDecompositionSimplicial(1, cone);

    EXPECT_EQ(r.size(), 2u);

    ConeV mat = makeMatrix<MPInt>(2, 2, {{MPInt(0), MPInt(1)}, {MPInt(1), MPInt(10)}});
    EXPECT_EQ(r[0].first, -1);
    for (unsigned i = 0; i < 2; i++)
        for (unsigned j = 0; j < 2; j++)
            EXPECT_EQ(r[0].second(i, j), mat(i, j));

    mat = makeMatrix<MPInt>(2, 2, {{MPInt(1), MPInt(0)}, {MPInt(0), MPInt(1)}});
    EXPECT_EQ(r[1].first, 1);
    for (unsigned i = 0; i < 2; i++)
        for (unsigned j = 0; j < 2; j++)
            EXPECT_EQ(r[1].second(i, j), mat(i, j));

    cone = makeMatrix<MPInt>(2, 2, {{MPInt(2), MPInt(0)}, {MPInt(0), MPInt(2)}});
    r = unimodularDecompositionSimplicial(1, cone);

    EXPECT_EQ(r.size(), 1u);

    mat = makeMatrix<MPInt>(2, 2, {{MPInt(1), MPInt(0)}, {MPInt(0), MPInt(1)}});
    EXPECT_EQ(r[0].first, 1);
    for (unsigned i = 0; i < 2; i++)
        for (unsigned j = 0; j < 2; j++)
            EXPECT_EQ(r[0].second(i, j), mat(i, j));

    cone = makeMatrix<MPInt>(3, 3, {{MPInt(4), MPInt(5), MPInt(0)},
                                    {MPInt(0), MPInt(3), MPInt(5)},
                                    {MPInt(0), MPInt(0), MPInt(3)}});
    r = unimodularDecompositionSimplicial(1, cone);
    EXPECT_EQ(r.size(), 8u);
}

TEST(BarvinokTest, genToIneq) {
    ConeV cone = makeMatrix<MPInt>(4, 4, {{ MPInt(4), MPInt(14), MPInt(11),  MPInt(3)},
                                    {MPInt(13),  MPInt(5), MPInt(14), MPInt(12)},
                                    {MPInt(13),  MPInt(9),  MPInt(7), MPInt(14)},
                                    { MPInt(2),  MPInt(3), MPInt(12),  MPInt(7)}});
    Matrix<MPInt> normals = cone.integerInverse().transpose();
    normals.normalizeByRows();

    Matrix<MPInt> coneH = generatorsToNormals(cone);

    for (unsigned i = 0; i < 4u; i++)
        for (unsigned j = 0; j < 4u; j++)
            EXPECT_EQ(coneH(i, j), normals(i, j));

    cone = makeMatrix<MPInt>(7, 7, {{ MPInt(4),  MPInt(9),  MPInt(3),  MPInt(4), MPInt(13),  MPInt(4), MPInt(11)},
                                    { MPInt(6),  MPInt(5),  MPInt(5), MPInt(13),  MPInt(8), MPInt(14),  MPInt(8)},
                                    { MPInt(3),  MPInt(6),  MPInt(6),  MPInt(7),  MPInt(7), MPInt(14),  MPInt(9)},
                                    { MPInt(9),  MPInt(7),  MPInt(7),  MPInt(3),  MPInt(9),  MPInt(6),  MPInt(5)},
                                    {MPInt(14), MPInt(13),  MPInt(4), MPInt(10), MPInt(10),  MPInt(7), MPInt(11)},
                                    { MPInt(4),  MPInt(7),  MPInt(3),  MPInt(5),  MPInt(2), MPInt(11),  MPInt(4)},
                                    { MPInt(7),  MPInt(8),  MPInt(4), MPInt(11),  MPInt(5), MPInt(11), MPInt(12)}});
    normals = cone.integerInverse().transpose();
    normals.normalizeByRows();

    coneH = generatorsToNormals(cone);

    for (unsigned i = 0; i < 7u; i++)
        for (unsigned j = 0; j < 7u; j++)
            EXPECT_EQ(coneH(i, j), normals(i, j));

    cone = makeMatrix<MPInt>(10, 10, {{MPInt(14), MPInt(11),  MPInt(8),  MPInt(3),  MPInt(3),  MPInt(4), MPInt(13), MPInt(14), MPInt(12), MPInt(11)},
                                      {MPInt(14),  MPInt(3), MPInt(12),  MPInt(2), MPInt(12), MPInt(11), MPInt(10),  MPInt(9),  MPInt(9), MPInt(14)},
                                      { MPInt(5),  MPInt(5),  MPInt(6),  MPInt(5), MPInt(13),  MPInt(7), MPInt(14), MPInt(10), MPInt(10),  MPInt(3)},
                                      { MPInt(9), MPInt(13),  MPInt(9), MPInt(10),  MPInt(2),  MPInt(3),  MPInt(4), MPInt(12), MPInt(14), MPInt(11)},
                                      { MPInt(4),  MPInt(7), MPInt(13),  MPInt(9), MPInt(14), MPInt(12),  MPInt(2),  MPInt(7),  MPInt(8), MPInt(12)},
                                      { MPInt(5), MPInt(10),  MPInt(8), MPInt(11), MPInt(11), MPInt(11),  MPInt(9),  MPInt(2),  MPInt(2),  MPInt(9)},
                                      {MPInt(11), MPInt(11),  MPInt(7),  MPInt(9),  MPInt(7), MPInt(12),  MPInt(3),  MPInt(7),  MPInt(4), MPInt(11)},
                                      { MPInt(4),  MPInt(3), MPInt(10), MPInt(12),  MPInt(3),  MPInt(2),  MPInt(2),  MPInt(3),  MPInt(8),  MPInt(5)},
                                      { MPInt(5),  MPInt(4),  MPInt(6),  MPInt(8),  MPInt(3),  MPInt(3),  MPInt(5),  MPInt(6), MPInt(13),  MPInt(9)},
                                      { MPInt(6),  MPInt(2), MPInt(12),  MPInt(5),  MPInt(4), MPInt(12), MPInt(14),  MPInt(9),  MPInt(4),  MPInt(7)}});
    normals = cone.integerInverse().transpose();
    normals.normalizeByRows();

    coneH = generatorsToNormals(cone);

    for (unsigned i = 0; i < 7u; i++)
        for (unsigned j = 0; j < 7u; j++)
            EXPECT_EQ(coneH(i, j), normals(i, j));

}

TEST(BarvinokTest, triangulate) {
    ConeV cone = makeMatrix<MPInt>(4, 3, {{MPInt(3), MPInt(0), MPInt(4)},
                                          {MPInt(4), MPInt(5), MPInt(0)},
                                          {MPInt(0), MPInt(3), MPInt(5)},
                                          {MPInt(0), MPInt(0), MPInt(3)}});
    
    SmallVector<ConeV, 2> decomp = triangulate(cone);
    EXPECT_EQ(decomp.size(), 2u);

    for (unsigned i = 0; i < 3u; i++)
        for (unsigned j = 0; j < 3u; j++)
            EXPECT_EQ(decomp[1](i, j), makeMatrix<MPInt>(3, 3, {{MPInt(3), MPInt(0), MPInt(4)},
                                                                {MPInt(4), MPInt(5), MPInt(0)},
                                                                {MPInt(0), MPInt(0), MPInt(3)}})(i, j));
    for (unsigned i = 0; i < 3u; i++)
        for (unsigned j = 0; j < 3u; j++)
            EXPECT_EQ(decomp[0](i, j), makeMatrix<MPInt>(3, 3, {{MPInt(4), MPInt(5), MPInt(0)},
                                                                {MPInt(0), MPInt(3), MPInt(5)},
                                                                {MPInt(0), MPInt(0), MPInt(3)}})(i, j));
}

TEST(BarvinokTest, unimodDecomp) {
    ConeH cone = defineHRep(4, 3);
    Matrix<MPInt> ineqs = makeMatrix<MPInt>(4, 4, {{MPInt(3), MPInt(0), MPInt(4), MPInt(0)},
                                                   {MPInt(4), MPInt(5), MPInt(0), MPInt(0)},
                                                   {MPInt(0), MPInt(3), MPInt(5), MPInt(0)},
                                                   {MPInt(0), MPInt(0), MPInt(3), MPInt(0)}});
    for (unsigned i = 0; i < 4; i++)
        cone.addInequality(ineqs.getRow(i));
 
    SmallVector<std::pair<int, ConeH>, 2> decomp = unimodularDecomposition(cone);

    EXPECT_EQ(decomp.size(), 14u);
}