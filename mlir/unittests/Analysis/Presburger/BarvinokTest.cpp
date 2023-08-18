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
    ConeH cone = defineHRep(2, 2);
    Matrix<MPInt> ineqs = makeMatrix<MPInt>(2, 3, {{MPInt(0),  MPInt(1), MPInt(0)},
                                                   {MPInt(3), -MPInt(1), MPInt(0)}});
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
    ineqs = makeMatrix<MPInt>(3, 4, {{MPInt(-20),  MPInt(16), MPInt(15), MPInt(0)},
                                     {MPInt(20), MPInt(9), MPInt(-15), MPInt(0)},
                                     {MPInt(15), MPInt(-12), MPInt(20), MPInt(0)}});
    for (unsigned i = 0; i < 3; i++)
        cone.addInequality(ineqs.getRow(i));

    decomp = unimodularDecomposition(cone);
    EXPECT_EQ(decomp.size(), 18u);

    EXPECT_EQ(decomp[0].first, 1);
    for (unsigned i = 0; i < 3; i++)
        for (unsigned j = 0; j < 3; j++)
            EXPECT_EQ(decomp[0].second.atIneq(i, j),
                      makeMatrix<MPInt>(3, 3, {{MPInt(5), -MPInt(4), MPInt(7)}, {MPInt(4), -MPInt(3), MPInt(5)}, {MPInt(15), -MPInt(12), MPInt(20)}})(i, j));
    EXPECT_EQ(decomp[1].first, 1);
    for (unsigned i = 0; i < 3; i++)
        for (unsigned j = 0; j < 3; j++)
            EXPECT_EQ(decomp[1].second.atIneq(i, j),
                      makeMatrix<MPInt>(3, 3, {{MPInt(0), MPInt(0), MPInt(1)}, {MPInt(4), -MPInt(3), MPInt(5)}, {MPInt(5), -MPInt(4), MPInt(7)}})(i, j));
    EXPECT_EQ(decomp[2].first, 1);
    for (unsigned i = 0; i < 3; i++)
        for (unsigned j = 0; j < 3; j++)
            EXPECT_EQ(decomp[2].second.atIneq(i, j),
                      makeMatrix<MPInt>(3, 3, {{MPInt(1), -MPInt(1), MPInt(2)}, {MPInt(1), -MPInt(0), MPInt(0)}, {MPInt(4), -MPInt(3), MPInt(5)}})(i, j));
    EXPECT_EQ(decomp[3].first, -1);
    for (unsigned i = 0; i < 3; i++)
        for (unsigned j = 0; j < 3; j++)
            EXPECT_EQ(decomp[3].second.atIneq(i, j),
                      makeMatrix<MPInt>(3, 3, {{MPInt(0), MPInt(0), MPInt(1)}, {MPInt(1), -MPInt(1), MPInt(2)}, {MPInt(4), -MPInt(3), MPInt(5)}})(i, j));
    EXPECT_EQ(decomp[4].first, 1);
    for (unsigned i = 0; i < 3; i++)
        for (unsigned j = 0; j < 3; j++)
            EXPECT_EQ(decomp[4].second.atIneq(i, j),
                      makeMatrix<MPInt>(3, 3, {{MPInt(0), MPInt(0), MPInt(1)}, {MPInt(1), MPInt(0), MPInt(0)}, {MPInt(1), -MPInt(1), MPInt(2)}})(i, j));
    EXPECT_EQ(decomp[5].first, 1);
    for (unsigned i = 0; i < 3; i++)
        for (unsigned j = 0; j < 3; j++)
            EXPECT_EQ(decomp[5].second.atIneq(i, j),
                      makeMatrix<MPInt>(3, 3, {{-MPInt(1), MPInt(1), MPInt(1)}, {MPInt(1), MPInt(0), MPInt(0)}, {MPInt(0), MPInt(0), MPInt(1)}})(i, j));
    EXPECT_EQ(decomp[6].first, 1);
    for (unsigned i = 0; i < 3; i++)
        for (unsigned j = 0; j < 3; j++)
            EXPECT_EQ(decomp[6].second.atIneq(i, j),
                      makeMatrix<MPInt>(3, 3, {{-MPInt(5), MPInt(4), MPInt(4)}, {-MPInt(1), MPInt(1), MPInt(1)}, {MPInt(0), MPInt(0), MPInt(1)}})(i, j));
    EXPECT_EQ(decomp[7].first, 1);
    for (unsigned i = 0; i < 3; i++)
        for (unsigned j = 0; j < 3; j++)
            EXPECT_EQ(decomp[7].second.atIneq(i, j),
                      makeMatrix<MPInt>(3, 3, {{-MPInt(20), MPInt(16), MPInt(15)}, {-MPInt(1), MPInt(1), MPInt(1)}, {-MPInt(5), MPInt(4), MPInt(4)}})(i, j));
    EXPECT_EQ(decomp[8].first, 1);
    for (unsigned i = 0; i < 3; i++)
        for (unsigned j = 0; j < 3; j++)
            EXPECT_EQ(decomp[8].second.atIneq(i, j),
                      makeMatrix<MPInt>(3, 3, {{-MPInt(20), MPInt(16), MPInt(15)}, {MPInt(1), MPInt(0), MPInt(0)}, {-MPInt(1), MPInt(1), MPInt(1)}})(i, j));
    EXPECT_EQ(decomp[9].first, 1);
    for (unsigned i = 0; i < 3; i++)
        for (unsigned j = 0; j < 3; j++)
            EXPECT_EQ(decomp[9].second.atIneq(i, j),
                      makeMatrix<MPInt>(3, 3, {{MPInt(4), MPInt(2), -MPInt(3)}, {MPInt(7), MPInt(3), -MPInt(5)}, {MPInt(1), MPInt(0), MPInt(0)}})(i, j));
    EXPECT_EQ(decomp[10].first, 1);
    for (unsigned i = 0; i < 3; i++)
        for (unsigned j = 0; j < 3; j++)
            EXPECT_EQ(decomp[10].second.atIneq(i, j),
                      makeMatrix<MPInt>(3, 3, {{MPInt(4), MPInt(2), -MPInt(3)}, {MPInt(20), MPInt(9), -MPInt(15)}, {MPInt(7), MPInt(3), -MPInt(5)}})(i, j));
    EXPECT_EQ(decomp[11].first, -1);
    for (unsigned i = 0; i < 3; i++)
        for (unsigned j = 0; j < 3; j++)
            EXPECT_EQ(decomp[11].second.atIneq(i, j),
                      makeMatrix<MPInt>(3, 3, {{-MPInt(1), -MPInt(1), MPInt(1)}, {MPInt(4), MPInt(2), -MPInt(3)}, {MPInt(1), MPInt(0), MPInt(0)}})(i, j));
    EXPECT_EQ(decomp[12].first, -1);
    for (unsigned i = 0; i < 3; i++)
        for (unsigned j = 0; j < 3; j++)
            EXPECT_EQ(decomp[12].second.atIneq(i, j),
                      makeMatrix<MPInt>(3, 3, {{MPInt(0), MPInt(1), MPInt(0)}, {-MPInt(1), -MPInt(1), MPInt(1)}, {MPInt(1), MPInt(0), MPInt(0)}})(i, j));
    EXPECT_EQ(decomp[13].first, 1);
    for (unsigned i = 0; i < 3; i++)
        for (unsigned j = 0; j < 3; j++)
            EXPECT_EQ(decomp[13].second.atIneq(i, j),
                      makeMatrix<MPInt>(3, 3, {{MPInt(0), MPInt(1), MPInt(0)}, {MPInt(4), MPInt(2), -MPInt(3)}, {-MPInt(1), -MPInt(1), MPInt(1)}})(i, j));
    EXPECT_EQ(decomp[14].first, 1);
    for (unsigned i = 0; i < 3; i++)
        for (unsigned j = 0; j < 3; j++)
            EXPECT_EQ(decomp[14].second.atIneq(i, j),
                      makeMatrix<MPInt>(3, 3, {{-MPInt(1), MPInt(1), MPInt(1)}, {MPInt(0), MPInt(1), MPInt(0)}, {MPInt(1), MPInt(0), MPInt(0)}})(i, j));
    EXPECT_EQ(decomp[15].first, -1);
    for (unsigned i = 0; i < 3; i++)
        for (unsigned j = 0; j < 3; j++)
            EXPECT_EQ(decomp[15].second.atIneq(i, j),
                      makeMatrix<MPInt>(3, 3, {{MPInt(4), -MPInt(3), -MPInt(3)}, {MPInt(0), MPInt(1), MPInt(0)}, {-MPInt(1), MPInt(1), MPInt(1)}})(i, j));
    EXPECT_EQ(decomp[16].first, -1);
    for (unsigned i = 0; i < 3; i++)
        for (unsigned j = 0; j < 3; j++)
            EXPECT_EQ(decomp[16].second.atIneq(i, j),
                      makeMatrix<MPInt>(3, 3, {{MPInt(-20), MPInt(16), MPInt(15)}, {-MPInt(1), MPInt(1), MPInt(1)}, {MPInt(1), MPInt(0), MPInt(0)}})(i, j));
    EXPECT_EQ(decomp[17].first, 1);
    for (unsigned i = 0; i < 3; i++)
        for (unsigned j = 0; j < 3; j++)
            EXPECT_EQ(decomp[17].second.atIneq(i, j),
                      makeMatrix<MPInt>(3, 3, {{MPInt(-20), MPInt(16), MPInt(15)}, {MPInt(4), -MPInt(3), -MPInt(3)}, {-MPInt(1), MPInt(1), MPInt(1)}})(i, j));
}
