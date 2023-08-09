#include "mlir/Analysis/Presburger/Barvinok.h"
#include "mlir/Analysis/Presburger/LinearTransform.h"
#include "./Utils.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace mlir;
using namespace presburger;

TEST(BarvinokTest, samplePoint) {
    ConeV cone = makeMatrix<MPInt>(2, 2, {{MPInt(2), MPInt(7)}, {MPInt(1), MPInt(0)}});
    std::pair<Point, SmallVector<MPInt>> p = getSamplePoint(cone);

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
            EXPECT_EQ(decomp[0](i, j), makeMatrix<MPInt>(3, 3, {{MPInt(3), MPInt(0), MPInt(4)},
                                                                {MPInt(4), MPInt(5), MPInt(0)},
                                                                {MPInt(0), MPInt(0), MPInt(3)}})(i, j));
    for (unsigned i = 0; i < 3u; i++)
        for (unsigned j = 0; j < 3u; j++)
            EXPECT_EQ(decomp[1](i, j), makeMatrix<MPInt>(3, 3, {{MPInt(4), MPInt(5), MPInt(0)},
                                                                {MPInt(0), MPInt(3), MPInt(5)},
                                                                {MPInt(0), MPInt(0), MPInt(3)}})(i, j));
}