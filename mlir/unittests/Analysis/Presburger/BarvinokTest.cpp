#include "mlir/Analysis/Presburger/Barvinok.h"
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