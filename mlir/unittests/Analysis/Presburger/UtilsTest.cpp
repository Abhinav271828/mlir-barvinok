//===- Utils.cpp - Tests for Utils file ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/Utils.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace mlir;
using namespace presburger;

static DivisionRepr parseDivisionRepr(unsigned numVars, unsigned numDivs,
                                      ArrayRef<ArrayRef<MPInt>> dividends,
                                      ArrayRef<MPInt> divisors) {
  DivisionRepr repr(numVars, numDivs);
  for (unsigned i = 0, rows = dividends.size(); i < rows; ++i)
    repr.setDiv(i, dividends[i], divisors[i]);
  return repr;
}

static void checkEqual(DivisionRepr &a, DivisionRepr &b) {
  EXPECT_EQ(a.getNumVars(), b.getNumVars());
  EXPECT_EQ(a.getNumDivs(), b.getNumDivs());
  for (unsigned i = 0, rows = a.getNumDivs(); i < rows; ++i) {
    EXPECT_EQ(a.hasRepr(i), b.hasRepr(i));
    if (!a.hasRepr(i))
      continue;
    EXPECT_TRUE(a.getDenom(i) == b.getDenom(i));
    EXPECT_TRUE(a.getDividend(i).equals(b.getDividend(i)));
  }
}

TEST(UtilsTest, ParseAndCompareDivisionReprTest) {
  auto merge = [](unsigned i, unsigned j) -> bool { return true; };
  DivisionRepr a = parseDivisionRepr(1, 1, {{MPInt(1), MPInt(2)}}, {MPInt(2)}),
               b = parseDivisionRepr(1, 1, {{MPInt(1), MPInt(2)}}, {MPInt(2)}),
               c = parseDivisionRepr(2, 2,
                                     {{MPInt(0), MPInt(1), MPInt(2)},
                                      {MPInt(0), MPInt(1), MPInt(2)}},
                                     {MPInt(2), MPInt(2)});
  c.removeDuplicateDivs(merge);
  checkEqual(a, b);
  checkEqual(a, c);
}

TEST(UtilsTest, DivisionReprNormalizeTest) {
  auto merge = [](unsigned i, unsigned j) -> bool { return true; };
  DivisionRepr a = parseDivisionRepr(2, 1, {{MPInt(1), MPInt(2), MPInt(-1)}},
                                     {MPInt(2)}),
               b = parseDivisionRepr(2, 1, {{MPInt(16), MPInt(32), MPInt(-16)}},
                                     {MPInt(32)}),
               c = parseDivisionRepr(1, 1, {{MPInt(12), MPInt(-4)}},
                                     {MPInt(8)}),
               d = parseDivisionRepr(2, 2,
                                     {{MPInt(1), MPInt(2), MPInt(-1)},
                                      {MPInt(4), MPInt(8), MPInt(-4)}},
                                     {MPInt(2), MPInt(8)});
  b.removeDuplicateDivs(merge);
  c.removeDuplicateDivs(merge);
  d.removeDuplicateDivs(merge);
  checkEqual(a, b);
  checkEqual(c, d);
}

TEST(UtilsTest, convolution) {
  std::vector<Fraction> aVals({1, 2, 3, 4});
  std::vector<Fraction> bVals({7, 3, 1, 6});
  ArrayRef<Fraction> a(aVals);
  ArrayRef<Fraction> b(bVals);

  std::vector<Fraction> conv = multiplyPolynomials(a, b);

  EXPECT_EQ(conv, std::vector<Fraction>({7, 17, 28, 45, 27, 22, 24}));

  aVals = {3, 6, 0, 2, 5};
  bVals = {2, 0, 6};
  a = aVals;
  b = bVals;

  conv = multiplyPolynomials(a, b);
  EXPECT_EQ(conv, std::vector<Fraction>({6, 12, 18, 40, 10, 12, 30}));
}

// The following polynomials are randomly generated and the
// coefficients are computed by hand.
// Although the function allows the coefficients of the numerator
// to be arbitrary quasipolynomials, we stick to constants for simplicity,
// as the relevant arithmetic operations on quasipolynomials
// are tested separately.
TEST(UtilsTest, getCoefficientInRationalFunction) {
  std::vector<QuasiPolynomial> numerator = {
      QuasiPolynomial(0, 2), QuasiPolynomial(0, 3), QuasiPolynomial(0, 5)};
  std::vector<Fraction> denominator = {Fraction(1), Fraction(0), Fraction(4),
                                       Fraction(3)};
  QuasiPolynomial coeff =
      getCoefficientInRationalFunction(1, numerator, denominator);
  EXPECT_EQ(coeff.getConstantTerm(), 3);

  numerator = {QuasiPolynomial(0, -1), QuasiPolynomial(0, 4),
               QuasiPolynomial(0, -2), QuasiPolynomial(0, 5),
               QuasiPolynomial(0, 6)};
  denominator = {Fraction(8), Fraction(4), Fraction(0), Fraction(-2)};
  coeff = getCoefficientInRationalFunction(3, numerator, denominator);
  EXPECT_EQ(coeff.getConstantTerm(), Fraction(55, 64));
}
