//===- MatrixTest.cpp - Tests for Matrix ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/Matrix.h"
#include "mlir/Analysis/Presburger/Fraction.h"
#include "./Utils.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace mlir;
using namespace presburger;

TEST(MatrixTest, ReadWrite) {
  Matrix<MPInt> mat(5, 5);
  for (unsigned row = 0; row < 5; ++row)
    for (unsigned col = 0; col < 5; ++col)
      mat(row, col) = 10 * row + col;
  for (unsigned row = 0; row < 5; ++row)
    for (unsigned col = 0; col < 5; ++col)
      EXPECT_EQ(mat(row, col), int(10 * row + col));
}

TEST(MatrixTest, SwapColumns) {
  Matrix<MPInt> mat(5, 5);
  for (unsigned row = 0; row < 5; ++row)
    for (unsigned col = 0; col < 5; ++col)
      mat(row, col) = col == 3 ? 1 : 0;
  mat.swapColumns(3, 1);
  for (unsigned row = 0; row < 5; ++row)
    for (unsigned col = 0; col < 5; ++col)
      EXPECT_EQ(mat(row, col), col == 1 ? 1 : 0);

  // swap around all the other columns, swap (1, 3) twice for no effect.
  mat.swapColumns(3, 1);
  mat.swapColumns(2, 4);
  mat.swapColumns(1, 3);
  mat.swapColumns(0, 4);
  mat.swapColumns(2, 2);

  for (unsigned row = 0; row < 5; ++row)
    for (unsigned col = 0; col < 5; ++col)
      EXPECT_EQ(mat(row, col), col == 1 ? 1 : 0);
}

TEST(MatrixTest, SwapRows) {
  Matrix<MPInt> mat(5, 5);
  for (unsigned row = 0; row < 5; ++row)
    for (unsigned col = 0; col < 5; ++col)
      mat(row, col) = row == 2 ? 1 : 0;
  mat.swapRows(2, 0);
  for (unsigned row = 0; row < 5; ++row)
    for (unsigned col = 0; col < 5; ++col)
      EXPECT_EQ(mat(row, col), row == 0 ? 1 : 0);

  // swap around all the other rows, swap (2, 0) twice for no effect.
  mat.swapRows(3, 4);
  mat.swapRows(1, 4);
  mat.swapRows(2, 0);
  mat.swapRows(1, 1);
  mat.swapRows(0, 2);

  for (unsigned row = 0; row < 5; ++row)
    for (unsigned col = 0; col < 5; ++col)
      EXPECT_EQ(mat(row, col), row == 0 ? 1 : 0);
}

TEST(MatrixTest, resizeVertically) {
  Matrix<MPInt> mat(5, 5);
  EXPECT_EQ(mat.getNumRows(), 5u);
  EXPECT_EQ(mat.getNumColumns(), 5u);
  for (unsigned row = 0; row < 5; ++row)
    for (unsigned col = 0; col < 5; ++col)
      mat(row, col) = 10 * row + col;

  mat.resizeVertically(3);
  ASSERT_TRUE(mat.hasConsistentState());
  EXPECT_EQ(mat.getNumRows(), 3u);
  EXPECT_EQ(mat.getNumColumns(), 5u);
  for (unsigned row = 0; row < 3; ++row)
    for (unsigned col = 0; col < 5; ++col)
      EXPECT_EQ(mat(row, col), int(10 * row + col));

  mat.resizeVertically(5);
  ASSERT_TRUE(mat.hasConsistentState());
  EXPECT_EQ(mat.getNumRows(), 5u);
  EXPECT_EQ(mat.getNumColumns(), 5u);
  for (unsigned row = 0; row < 5; ++row)
    for (unsigned col = 0; col < 5; ++col)
      EXPECT_EQ(mat(row, col), row >= 3 ? 0 : int(10 * row + col));
}

TEST(MatrixTest, insertColumns) {
  Matrix<MPInt> mat(5, 5, 5, 10);
  EXPECT_EQ(mat.getNumRows(), 5u);
  EXPECT_EQ(mat.getNumColumns(), 5u);
  for (unsigned row = 0; row < 5; ++row)
    for (unsigned col = 0; col < 5; ++col)
      mat(row, col) = 10 * row + col;

  mat.insertColumns(3, 100);
  ASSERT_TRUE(mat.hasConsistentState());
  EXPECT_EQ(mat.getNumRows(), 5u);
  EXPECT_EQ(mat.getNumColumns(), 105u);
  for (unsigned row = 0; row < 5; ++row) {
    for (unsigned col = 0; col < 105; ++col) {
      if (col < 3)
        EXPECT_EQ(mat(row, col), int(10 * row + col));
      else if (3 <= col && col <= 102)
        EXPECT_EQ(mat(row, col), 0);
      else
        EXPECT_EQ(mat(row, col), int(10 * row + col - 100));
    }
  }

  mat.removeColumns(3, 100);
  ASSERT_TRUE(mat.hasConsistentState());
  mat.insertColumns(0, 0);
  ASSERT_TRUE(mat.hasConsistentState());
  mat.insertColumn(5);
  ASSERT_TRUE(mat.hasConsistentState());

  EXPECT_EQ(mat.getNumRows(), 5u);
  EXPECT_EQ(mat.getNumColumns(), 6u);
  for (unsigned row = 0; row < 5; ++row)
    for (unsigned col = 0; col < 6; ++col)
      EXPECT_EQ(mat(row, col), col == 5 ? 0 : 10 * row + col);
}

TEST(MatrixTest, insertRows) {
  Matrix<MPInt> mat(5, 5, 5, 10);
  ASSERT_TRUE(mat.hasConsistentState());
  EXPECT_EQ(mat.getNumRows(), 5u);
  EXPECT_EQ(mat.getNumColumns(), 5u);
  for (unsigned row = 0; row < 5; ++row)
    for (unsigned col = 0; col < 5; ++col)
      mat(row, col) = 10 * row + col;

  mat.insertRows(3, 100);
  ASSERT_TRUE(mat.hasConsistentState());
  EXPECT_EQ(mat.getNumRows(), 105u);
  EXPECT_EQ(mat.getNumColumns(), 5u);
  for (unsigned row = 0; row < 105; ++row) {
    for (unsigned col = 0; col < 5; ++col) {
      if (row < 3)
        EXPECT_EQ(mat(row, col), int(10 * row + col));
      else if (3 <= row && row <= 102)
        EXPECT_EQ(mat(row, col), 0);
      else
        EXPECT_EQ(mat(row, col), int(10 * (row - 100) + col));
    }
  }

  mat.removeRows(3, 100);
  ASSERT_TRUE(mat.hasConsistentState());
  mat.insertRows(0, 0);
  ASSERT_TRUE(mat.hasConsistentState());
  mat.insertRow(5);
  ASSERT_TRUE(mat.hasConsistentState());

  EXPECT_EQ(mat.getNumRows(), 6u);
  EXPECT_EQ(mat.getNumColumns(), 5u);
  for (unsigned row = 0; row < 6; ++row)
    for (unsigned col = 0; col < 5; ++col)
      EXPECT_EQ(mat(row, col), row == 5 ? 0 : 10 * row + col);
}

TEST(MatrixTest, resize) {
  Matrix<MPInt> mat(5, 5);
  EXPECT_EQ(mat.getNumRows(), 5u);
  EXPECT_EQ(mat.getNumColumns(), 5u);
  for (unsigned row = 0; row < 5; ++row)
    for (unsigned col = 0; col < 5; ++col)
      mat(row, col) = 10 * row + col;

  mat.resize(3, 3);
  ASSERT_TRUE(mat.hasConsistentState());
  EXPECT_EQ(mat.getNumRows(), 3u);
  EXPECT_EQ(mat.getNumColumns(), 3u);
  for (unsigned row = 0; row < 3; ++row)
    for (unsigned col = 0; col < 3; ++col)
      EXPECT_EQ(mat(row, col), int(10 * row + col));

  mat.resize(7, 7);
  ASSERT_TRUE(mat.hasConsistentState());
  EXPECT_EQ(mat.getNumRows(), 7u);
  EXPECT_EQ(mat.getNumColumns(), 7u);
  for (unsigned row = 0; row < 7; ++row)
    for (unsigned col = 0; col < 7; ++col)
      EXPECT_EQ(mat(row, col), row >= 3 || col >= 3 ? 0 : int(10 * row + col));
}

static void checkHermiteNormalForm(const Matrix<MPInt> &mat,
                                   const Matrix<MPInt> &hermiteForm) {
  auto [h, u] = mat.computeHermiteNormalForm();

  for (unsigned row = 0; row < mat.getNumRows(); row++)
    for (unsigned col = 0; col < mat.getNumColumns(); col++)
      EXPECT_EQ(h(row, col), hermiteForm(row, col));
}

TEST(MatrixTest, computeHermiteNormalForm) {
  // TODO: Add a check to test the original statement of hermite normal form
  // instead of using a precomputed result.

  {
    // Hermite form of a unimodular matrix is the identity matrix.
    Matrix<MPInt> mat = makeIntMatrix(3, 3, {{2, 3, 6}, {3, 2, 3}, {17, 11, 16}});
    Matrix<MPInt> hermiteForm = makeIntMatrix(3, 3, {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}});
    checkHermiteNormalForm(mat, hermiteForm);
  }

  {
    // Hermite form of a unimodular is the identity matrix.
    Matrix<MPInt> mat = makeIntMatrix(
        4, 4,
        {{-6, -1, -19, -20}, {0, 1, 0, 0}, {-5, 0, -15, -16}, {6, 0, 18, 19}});
    Matrix<MPInt> hermiteForm = makeIntMatrix(
        4, 4, {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}});
    checkHermiteNormalForm(mat, hermiteForm);
  }

  {
    Matrix<MPInt> mat = makeIntMatrix(
        4, 4, {{3, 3, 1, 4}, {0, 1, 0, 0}, {0, 0, 19, 16}, {0, 0, 0, 3}});
    Matrix<MPInt> hermiteForm = makeIntMatrix(
        4, 4, {{1, 0, 0, 0}, {0, 1, 0, 0}, {1, 0, 3, 0}, {18, 0, 54, 57}});
    checkHermiteNormalForm(mat, hermiteForm);
  }

  {
    Matrix<MPInt> mat = makeIntMatrix(
        4, 4, {{3, 3, 1, 4}, {0, 1, 0, 0}, {0, 0, 19, 16}, {0, 0, 0, 3}});
    Matrix<MPInt> hermiteForm = makeIntMatrix(
        4, 4, {{1, 0, 0, 0}, {0, 1, 0, 0}, {1, 0, 3, 0}, {18, 0, 54, 57}});
    checkHermiteNormalForm(mat, hermiteForm);
  }

  {
    Matrix<MPInt> mat =
        makeIntMatrix(3, 5, {{0, 2, 0, 7, 1}, {-1, 0, 0, -3, 0}, {0, 4, 1, 0, 8}});
    Matrix<MPInt> hermiteForm =
        makeIntMatrix(3, 5, {{1, 0, 0, 0, 0}, {0, 1, 0, 0, 0}, {0, 0, 1, 0, 0}});
    checkHermiteNormalForm(mat, hermiteForm);
  }
}

TEST(MatrixTest, inverse) {
    Matrix<Fraction> mat = makeFracMatrix(2, 2, {{Fraction(2, 1), Fraction(1, 1)}, {Fraction(7, 1), Fraction(0, 1)}});
    Matrix<Fraction> inverse = makeFracMatrix(2, 2, {{Fraction(0, 1), Fraction(1, 7)}, {Fraction(1, 1), Fraction(-2, 7)}});

    Matrix<Fraction> inv = mat.inverse();

    for (unsigned row = 0; row < 2; row++)
      for (unsigned col = 0; col < 2; col++)
        EXPECT_EQ(inv(row, col), inverse(row, col));
}

TEST(MatrixTest, intInverse) {
    Matrix<MPInt> mat = makeIntMatrix(2, 2, {{2, 1}, {7, 0}});
    Matrix<MPInt> inverse = makeIntMatrix(2, 2, {{0, -1}, {-7, 2}});
    
    Matrix<MPInt> inv = mat.integerInverse();

    for (unsigned i = 0; i < 2u; i++)
      for (unsigned j = 0; j < 2u; j++)
        EXPECT_EQ(inv(i, j), inverse(i, j));

    mat = makeIntMatrix(4, 4, {{ 4, 14, 11,  3},
                               {13,  5, 14, 12},
                               {13,  9,  7, 14},
                               { 2,  3, 12,  7}});
    inverse = makeIntMatrix(4, 4, {{155, 1636, -579, -1713},
                                   {725, -743, 537, -111},
                                   {210, 735, -855, 360},
                                   {-715, -1409, 1401, 1482}});

    inv = mat.integerInverse();

    for (unsigned i = 0; i < 2u; i++)
      for (unsigned j = 0; j < 2u; j++)
        EXPECT_EQ(inv(i, j), inverse(i, j));

}

TEST(MatrixTest, gramSchmidt) {
    Matrix<Fraction> mat = makeFracMatrix(3, 5, {{Fraction(3, 1), Fraction(4, 1), Fraction(5, 1), Fraction(12, 1), Fraction(19, 1)},
                                                 {Fraction(4, 1), Fraction(5, 1), Fraction(6, 1), Fraction(13, 1), Fraction(20, 1)},
                                                 {Fraction(7, 1), Fraction(8, 1), Fraction(9, 1), Fraction(16, 1), Fraction(24, 1)}});

    Matrix<Fraction> gramSchmidt = makeFracMatrix(3, 5,
           {{Fraction(3, 1),     Fraction(4, 1),     Fraction(5, 1),    Fraction(12, 1),     Fraction(19, 1)},
            {Fraction(142, 185), Fraction(383, 555), Fraction(68, 111), Fraction(13, 185),   Fraction(-262, 555)},
            {Fraction(53, 463),  Fraction(27, 463),  Fraction(1, 463),  Fraction(-181, 463), Fraction(100, 463)}});

    Matrix<Fraction> gs = mat.gramSchmidt();

    for (unsigned row = 0; row < 3; row++)
      for (unsigned col = 0; col < 5; col++)
        EXPECT_EQ(gs(row, col), gramSchmidt(row, col));
}

TEST(MatrixTest, LLL) {
    Matrix<Fraction> mat = makeFracMatrix(3, 3, {{Fraction(1, 1), Fraction(1, 1), Fraction(1, 1)},
                                                 {Fraction(-1, 1), Fraction(0, 1), Fraction(2, 1)},
                                                 {Fraction(3, 1), Fraction(5, 1), Fraction(6, 1)}});
    mat.LLL(Fraction(3, 4));
    
    Matrix<Fraction> LLL = makeFracMatrix(3, 3, {{Fraction(0, 1), Fraction(1, 1), Fraction(0, 1)},
                                                 {Fraction(1, 1), Fraction(0, 1), Fraction(1, 1)},
                                                 {Fraction(-1, 1), Fraction(0, 1), Fraction(2, 1)}});

    for (unsigned row = 0; row < 3; row++)
      for (unsigned col = 0; col < 3; col++)
        EXPECT_EQ(mat(row, col), LLL(row, col));


    mat = makeFracMatrix(2, 2, {{Fraction(12, 1), Fraction(2, 1)}, {Fraction(13, 1), Fraction(4, 1)}});
    LLL = makeFracMatrix(2, 2, {{Fraction(1, 1),  Fraction(2, 1)}, {Fraction(9, 1),  Fraction(-4, 1)}});

    mat.LLL(Fraction(3, 4));

    for (unsigned row = 0; row < 2; row++)
      for (unsigned col = 0; col < 2; col++)
        EXPECT_EQ(mat(row, col), LLL(row, col));

    mat = makeFracMatrix(3, 3, {{Fraction(1, 1), Fraction(0, 1), Fraction(2, 1)},
                                {Fraction(0, 1), Fraction(1, 3), -Fraction(5, 3)},
                                {Fraction(0, 1), Fraction(0, 1), Fraction(1, 1)}});
    LLL = makeFracMatrix(3, 3, {{Fraction(0, 1), Fraction(1, 3), Fraction(1, 3)},
                                {Fraction(0, 1), Fraction(1, 3), -Fraction(2, 3)},
                                {Fraction(1, 1), Fraction(0, 1), Fraction(0, 1)}});

    mat.LLL(Fraction(3, 4));

    for (unsigned row = 0; row < 3; row++)
      for (unsigned col = 0; col < 3; col++)
        EXPECT_EQ(mat(row, col), LLL(row, col));
}

TEST(MatrixTest, nullSpace) {
    Matrix<MPInt> mat = makeIntMatrix(3, 5, {{-3, 6, -1, 1, -7},
                                             {1, -2, 2, 3, -1},
                                             {2, -4, 5, 8, -4}});
    
    unsigned r = mat.getNumRows();
    unsigned c = mat.getNumColumns();
    Matrix<MPInt> augmentedMatrix(r+c, c);
    for (unsigned i = 0; i < r; i++)
        augmentedMatrix.setRow(i, mat.getRow(i));
    for (unsigned i = 0; i < c; i++)
        augmentedMatrix(r+i, i) = MPInt(1);
    Matrix<MPInt> reducedCEF = augmentedMatrix.computeHermiteNormalForm().first;

    Matrix<MPInt> null = mat.nullSpace();

    EXPECT_EQ(null.getNumRows(), 3u);

    for (unsigned i = 0; i < 3u; i++)
        for (unsigned j = 0; j < 3u; j++)
            EXPECT_EQ(mat.dotProduct(mat.getRow(i), null.getRow(j)), MPInt(0));
}