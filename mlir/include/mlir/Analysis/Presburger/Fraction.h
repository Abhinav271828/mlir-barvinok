//===- Fraction.h - MLIR Fraction Class -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a simple class to represent fractions. It supports multiplication,
// comparison, floor, and ceiling operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_FRACTION_H
#define MLIR_ANALYSIS_PRESBURGER_FRACTION_H

#include "mlir/Analysis/Presburger/MPInt.h"
#include "mlir/Analysis/Presburger/Utils.h"
#include "mlir/Support/MathExtras.h"

namespace mlir {
namespace presburger {

/// A class to represent fractions. The sign of the fraction is represented
/// in the sign of the numerator; the denominator is always positive.
///
/// Note that overflows may occur if the numerator or denominator are not
/// representable by 64-bit integers.
struct Fraction {
  /// Default constructor initializes the represented rational number to zero.
  Fraction() = default;

  /// Construct a Fraction from a numerator and denominator.
  Fraction(const MPInt &oNum, const MPInt &oDen = MPInt(1)) : num(oNum), den(oDen) {
    if (den < 0) {
      num = -num;
      den = -den;
    }
  }
  /// Overloads for passing literals.
  Fraction(const MPInt &num, int64_t den = 1) : Fraction(num, MPInt(den)) {}
  Fraction(int64_t num, const MPInt &den = MPInt(1)) : Fraction(MPInt(num), den) {}
  Fraction(int64_t num, int64_t den) : Fraction(MPInt(num), MPInt(den)) {}

  // Return the value of the fraction as an integer. This should only be called
  // when the fraction's value is really an integer.
  MPInt getAsInteger() const {
    assert(num % den == 0 && "Get as integer called on non-integral fraction!");
    return num / den;
  }

  llvm::raw_ostream &print(llvm::raw_ostream &os) const {
    return os << "(" << num << "/" << den << ")";
  }

  /// The numerator and denominator, respectively. The denominator is always
  /// positive.
  MPInt num{0}, den{1};
};

/// Three-way comparison between two fractions.
/// Returns +1, 0, and -1 if the first fraction is greater than, equal to, or
/// less than the second fraction, respectively.
inline int compare(const Fraction &x, const Fraction &y) {
  MPInt diff = x.num * y.den - y.num * x.den;
  if (diff > 0)
    return +1;
  if (diff < 0)
    return -1;
  return 0;
}

inline MPInt floor(const Fraction &f) { return floorDiv(f.num, f.den); }

inline MPInt ceil(const Fraction &f) { return ceilDiv(f.num, f.den); }

inline Fraction reduce(const Fraction &f) {
  MPInt gcd = presburger::gcdRange({f.num, f.den});
  if (f.num == MPInt(0))
    return Fraction(MPInt(0), MPInt(1));
  return Fraction(f.num / gcd, f.den / gcd);
}

inline Fraction abs(const Fraction &f)  { return Fraction(abs(f.num), f.den); }

inline Fraction operator-(const Fraction &x) { return Fraction(-x.num, x.den); }

inline bool operator<(const Fraction &x, const Fraction &y) {
  return compare(x, y) < 0;
}

inline bool operator<=(const Fraction &x, const Fraction &y) {
  return compare(x, y) <= 0;
}

inline bool operator==(const Fraction &x, const Fraction &y) {
  return compare(x, y) == 0;
}

inline bool operator!=(const Fraction &x, const Fraction &y) {
  return compare(x, y) != 0;
}

inline bool operator>(const Fraction &x, const Fraction &y) {
  return compare(x, y) > 0;
}

inline bool operator>=(const Fraction &x, const Fraction &y) {
  return compare(x, y) >= 0;
}

inline Fraction operator*(const Fraction &x, const Fraction &y) {
  return reduce(Fraction(x.num * y.num, x.den * y.den));
}

inline Fraction operator/(const Fraction &x, const Fraction &y) {
  return reduce(Fraction(x.num * y.den, x.den * y.num));
}

inline Fraction operator+(const Fraction &x, const Fraction &y) {
  return reduce(Fraction(x.num * y.den + x.den * y.num, x.den * y.den));
}

inline Fraction operator-(const Fraction &x, const Fraction &y) {
  return reduce(Fraction(x.num * y.den - x.den * y.num, x.den * y.den));
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Fraction &x) {
  x.print(os);
  return os;
}

} // namespace presburger
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_FRACTION_H
