package cmplx

import "math/cmplx"

// The functions below are one call to a correctly specified real function
// (math.Hypot, math.Atan2) or a classification; they are those of
// math/cmplx, unchanged.

// Abs returns the absolute value (modulus) of x, math.Hypot(real(x), imag(x)).
func Abs(x complex128) float64 { return cmplx.Abs(x) }

// Phase returns the argument of x in [-π, π], math.Atan2(imag(x), real(x)).
func Phase(x complex128) float64 { return cmplx.Phase(x) }

// Conj returns the complex conjugate of x.
func Conj(x complex128) complex128 { return cmplx.Conj(x) }

// Polar returns the modulus r and the argument θ of x.
func Polar(x complex128) (r, θ float64) { return cmplx.Polar(x) }

// Rect returns the complex number with modulus r and argument θ.
func Rect(r, θ float64) complex128 { return cmplx.Rect(r, θ) }

// Inf returns a complex infinity, complex(+Inf, +Inf).
func Inf() complex128 { return cmplx.Inf() }

// NaN returns a complex "not-a-number" value.
func NaN() complex128 { return cmplx.NaN() }

// IsInf reports whether either part of x is an infinity.
func IsInf(x complex128) bool { return cmplx.IsInf(x) }

// IsNaN reports whether either part of x is NaN and neither is an infinity.
func IsNaN(x complex128) bool { return cmplx.IsNaN(x) }
