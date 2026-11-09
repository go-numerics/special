package cmplx

import (
	"math/cmplx"

	"github.com/go-numerics/special/internal/libm"
)

// The functions below are one call to a correctly specified real function
// (Hypot, Atan2, Sincos) or a classification: those of math/cmplx, with the
// real functions taken from internal/libm so that the bits are the same on
// every architecture (math's Hypot, Atan2 and Sincos have assembly versions
// on some).

// Abs returns the absolute value (modulus) of x, Hypot(real(x), imag(x)).
func Abs(x complex128) float64 { return libm.Hypot(real(x), imag(x)) }

// Phase returns the argument of x in [-π, π], Atan2(imag(x), real(x)).
func Phase(x complex128) float64 { return libm.Atan2(imag(x), real(x)) }

// Conj returns the complex conjugate of x.
func Conj(x complex128) complex128 { return cmplx.Conj(x) }

// Polar returns the modulus r and the argument θ of x.
func Polar(x complex128) (r, θ float64) { return Abs(x), Phase(x) }

// Rect returns the complex number with modulus r and argument θ.
func Rect(r, θ float64) complex128 {
	s, c := libm.Sincos(θ)
	return complex(float64(r*c), float64(r*s))
}

// Inf returns a complex infinity, complex(+Inf, +Inf).
func Inf() complex128 { return cmplx.Inf() }

// NaN returns a complex "not-a-number" value.
func NaN() complex128 { return cmplx.NaN() }

// IsInf reports whether either part of x is an infinity.
func IsInf(x complex128) bool { return cmplx.IsInf(x) }

// IsNaN reports whether either part of x is NaN and neither is an infinity.
func IsNaN(x complex128) bool { return cmplx.IsNaN(x) }
