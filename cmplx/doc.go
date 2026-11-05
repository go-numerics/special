// Package cmplx is a drop-in superset of the standard library's math/cmplx:
// the same names and signatures, with the special values of C11 Annex G
// (ISO/IEC 9899:2011 G.5.1, G.6) and an accuracy contract measured per
// component against mpmath (docs/design-complex.md).
//
// A file switches from math/cmplx by changing its import path. The functions
// that are one call to a correctly specified real function (Abs, Phase,
// Conj, Polar, Rect, Inf, NaN, IsInf, IsNaN) are those of math/cmplx; the
// others are replaced, because math/cmplx evaluates each formula directly
// and loses the result in the finite range: Tanh(400+1i) is NaN there,
// Exp and Cosh overflow both parts once e^x does although a part is finite
// up to Re z = 1454.9. Additions: Mul and Div (the Annex G product and
// quotient, each part accurate where the builtin operators cancel or
// overflow) and Proj (C11 7.3.9.5).
//
// Accuracy: the error of each part is measured in units in the last place
// of that part's exact value, per region, against mpmath at 192 and 320
// bits (testdata/c_*.tsv.gz, written by tools/gencomplex); the tables are in
// docs/accuracy-complex.md. Every function is safe for concurrent use, does
// not allocate, and returns the same bits on every architecture: products
// that must not be fused with an addition are written float64(a*b), and
// fused ones call math.FMA.
//
// Algorithms and attribution: the exponential, hyperbolic and
// trigonometric functions are FreeBSD msun's (cexp, ccosh, csinh by
// David Schultz and Bruce D. Evans; ctanh by Bruce D. Evans after Kahan;
// BSD-2-Clause), in the form NumPy carries them in
// numpy/_core/src/npymath/npy_math_complex.c.src, with the exponential of
// the overflow range evaluated in double-double by special/dd. Log is
// npymath's npy_clog (after CPython) and Sqrt CACM Algorithm 312 with
// msun's special cases, both carried in double-double. The real sine,
// cosine, tangent and arctangent underneath are this package's (trig.go):
// math.Sincos and math.Tan lose up to 439 ulp near their zeros and poles,
// math.Atan2 the sign of an underflowed quotient. Pow follows NumPy's
// npy_cpow (integer powers by binary powering, here in double-double;
// exp(y log x) otherwise, in double-double).
package cmplx
