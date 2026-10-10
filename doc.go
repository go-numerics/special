// Package special implements special mathematical functions in pure Go
// (no cgo): the gamma family, error functions, incomplete gamma and beta
// functions and their inverses, Bessel functions, zeta functions and
// combinatorial and logistic helpers.
//
// Accuracy is the product. Every function is measured in ULPs against
// mpmath evaluated at high precision over its whole domain, tails, poles
// and special values included; the fixtures and the measured table are in
// the repository (testdata/fixtures, docs/accuracy.md). Each algorithm
// cites its source in a comment at its definition.
//
// Conventions: a function returns NaN outside its domain and ±Inf at a
// pole, as the IEEE 754 operations do; no function panics on a float
// argument.
package special
