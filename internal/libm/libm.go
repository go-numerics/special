// Package libm holds the functions of Go's package math whose results
// depend on the architecture, in a form whose results do not.
//
// math.Exp, Log, Log1p, Pow, Sin, ... have assembly versions on some
// architectures (amd64, s390x, arm64 for some), and their pure-Go versions
// are compiled with fused multiply-adds where the compiler fuses (arm64,
// ppc64le, s390x, riscv64, loong64) and without elsewhere. The same call can
// therefore return different bits on different machines. The files here are
// the pure-Go implementations of Go 1.27.3's src/math (Copyright The Go
// Authors, BSD-style license, reproduced in LICENSE-GO), with the assembly
// dispatch switched off (arch.go) and every product rounded explicitly
// (tools/fmalint --fix), so that they return the same bits everywhere.
// The exact functions (Abs, Sqrt, Floor, Ldexp, Frexp, ...) are taken from
// package math unchanged: they are correctly rounded or exact on every
// architecture.
package libm

import "math"

const (
	Pi    = math.Pi
	Ln2   = math.Ln2
	Log2E = math.Log2E
	Sqrt2 = math.Sqrt2

	MaxFloat64             = math.MaxFloat64
	SmallestNonzeroFloat64 = math.SmallestNonzeroFloat64

	uvnan    = 0x7FF8000000000001
	uvinf    = 0x7FF0000000000000
	uvneginf = 0xFFF0000000000000
	uvone    = 0x3FF0000000000000
	mask     = 0x7FF
	shift    = 64 - 11 - 1
	bias     = 1023
	signMask = 1 << 63
	fracMask = 1<<shift - 1
)

// Exact on every architecture.
func Abs(x float64) float64                   { return math.Abs(x) }
func Copysign(f, sign float64) float64        { return math.Copysign(f, sign) }
func Float64bits(f float64) uint64            { return math.Float64bits(f) }
func Float64frombits(b uint64) float64        { return math.Float64frombits(b) }
func Floor(x float64) float64                 { return math.Floor(x) }
func Frexp(f float64) (frac float64, exp int) { return math.Frexp(f) }
func Inf(sign int) float64                    { return math.Inf(sign) }
func IsInf(f float64, sign int) bool          { return math.IsInf(f, sign) }
func IsNaN(f float64) bool                    { return math.IsNaN(f) }
func Ldexp(frac float64, exp int) float64     { return math.Ldexp(frac, exp) }
func Modf(f float64) (int float64, frac float64) {
	return math.Modf(f)
}
func NaN() float64            { return math.NaN() }
func Signbit(x float64) bool  { return math.Signbit(x) }
func Sqrt(x float64) float64  { return math.Sqrt(x) }
func Trunc(x float64) float64 { return math.Trunc(x) }
