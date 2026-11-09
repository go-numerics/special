package cmplx

import (
	"github.com/go-numerics/special/internal/libm"
	std "math/cmplx"
	"testing"
)

var (
	stdExp  = std.Exp
	stdTanh = std.Tanh
)

// z2 adapts a one-argument function to the fixture arguments (x, y).
func z2(f func(complex128) complex128) func([]float64) complex128 {
	return func(a []float64) complex128 { return f(complex(a[0], a[1])) }
}

func init() {
	// The builtin operators are the comparison column of Mul and Div.
	register(accCase{fixture: "mul", std: func(a []float64) complex128 { return complex(a[0], a[1]) * complex(a[2], a[3]) }})
	register(accCase{fixture: "div", std: func(a []float64) complex128 { return complex(a[0], a[1]) / complex(a[2], a[3]) }})
	for _, c := range []struct {
		name string
		f    func(complex128) complex128
	}{
		{"exp", std.Exp}, {"cosh", std.Cosh}, {"sinh", std.Sinh}, {"cos", std.Cos}, {"sin", std.Sin},
		{"tanh", std.Tanh}, {"tan", std.Tan}, {"cot", std.Cot},
	} {
		register(accCase{fixture: c.name, std: z2(c.f)})
	}
}

func TestStdlib(t *testing.T) {
	z := complex(3, -4)
	if Abs(z) != 5 || Phase(z) != libm.Atan2(-4, 3) || Conj(z) != complex(3, 4) {
		t.Error("Abs/Phase/Conj")
	}
	if r, th := Polar(z); r != 5 || th != Phase(z) || Rect(r, th) != std.Rect(r, th) {
		t.Error("Polar/Rect")
	}
	if !IsInf(Inf()) || !IsNaN(NaN()) || IsInf(z) || IsNaN(z) {
		t.Error("Inf/NaN")
	}
}
