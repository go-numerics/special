package special

import "testing"

// The inverse of the incomplete beta function allocates nothing: the
// iteration's evaluator is a value with a method, not a closure (go-numerics/
// special#33). The count is exact under any load.
func TestBetaIncInvAllocatesNothing(t *testing.T) {
	cases := [][3]float64{{2, 3, 0.4}, {0.5, 0.5, 0.9}, {30, 40, 0.01}, {0.0005, 2, 0.3}, {5, 0.0005, 0.7}, {10, 10, 0.3}}
	for _, c := range cases {
		for name, f := range map[string]func(a, b, p float64) float64{"BetaIncInv": BetaIncInv, "BetaIncCInv": BetaIncCInv} {
			if n := testing.AllocsPerRun(20, func() { sinkF = f(c[0], c[1], c[2]) }); n != 0 {
				t.Errorf("%s(%v, %v, %v): %v allocations per call, want 0", name, c[0], c[1], c[2], n)
			}
		}
	}
}
