package special

import (
	"math"
	"testing"
	"time"
)

// Termination guards for `mutsweep -sites all` (the 28 mutants that were
// HUNG: a loop or branch whose removal sends an iteration to its
// besselMaxIter = 1e8 bound, which is a hang for every practical purpose).
// A converged iteration costs microseconds; every call below must return
// within a deadline orders of magnitude above that. The assertion is
// termination, not the value (values are held by the oracle tests). The file
// sorts first so that a mutant that spins is reported by a failure.

const terminationDeadline = 3 * time.Second

// terminates runs f and fails the test if it has not returned by the
// deadline. A spinning goroutine is abandoned: go test exits regardless.
func terminates(t *testing.T, what string, f func()) {
	t.Helper()
	done := make(chan struct{})
	go func() {
		defer close(done)
		defer func() { _ = recover() }() // a crash is for other tests to report
		f()
	}()
	select {
	case <-done:
	case <-time.After(terminationDeadline):
		t.Fatalf("%s did not return within %v", what, terminationDeadline)
	}
}

var (
	termOrders = []float64{0, 0.25, 0.5, 1, 1.5, 2, 3.7, 10, 25.3, 100}
	termArgs   = []float64{1e-8, 1e-3, 0.3, 0.5, 0.75, 1, 2, 5, 12, 19.9, 20.1, 40, 150, 1e3, 1e4}
	termShapes = []float64{1e-4, 1e-3, 0.01, 0.3, 0.5, 1, 1.5, 3, 10, 50, 1e3, 1e5}
	termProbs  = []float64{1e-300, 1e-10, 0.01, 0.3, 0.5, 0.7, 0.99, 1 - 1e-12}
)

// The domain edges: the special-value guards exist so that these never reach
// the iterations, which have no convergence to find there.
var (
	termInf  = math.Inf(1)
	termNaN  = math.NaN()
	termEdge = []float64{0, math.Copysign(0, -1), -1, -0.5, 1, 1.5, termInf, -termInf, termNaN, 1e30, 1e300, 5e-324}
)

func TestTerminationEdges(t *testing.T) {
	for _, a := range termEdge {
		terminates(t, "BetaInc edges", func() {
			for _, b := range termEdge {
				for _, x := range termEdge {
					BetaInc(a, b, x)
					BetaIncC(a, b, x)
					BetaIncInv(a, b, x)
					BetaIncCInv(a, b, x)
				}
			}
		})
		terminates(t, "Bessel edges", func() {
			for _, x := range termEdge {
				Iv(a, x)
				Ive(a, x)
				Kv(a, x)
				Kve(a, x)
				Jv(a, x)
				Yv(a, x)
			}
		})
		terminates(t, "Zeta edges", func() {
			Zeta(a)
			for _, q := range termEdge {
				HurwitzZeta(a, q)
			}
		})
	}
}

// TestTerminationBetaIncInvUnderflow (incbeta_inv.go:254-255): for b = 1,
// I_u(a, 1) = u^a, so the root is p^(1/a), which is below the smallest
// subnormal when ln(p)/a < -745: the answer is exactly 0, found without
// iterating to a root that does not exist in float64.
func TestTerminationBetaIncInvUnderflow(t *testing.T) {
	for _, c := range [][2]float64{{1e-5, 0.5}, {1e-3, 1e-300}, {0.01, 1e-300}, {0.0009, 1e-290}} {
		a, p := c[0], c[1]
		terminates(t, "BetaIncInv underflow", func() {
			if got := BetaIncInv(a, 1, p); got != 0 {
				t.Errorf("BetaIncInv(%g, 1, %g) = %g, want 0 (p^(1/a) underflows)", a, p, got)
			}
		})
	}
}

// TestTerminationBetaIncInvTinyShapes (incbeta_inv.go:254): for shapes far
// below 1e-3 the power-law starting guess is not to be trusted, and the
// short-cut at 2^-1074 decides the root is below the float64 range; without
// it the bracketing iteration runs on a flat function.
func TestTerminationBetaIncInvTinyShapes(t *testing.T) {
	tiny := []float64{math.Float64frombits(0x35329e72622ff367), math.Float64frombits(0x1b213d7e658883e7), 1e-52, 1e-177, 1e-300, 5e-324, 1e-5}
	for _, a := range tiny {
		terminates(t, "BetaIncInv tiny", func() {
			for _, b := range tiny {
				for _, p := range []float64{1e-10, 0.3, 0.5, 0.7, 0.99} {
					BetaIncInv(a, b, p)
					BetaIncCInv(a, b, p)
					BetaIncInv(b, a, p)
					BetaIncCInv(b, a, p)
				}
			}
		})
	}
}

// TestTerminationKvAsymPair (kv_fast.go:250): below its validity range the
// asymptotic series of K has terms that grow without bound; the loop returns
// ok = false when they first grow (the series ends by itself for w = 1/2,
// where mu = 1). Without that, the terms reach Inf and NaN and the loop never
// ends.
func TestTerminationKvAsymPair(t *testing.T) {
	for _, x := range []float64{0.05, 0.3, 1, 2, 3, 5} {
		for _, w := range []float64{0, 0.5, 1, 1.5, 1.99} {
			terminates(t, "kvAsymPair", func() {
				kvAsymPair(w, x) // the value is for the oracle tests; termination is the point
			})
		}
	}
}

func TestTerminationBessel(t *testing.T) {
	fs := []struct {
		name string
		f    func(v, x float64) float64
	}{{"Iv", Iv}, {"Ive", Ive}, {"Kv", Kv}, {"Kve", Kve}, {"Jv", Jv}, {"Yv", Yv}}
	for _, c := range fs {
		for _, v := range termOrders {
			terminates(t, c.name, func() {
				for _, x := range termArgs {
					c.f(v, x)
					c.f(-v, x)
				}
			})
		}
	}
	for _, n := range []int{0, 1, 2, 7, 30, -3} {
		terminates(t, "Jn/Yn/In/Kn", func() {
			for _, x := range termArgs {
				Jn(n, x)
				Yn(n, x)
				In(n, x)
				if n >= 0 {
					Kn(n, x)
				}
			}
		})
	}
}

func TestTerminationBetaInc(t *testing.T) {
	for _, a := range termShapes {
		terminates(t, "BetaInc", func() {
			for _, b := range termShapes {
				for _, x := range termProbs {
					BetaInc(a, b, x)
					BetaIncC(a, b, x)
					BetaIncInv(a, b, x)
					BetaIncCInv(a, b, x)
				}
			}
		})
	}
}

func TestTerminationZetaComb(t *testing.T) {
	terminates(t, "Zeta", func() {
		for _, s := range []float64{-30.5, -2, -1, 0.5, 1 + 1e-9, 1.5, 2, 10, 60, 1000} {
			Zeta(s)
			for _, q := range []float64{1e-300, 1e-8, 0.3, 1, 7.5, 1e4, 1e15} {
				HurwitzZeta(s, q)
			}
		}
		for _, n := range []int{0, 1, 2, 5, 20} {
			for _, x := range []float64{1e-3, 0.5, 1, 7, 100, 1e6} {
				Polygamma(n, x)
			}
		}
	})
	terminates(t, "Comb/Perm/Binom", func() {
		for _, c := range [][2]int64{{5, -1}, {-5, 2}, {5, 9}, {0, 0}, {5, 5}, {100, 3}, {1 << 40, 1 << 39}, {1 << 62, 3}, {1 << 62, 1 << 61}, {1000, 500}} {
			Comb(c[0], c[1])
			Perm(c[0], c[1])
		}
		for _, c := range [][2]float64{{5, -1}, {-5, 2}, {5.5, 9}, {1e10, 1e5}, {1e300, 2}, {-0.5, 100}, {100.5, 50.25}} {
			Binom(c[0], c[1])
		}
	})
}
