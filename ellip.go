package special

import (
	"math"

	"github.com/go-numerics/special/internal/dd"
)

// agmDD runs the arithmetic-geometric mean of 1 and b > 0 in double-double
// and returns M(1, b) and, for the complete integral of the second kind,
// the sum Σ 2^(n-2) c_n² of Gauss's method (in the arrangement of Carlson,
// "Numerical computation of real or complex elliptic integrals", Numer.
// Algorithms 10, 1995, eq. 46, as Boost.Math's ellint_rg uses it).
func agmDD(b dd.DD) (m, sum dd.DD) {
	x, y := dd.F(1), b
	pow := 0.25
	for i := 0; i < 64; i++ {
		d := dd.Sub(x, y)
		if math.Abs(d.Hi) <= 0x1p-100*x.Hi {
			break
		}
		t := dd.Sqrt(dd.Mul(x, y))
		x = dd.MulF(dd.Add(x, y), 0.5)
		y = t
		pow *= 2
		d = dd.Sub(x, y)
		sum = dd.Add(sum, dd.MulF(dd.Mul(d, d), pow))
	}
	return dd.MulF(dd.Add(x, y), 0.5), sum
}

// EllipK returns the complete elliptic integral of the first kind
// K(m) = ∫₀^{π/2} dθ / sqrt(1 - m sin²θ), parameter m = k² (SciPy's ellipk
// convention), for m <= 1.
//
// Special values: K(1) = +Inf, K(-Inf) = 0, NaN for m > 1 or NaN.
//
// Algorithm: K(m) = π / (2 M(1, sqrt(1-m))) (DLMF 19.8.5) with the
// arithmetic-geometric mean in double-double; 1-m is exact for m >= 1/2 and
// carried in double-double otherwise. Near m = 1 use EllipKm1 with p = 1-m
// to keep the information that m itself cannot hold.
func EllipK(m float64) float64 {
	switch {
	case math.IsNaN(m) || m > 1:
		return math.NaN()
	case m == 1:
		return math.Inf(1)
	case math.IsInf(m, -1):
		return 0
	}
	return ellipKDD(sumDD(1, -m))
}

// EllipKm1 returns K(1 - p) for 0 <= p (SciPy's ellipkm1): the form to use
// near m = 1, where p carries digits that 1-p would lose.
func EllipKm1(p float64) float64 {
	switch {
	case math.IsNaN(p) || p < 0:
		return math.NaN()
	case p == 0:
		return math.Inf(1)
	case math.IsInf(p, 1):
		return 0
	}
	return ellipKDD(dd.F(p))
}

func ellipKDD(p dd.DD) float64 {
	if p.Hi > 1e300 {
		// K(m) ~ log(4 sqrt(-m)) / sqrt(-m) for m -> -Inf (DLMF 19.12.1
		// after the imaginary-modulus transformation).
		s := math.Sqrt(p.Hi)
		return (math.Log(4) + 0.5*math.Log(p.Hi)) / s
	}
	mm, _ := agmDD(dd.Sqrt(p))
	return dd.Div(dd.MulF(piDD, 0.5), mm).Float()
}

// EllipE returns the complete elliptic integral of the second kind
// E(m) = ∫₀^{π/2} sqrt(1 - m sin²θ) dθ, parameter m = k², for m <= 1.
//
// Special values: E(1) = 1, E(-Inf) = +Inf, NaN for m > 1 or NaN.
//
// Algorithm: E(m) = 2 R_G(0, 1-m, 1) computed by the arithmetic-geometric
// mean with Gauss's sum (Carlson 1995, eq. 46, the route of Boost.Math's
// ellint_rg), all in double-double.
func EllipE(m float64) float64 {
	switch {
	case math.IsNaN(m) || m > 1:
		return math.NaN()
	case m == 1:
		return 1
	case math.IsInf(m, -1):
		return math.Inf(1)
	}
	p := sumDD(1, -m)
	if p.Hi > 1e300 {
		// E(m) ~ sqrt(-m) for m -> -Inf.
		return math.Sqrt(p.Hi)
	}
	b := dd.Sqrt(p)
	mm, sum := agmDD(b)
	// 2 R_G = ((1+b)/2)² - sum) π / (2 M)
	h := dd.MulF(dd.AddF(b, 1), 0.5)
	t := dd.Sub(dd.Mul(h, h), sum)
	return dd.Div(dd.Mul(t, dd.MulF(piDD, 0.5)), mm).Float()
}
