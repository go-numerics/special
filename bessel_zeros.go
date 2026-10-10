package special

import "math"

// JnZeros returns the first nt positive zeros of the Bessel function J_n,
// in increasing order. Since J_{-n} = (-1)^n J_n, a negative n gives the
// zeros of J_|n|. nt <= 0 gives an empty result. (x = 0, a zero of J_n for
// n != 0, is not counted, as in SciPy's jn_zeros.)
//
// Algorithm: an initial estimate from McMahon's expansion (DLMF 10.21.19)
// for n = 0, and from Olver's uniform expansion (A&S 9.5.22, with the Airy
// zeros of A&S 10.4.94) for n >= 1, as in Boost.Math bessel_jy_zero.hpp;
// then Halley's iteration on Jn, with J'_n = J_{n-1} - (n/x) J_n and J''_n
// from Bessel's equation, run to convergence in double precision.
func JnZeros(n int, nt int) []float64 {
	if n < 0 {
		n = -n
	}
	if nt <= 0 {
		return []float64{}
	}
	out := make([]float64, nt)
	prev := 0.0
	for k := 1; k <= nt; k++ {
		g := besselJZeroGuess(n, k)
		z := besselJZeroRefine(n, g)
		if !(z > prev+1) { // converged to an earlier zero: step on from it
			z = besselJZeroRefine(n, prev+math.Pi)
		}
		out[k-1] = z
		prev = z
	}
	return out
}

// besselJZeroGuess estimates the k-th positive zero of J_n.
func besselJZeroGuess(n, k int) float64 {
	v := float64(n)
	if n == 0 {
		// McMahon, DLMF 10.21.19 with a = (k - 1/4) pi
		a := (float64(k) - 0.25) * math.Pi
		mu := 4 * v * v
		m1 := mu - 1
		e := 1 / (8 * a)
		e2 := e * e
		t3 := m1 * 4 * (7*mu - 31) / 3
		t5 := m1 * 32 * ((83*mu-982)*mu + 3779) / 15
		t7 := m1 * 64 * (((6949*mu-153855)*mu+1585743)*mu - 6277237) / 105
		return a + ((((-t7*e2-t5)*e2-t3)*e2 - m1) * e)
	}
	// Olver: j ~ v z(zeta), zeta = v^(-2/3) |a_k|, a_k the k-th zero of Ai,
	// with (2/3) zeta^(3/2) = sqrt(z^2 - 1) - arcsec z.
	t := 3 * math.Pi * (4*float64(k) - 1) / 8
	ti := 1 / (t * t)
	ak := math.Pow(t, 2.0/3) * (1 + ti*(5.0/48+ti*(-5.0/36+ti*(77125.0/82944-ti*108056875.0/6967296))))
	zeta := ak / math.Pow(v, 2.0/3)
	w := 2.0 / 3 * zeta * math.Sqrt(zeta)
	b := -(w + math.Pi/2)
	z := (-b + math.Sqrt(b*b-2)) / 2
	for i := 0; i < 20; i++ {
		s := math.Sqrt(z*z - 1)
		f := s - math.Acos(1/z) - w
		dz := f / (s / z)
		z -= dz
		if z < 1 {
			z = 1 + 1e-6
		}
		if math.Abs(dz) < 1e-12*z {
			break
		}
	}
	return v * z
}

// besselJZeroRefine runs Halley's iteration for J_n from x.
func besselJZeroRefine(n int, x float64) float64 {
	v := float64(n)
	for i := 0; i < 60; i++ {
		f := Jn(n, x)
		var d float64
		if n == 0 {
			d = -J1(x)
		} else {
			d = Jn(n-1, x) - v/x*f
		}
		d2 := -d/x - (1-v*v/(x*x))*f
		dx := f / d / (1 - f*d2/(2*d*d))
		x -= dx
		if math.Abs(dx) <= 0x1p-53*x {
			break
		}
	}
	return x
}
