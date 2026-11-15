package special

import (
	"math"
	"math/rand"
	"testing"
)

// besselOldJY is what Jv and Yv computed before jv_hankel_fast.go (the
// branches below it), for the comparisons of the tests.
func besselOldJY(v, x float64, y bool) float64 {
	var j, r float64
	if v > 0 {
		j, r = besselJYPos(v, x, !y, y)
	} else {
		j, r = besselJYRefl(v, x, !y, y)
	}
	if y {
		return r
	}
	return j
}

// hankelU is uniform on (-1, 1).
func hankelU(r *rand.Rand) float64 { return float64(r.Float64()*2) - 1 }

// hankelDraw returns a non-integer order of either sign, |v| <= 41, and x in
// (20, 1e4) log-uniformly, or beyond now and then.
func hankelDraw(r *rand.Rand) (v, x float64) {
	for {
		v = float64(hankelU(r) * 41)
		if r.Intn(2) == 0 {
			v = float64(hankelU(r) * 6)
		}
		if v != math.Trunc(v) {
			break
		}
	}
	x = float64(20 * math.Exp(float64(r.Float64()*math.Log(500))))
	if r.Intn(50) == 0 {
		x = 20 + float64(r.Float64()*0.01)
	}
	return v, x
}

// TestHankelDeclineBitIdentical: on 200000 draws, wherever the path declines
// Jv and Yv are the bits they were before it; wherever it answers, the answer
// is within 3 ulp of the old code's (whose own bound is 2 ulp) and Jv, Yv
// return it.
func TestHankelDeclineBitIdentical(t *testing.T) {
	n := 200000
	if testing.Short() {
		n = 20000
	}
	r := rand.New(rand.NewSource(100))
	var answered int
	for i := 0; i < n; i++ {
		v, x := hankelDraw(r)
		for _, y := range []bool{false, true} {
			got, _, ok := hankelFast(v, x, y)
			var top, old float64
			old = besselOldJY(v, x, y)
			if y {
				top = Yv(v, x)
			} else {
				top = Jv(v, x)
			}
			if !ok {
				if math.Float64bits(top) != math.Float64bits(old) {
					t.Fatalf("declined at (v=%v, x=%v, Y=%v): %v, before %v", v, x, y, top, old)
				}
				continue
			}
			answered++
			if top != got {
				t.Fatalf("answered at (v=%v, x=%v, Y=%v): %v, but the function returns %v", v, x, y, got, top)
			}
			if d := math.Abs(got-old) / ulp(old); d > 3 {
				t.Fatalf("(v=%v, x=%v, Y=%v): %v is %.2f ulp from the old %v", v, x, y, got, d, old)
			}
		}
	}
	if answered < n/2 {
		t.Errorf("only %d of %d answered", answered, 2*n)
	}
	t.Logf("%d of %d answered", answered, 2*n)
}

// TestHankelGuards: the range guards decline, they do not panic and they do
// not extrapolate: x <= 20, x at the top, |v| above 40, NaN, infinities.
func TestHankelGuards(t *testing.T) {
	for _, c := range []struct {
		v, x float64
		ok   bool
	}{
		{0.5, 20, false},
		{0.5, math.Nextafter(20, 0), false},
		{0.5, 20 + 0x1p-40, true},
		{0.5, 0x1p500, false},
		{0.3, math.Inf(1), false},
		{0.3, math.NaN(), false},
		{math.NaN(), 30, false},
		{math.Inf(1), 30, false},
		{40.5, 30, false}, // |v| > 40
		{-40.5, 30, false},
		{40, 3000, true},
		{-40, 3000, true},
		{0.3, 0, false},
		{0.3, -30, false},
	} {
		for _, y := range []bool{false, true} {
			if _, _, ok := hankelFast(c.v, c.x, y); ok != c.ok {
				t.Errorf("hankelFast(%v, %v, Y=%v): ok = %v, want %v", c.v, c.x, y, ok, c.ok)
			}
		}
	}
}

// TestHankelZeros: next to a zero of J_{1/2} (sin x, x = k pi) the path
// declines, where the relative error cannot be certified, and a little off it
// answers; the same for Y_{1/2} (-cos x, x = (k + 1/2) pi).
func TestHankelZeros(t *testing.T) {
	for k := 7; k < 60; k += 5 {
		zj := float64(float64(k) * math.Pi)
		zy := float64((float64(k) + 0.5) * math.Pi)
		for _, w := range []struct {
			z float64
			y bool
		}{{zj, false}, {zy, true}} {
			if _, _, ok := hankelFast(0.5, w.z, w.y); ok {
				t.Errorf("answers at the zero %v (Y=%v)", w.z, w.y)
			}
			if _, _, ok := hankelFast(0.5, w.z+0.5, w.y); !ok {
				t.Errorf("declines at 0.5 from the zero %v (Y=%v)", w.z, w.y)
			}
		}
	}
}

// TestHankelHalfInteger: for v = n + 1/2 the series terminate (a_k = 0 for
// k > n: the remainder is exactly 0) and the answer is the closed form
// J_{n+1/2}, Y_{n+1/2}, checked against the oracle.
func TestHankelHalfInteger(t *testing.T) {
	r := rand.New(rand.NewSource(5))
	pts := 40
	if testing.Short() {
		pts = 8
	}
	for n := -4; n <= 12; n++ {
		v := float64(n) + 0.5
		for i := 0; i < pts; i++ {
			x := 20.1 + float64(r.Float64()*80)
			for _, y := range []bool{false, true} {
				got, _, ok := hankelFast(v, x, y)
				if !ok {
					continue
				}
				want := bigJv(v, x)
				if y {
					want = bigYv(v, x)
				}
				if u := jvUlps(got, want); u > 0.63 {
					t.Fatalf("v=%v x=%v Y=%v: %.3f ulp", v, x, y, u)
				}
			}
		}
	}
}

// TestHankelCertifiedBoundary holds the acceptance test from both sides:
// e = 2^-56 |F| is accepted, one ulp above it is not, for either sign of F.
func TestHankelCertifiedBoundary(t *testing.T) {
	for _, f := range []float64{1, -1, 0.3, -0.3, 1.5e-5} {
		e := float64(0x1p-56 * math.Abs(f))
		if !hankelCertified(e, f) {
			t.Errorf("e = 2^-56 |F| refused at F = %v", f)
		}
		if hankelCertified(math.Nextafter(e, 1), f) {
			t.Errorf("e above 2^-56 |F| accepted at F = %v", f)
		}
	}
	if hankelCertified(math.NaN(), 1) || hankelCertified(1, math.NaN()) {
		t.Error("NaN accepted")
	}
}

// TestHankelBounds: the certificate e returned is the bound that was
// compared with 2^-56 |F|: it is positive, below 2^-56 |a F| and at least the
// double-double floor, and (the sin/cos error alone) above 2^-59 times the
// amplitude: a certificate that ignored the trigonometric error would be
// smaller.
func TestHankelBounds(t *testing.T) {
	r := rand.New(rand.NewSource(11))
	for i := 0; i < 2000; i++ {
		v, x := hankelDraw(r)
		for _, y := range []bool{false, true} {
			got, e, ok := hankelFast(v, x, y)
			if !ok {
				continue
			}
			a := math.Sqrt(2 / float64(math.Pi*x))
			if !(e > 0 && e <= float64(float64(0x1p-56*math.Abs(got))*1.0000001)+1e-300) {
				t.Fatalf("v=%v x=%v: e = %g for a result %g", v, x, e, got)
			}
			if e < float64(float64(a*hankelTrigErr)*0.99) {
				t.Fatalf("v=%v x=%v: e = %g is below the trigonometric error alone, %g", v, x, e, float64(a*hankelTrigErr))
			}
		}
	}
}
