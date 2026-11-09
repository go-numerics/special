package special

import (
	"math"
	"math/rand"
	"testing"

	"github.com/go-numerics/special/dd"
)

// binomRefMant returns the reference's double-double value D = m·2^ke of
// binomBetaDD (its sum before the final rounding).
func binomRefMant(np1, nk1 dd.DD, k float64) (dd.DD, int) {
	k1 := sumDD(k, 1)
	a, b := nk1.Float(), k1.Float()
	da, db := dd.AddF(nk1, -a).Float(), dd.AddF(k1, -b).Float()
	lb := logBetaPos(math.Min(a, b), math.Max(a, b))
	ps := Digamma(a + b)
	lb = dd.AddF(lb, float64(da*(Digamma(a)-ps))+float64(db*(Digamma(b)-ps)))
	return dd.ExpMant(dd.Neg(dd.Add(logDD(np1), lb)))
}

// binomSweepPoint draws (n, k) with min(a, b) in [1/2, 2) and the other
// argument spread log-uniformly over [10, 2^40], half of them with a, b on
// the grid of the refbench rows, half arbitrary reals.
func binomSweepPoint(r *rand.Rand) (n, k float64) {
	p := 0.5 + float64(1.5*r.Float64())
	q := float64(10 * math.Exp2(float64(r.Float64()*math.Log2(0x1p40/10))))
	if r.Intn(2) == 0 {
		q = 10 + float64(40*r.Float64())
	}
	if r.Intn(3) == 0 {
		// both below 10 with p + q < 10, the three-gamma branch
		p = 0.5 + float64(4.5*r.Float64())
		q = p + float64((10-float64(2*p))*r.Float64())
	}
	if r.Intn(2) == 0 {
		// a = p small: n - k + 1 = p, k + 1 = q
		k = q - 1
		n = k + p - 1
	} else {
		// b = p small: k + 1 = p, n - k + 1 = q
		k = p - 1
		n = q + k - 1
	}
	return n, k
}

// TestBinomFastCertified: whenever the fast path answers, it is the
// reference's float64; the largest distance |h + l - D|/D seen stays well
// below binomErr, and the fast path answers on nearly every point.
func TestBinomFastCertified(t *testing.T) {
	total := 2000000 // 2·10^7 points once: 0 mismatches, worst 2^-60.62
	if testing.Short() {
		total = 200000
	}
	r := rand.New(rand.NewSource(33))
	var inDomain, answered, bad int
	worst := 0.0
	for i := 0; i < total; i++ {
		n, k := binomSweepPoint(r)
		np1 := sumDD(n, 1)
		nk1 := dd.AddF(sumDD(n, -k), 1)
		if !(np1.Hi > 0 && nk1.Hi > 0 && k > -1) {
			continue
		}
		h, l, ke, ok := binomFastHL(np1, nk1, k)
		if !ok {
			continue
		}
		inDomain++
		m, kr := binomRefMant(np1, nk1, k)
		if kr != ke {
			// the mantissas straddle a power of two: compare as numbers
			h, l = math.Ldexp(h, ke-kr), math.Ldexp(l, ke-kr)
		}
		d := ((h - m.Hi) + (l - m.Lo)) / m.Hi
		if a := math.Abs(d); a > worst {
			worst = a
		}
		got, certified := binomFast(np1, nk1, k)
		if !certified {
			continue
		}
		answered++
		if want := binomBetaDD(np1, nk1, k); got != want {
			bad++
			if bad < 5 {
				t.Errorf("Binom(%v, %v): fast %v, reference %v", n, k, got, want)
			}
		}
	}
	t.Logf("points %d in domain %d answered %d (%.2f%% fall back), max |h+l-D|/D = 2^%.2f",
		total, inDomain, answered, 100*float64(inDomain-answered)/float64(inDomain), math.Log2(worst))
	if worst > binomErr/4 {
		t.Errorf("max |h+l-D|/D = 2^%.2f, margin to binomErr = 2^%.0f below 4", math.Log2(worst), math.Log2(binomErr))
	}
	if inDomain < total/2 {
		t.Errorf("only %d of %d points in the fast domain", inDomain, total)
	}
}

// TestBinomMatchesReference: over wide random (n, k) of the beta branch and
// the refbench grid of the Binom row, Binom is binomBetaDD's value, bit for
// bit, whichever path answers.
func TestBinomMatchesReference(t *testing.T) {
	total := 1000000
	if testing.Short() {
		total = 100000
	}
	r := rand.New(rand.NewSource(7))
	check := func(n, k float64) {
		np1 := sumDD(n, 1)
		nk1 := dd.AddF(sumDD(n, -k), 1)
		if !(np1.Hi > 0 && nk1.Hi > 0 && k > -1) || k == math.Trunc(k) && k <= 64 {
			return
		}
		if got, want := Binom(n, k), binomBetaDD(np1, nk1, k); got != want {
			t.Fatalf("Binom(%v, %v) = %v, reference %v", n, k, got, want)
		}
	}
	for i := 0; i < total; i++ {
		n := -1 + float64(r.Float64()*math.Exp2(float64(r.Float64()*14)))
		k := -1 + float64(r.Float64()*math.Exp2(float64(r.Float64()*14)))
		check(n, k)
	}
	for i := 0; i < 10000; i++ { // the refbench row
		x := float64(i) / 10000
		check(0.5+float64(39.5*x), float64(40*x))
	}
}
