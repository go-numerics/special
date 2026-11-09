package special

import (
	"math"
	"math/big"
	"testing"

	"github.com/go-numerics/special/dd"
)

// bigBinomial returns C(n, k) rounded to nearest-even float64.
func bigBinomial(n, k int64) float64 {
	b := new(big.Int).Binomial(n, k)
	f, _ := new(big.Float).SetInt(b).Float64()
	return f
}

// Comb must return the correctly rounded binomial coefficient wherever its
// product is carried (k <= 64): the integer recurrence below 2^64 and the
// double-double product beyond.
func TestCombCorrectlyRounded(t *testing.T) {
	for n := int64(0); n <= 400; n++ {
		for k := int64(0); k <= n; k++ {
			m := min(k, n-k)
			if m > 64 {
				continue
			}
			got, want := Comb(n, k), bigBinomial(n, k)
			if got != want {
				t.Fatalf("Comb(%d, %d) = %v, want %v", n, k, got, want)
			}
		}
	}
}

// combExact refuses (rather than wraps) when a partial value leaves 64 bits.
func TestCombExactRefuses(t *testing.T) {
	if c, ok := combExact(100, 30); ok {
		t.Fatalf("combExact(100, 30) = %d, true: C(130, 30) does not fit", c)
	}
	if c, ok := combExact(60, 4); !ok || c != 635376 {
		t.Fatalf("combExact(60, 4) = %d, %v, want C(64, 4) = 635376", c, ok)
	}
}

func TestFactorial2CorrectlyRounded(t *testing.T) {
	for n := 0; n <= 300; n++ {
		p := big.NewInt(1)
		for k := n; k > 1; k -= 2 {
			p.Mul(p, big.NewInt(int64(k)))
		}
		want, _ := new(big.Float).SetInt(p).Float64()
		if got := Factorial2(n); got != want {
			t.Fatalf("Factorial2(%d) = %v, want %v", n, got, want)
		}
		if got := factorial2DD(n); got != want {
			t.Fatalf("factorial2DD(%d) = %v, want %v", n, got, want)
		}
	}
}

// GammaSign for x < 0 is (-1)^ceil(-x): checked against math.Mod on a sweep
// that includes the integers' neighbours and huge arguments.
func TestGammaSignNegative(t *testing.T) {
	ref := func(x float64) float64 {
		if math.Mod(math.Floor(x), 2) == 0 {
			return 1
		}
		return -1
	}
	check := func(x float64) {
		t.Helper()
		if x == math.Trunc(x) {
			return
		}
		if got := GammaSign(x); got != ref(x) {
			t.Fatalf("GammaSign(%v) = %v, want %v", x, got, ref(x))
		}
	}
	for i := 1; i < 3000; i++ {
		x := float64(-float64(i) * 0.0371)
		check(x)
		check(math.Nextafter(x, 0))
		check(math.Nextafter(x, math.Inf(-1)))
		check(-float64(i) + 0.5)
	}
	for _, x := range []float64{-0x1p52 - 0.5, -0x1p51 - 0.5, -4503599627370495.5, -1e15 - 0.5, -0.5, -1.5, -2.5} {
		check(x)
	}
}

// The pooled scratch of Softmax must not leak between calls of different
// lengths and callers.
func TestSoftmaxScratchReuse(t *testing.T) {
	long := make([]float64, 5000)
	for i := range long {
		long[i] = float64(i%37) - 18
	}
	want := Softmax(nil, long)
	short := []float64{1, 2, 3}
	ws := Softmax(nil, short)
	for r := 0; r < 4; r++ {
		got := Softmax(nil, long)
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("round %d: long[%d] = %v, want %v", r, i, got[i], want[i])
			}
		}
		gs := Softmax(nil, short)
		for i := range gs {
			if gs[i] != ws[i] {
				t.Fatalf("round %d: short[%d] = %v, want %v", r, i, gs[i], ws[i])
			}
		}
	}
}

func gapsRand(seed uint64) func() float64 {
	s := seed
	return func() float64 { // splitmix64 -> [0, 1)
		s += 0x9e3779b97f4a7c15
		z := s
		z = (z ^ z>>30) * 0xbf58476d1ce4e5b9
		z = (z ^ z>>27) * 0x94d049bb133111eb
		z ^= z >> 31
		return float64(float64(z>>11) * 0x1p-53)
	}
}

// The certified fast paths must return the reference's float64 wherever they
// claim a result, and claim most points.
func TestXlog1pyFastEqualsReference(t *testing.T) {
	r := gapsRand(1)
	n, hit := 300000, 0 // 3e6 in the PR's evidence run; -race and coverage cost 20x on Windows
	for i := 0; i < n; i++ {
		x := float64((r() - 0.5) * math.Exp2(float64(int(float64(r()*80))-40)))
		y := float64(math.Exp2(float64(int(float64(r()*70))-20)) * (r() + 0.5))
		if r() < 0.3 {
			y = -r()
		}
		if y <= -1 {
			continue
		}
		v, ok := xlog1pyFast(x, y)
		if !ok {
			continue
		}
		hit++
		if want := dd.MulF(log1pDD(dd.F(y)), x).Float(); v != want {
			t.Fatalf("xlog1pyFast(%v, %v) = %v, reference %v", x, y, v, want)
		}
	}
	t.Logf("certified %d of %d", hit, n)
	if hit < n*9/10 {
		t.Fatalf("only %d of %d certified", hit, n)
	}
}

func TestFactorialFastEqualsReference(t *testing.T) {
	r := gapsRand(2)
	n, hit := 300000, 0
	for i := 0; i < n; i++ {
		x := 10 + float64(160*r())
		if x == math.Trunc(x) {
			continue
		}
		v, ok := factorialLargeFast(x)
		if !ok {
			continue
		}
		hit++
		if want := dd.Exp(dd.Add(dd.Log(x), logGammaPos(x))); v != want {
			t.Fatalf("factorialLargeFast(%v) = %v, reference %v", x, v, want)
		}
	}
	t.Logf("certified %d of %d", hit, n)
	if hit < n*9/10 {
		t.Fatalf("only %d of %d certified", hit, n)
	}
}
