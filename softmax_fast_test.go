package special

import (
	"math"
	"math/big"
	"slices"
	"testing"

	"github.com/go-numerics/special/dd"
)

// smShift must return h = fl(v - m) and l with h + l = v - m exactly.
func TestSmShiftExact(t *testing.T) {
	pairs := [][2]float64{
		{-1e-300, 1e300}, {0.1, 3}, {-50.123456789, 49.987654321}, {1e15 + 0.5, 1e15 + 3}, {-0.3, 0.7},
		{-745.5, 0.001}, {1, 1}, {0, 0}, {-0.0, 0.0}, {0x1p-1074, 3}, {-0x1p-1074, 0x1p-1074},
	}
	for _, p := range pairs {
		v, m := p[0], p[1]
		h, l := smShift(v, m)
		exact := new(big.Float).SetPrec(4000).SetFloat64(v)
		exact.Sub(exact, new(big.Float).SetPrec(4000).SetFloat64(m))
		sum := new(big.Float).SetPrec(4000).SetFloat64(h)
		sum.Add(sum, new(big.Float).SetPrec(4000).SetFloat64(l))
		if sum.Cmp(exact) != 0 || h != v-m {
			t.Errorf("smShift(%v, %v) = %v + %v, exact %v", v, m, h, l, exact)
		}
	}
	// -Inf and NaN give a zero low part, not the NaN of the TwoSum.
	for _, v := range []float64{math.Inf(-1), math.NaN()} {
		if h, l := smShift(v, 3); l != 0 || (h == h && !math.IsInf(h, -1)) {
			t.Errorf("smShift(%v, 3) = %v, %v", v, h, l)
		}
	}
	// an overflowing difference: -Inf and no NaN either
	if h, l := smShift(-1e308, 1e308); !math.IsInf(h, -1) || l != 0 {
		t.Errorf("smShift(-1e308, 1e308) = %v, %v", h, l)
	}
}

// The compensated sum: s + c is the exact sum of the terms to 2^-100, and the
// ones the accumulators start with leave nothing behind when the terms are tiny.
func TestSmChunkTinyOthers(t *testing.T) {
	// one 0 and 99 values at -200: the others sum to 99 e^-200 (about 1.4e-85)
	x := make([]float64, 100)
	for i := range x {
		x[i] = -200
	}
	x[37] = 0
	var lo [smSmall]float64
	var buf [smSmall]float64
	s, k := smChunk(buf[:], x[:smSmall], 0, -1, lo[:])
	if k != 1 {
		t.Fatalf("maxima = %d, want 1", k)
	}
	e200, ke := dd.ExpMant(dd.F(-200))
	want := dd.MulF(dd.Ldexp(e200, ke), 63) // 64 elements, one of them the maximum
	if rel, _ := relErr(s.Hi, want); rel > 0x1p-50 || s.Hi == 0 {
		t.Fatalf("sum of the others = %v, want %v (relative error %g)", s.Hi, want.Hi, rel)
	}
}

// LogSumExp takes the double-double evaluation where m + log(sum) cancels: m
// negative, and the log of the sum not below |lse|.
func TestLogSumExpGuard(t *testing.T) {
	for _, x := range [][]float64{
		{-1, -1},          // lse = -1 + ln 2 = -0.307, L = 0.693
		{-0.5, -0.6},      // lse = -0.5 + log1p(e^-0.1) = 0.144, L = 0.644
		{-1, -1, -1, -1},  // lse = -1 + ln 4
		{-0.7, -0.7, -50}, // two maxima
		{-1, -1, -1000},   // a difference below -746: exp 0, skipped
	} {
		m := floatMax(x)
		v, ok := logSumExpFast(x, m)
		if ok {
			t.Errorf("logSumExpFast(%v) = %v, ok: the guard must refuse (|lse| < L)", x, v)
		}
		got, want := LogSumExp(x), logSumExpDD(x, m)
		if got != want {
			t.Errorf("LogSumExp(%v) = %v, double-double path %v", x, got, want)
		}
		if rel, _ := relErr(got, logSumExpRef(x)); rel > 0x1p-52 {
			t.Errorf("LogSumExp(%v) = %v, oracle %v: relative error %g", x, got, logSumExpRef(x).Hi, rel)
		}
	}
	// where |lse| >= L the fast path answers: m >= 0, and m below -2L
	for _, x := range [][]float64{{0, 0}, {3, 1, 2}, {-100, -101}, {-1, -50}, {1e6, 1e6 + 1}} {
		if _, ok := logSumExpFast(x, floatMax(x)); !ok {
			t.Errorf("logSumExpFast(%v): the guard refused", x)
		}
	}
}

// A sum dominated by one term: the others in full relative accuracy, as the
// maximum is counted and not summed (mark -1).
func TestLogSumExpDominant(t *testing.T) {
	x := []float64{20, -20, -21, -30, 20 - 50, -1000}
	got, ref := LogSumExp(x), logSumExpRef(x)
	if rel, _ := relErr(got, ref); rel > logSumExpErr {
		t.Errorf("LogSumExp(%v) = %v, oracle %v, relative error %g", x, got, ref.Hi, rel)
	}
	// 0 and one value at -60: lse = log1p(e^-60) = 8.7e-27, to full relative accuracy
	y := []float64{0, -60}
	got, ref = LogSumExp(y), logSumExpRef(y)
	if rel, _ := relErr(got, ref); rel > logSumExpErr || got == 0 {
		t.Errorf("LogSumExp(%v) = %v, oracle %v, relative error %g", y, got, ref.Hi, rel)
	}
	ls := LogSoftmax(nil, y)
	lr := logSoftmaxRef(y)
	for i := range ls {
		if rel, _ := relErr(ls[i], lr[i]); rel > logSoftmaxErr {
			t.Errorf("LogSoftmax(%v)[%d] = %v, oracle %v, relative error %g", y, i, ls[i], lr[i].Hi, rel)
		}
	}
}

// Softmax and LogSoftmax in place: dst is x.
func TestSoftmaxInPlace(t *testing.T) {
	for _, n := range []int{1, 5, 64, 65, 600, 1500} {
		x := make([]float64, n)
		for i := range x {
			x[i] = float64(float64(i%23)*1.7) - 20
		}
		for _, f := range []func(dst, x []float64) []float64{Softmax, LogSoftmax} {
			want := f(nil, slices.Clone(x))
			y := slices.Clone(x)
			got := f(y, y)
			if &got[0] != &y[0] {
				t.Fatalf("n=%d: the result is not dst", n)
			}
			if !slices.Equal(got, want) {
				t.Fatalf("n=%d: in place differs from out of place", n)
			}
		}
	}
}

// Special values and the ties of the maximum.
func TestSoftmaxSpecial(t *testing.T) {
	inf := math.Inf(-1)
	if r := Softmax(nil, []float64{0, inf}); r[0] != 1 || r[1] != 0 {
		t.Errorf("Softmax(0, -Inf) = %v", r)
	}
	if r := LogSoftmax(nil, []float64{0, inf}); r[0] != 0 || !math.IsInf(r[1], -1) {
		t.Errorf("LogSoftmax(0, -Inf) = %v", r)
	}
	if v := LogSumExp([]float64{inf, 5, inf}); v != 5 {
		t.Errorf("LogSumExp(-Inf, 5, -Inf) = %v", v)
	}
	// -Inf next to a finite maximum below -709: the difference is -Inf, exp 0
	if r := Softmax(nil, []float64{-800, inf, -800}); r[0] != 0.5 || r[1] != 0 || r[2] != 0.5 {
		t.Errorf("Softmax(-800, -Inf, -800) = %v", r)
	}
	// ties: 1/3 within the bound, and the maxima are equal to each other
	r := Softmax(nil, []float64{7, 7, 7})
	if r[0] != r[1] || r[1] != r[2] || math.Abs(r[0]-1.0/3) > softmaxErr/3 {
		t.Errorf("Softmax(7, 7, 7) = %v", r)
	}
	if v, want := LogSumExp([]float64{7, 7, 7}), 7+math.Log(3); math.Abs(v-want) > 4*0x1p-53*want {
		t.Errorf("LogSumExp(7, 7, 7) = %v, want %v", v, want)
	}
	// a subnormal result: 2^-1074 is the absolute spacing
	r = Softmax(nil, []float64{0, -744})
	if rel, abs := relErr(r[1], softmaxRef([]float64{0, -744})[1]); abs > 0x1p-1073 {
		t.Errorf("Softmax(0, -744)[1] = %v: relative error %g, absolute %g", r[1], rel, abs)
	}
	// values below the smallest subnormal give exactly 0
	if r := Softmax(nil, []float64{0, -800}); r[1] != 0 {
		t.Errorf("Softmax(0, -800)[1] = %v", r[1])
	}
	// NaN and +Inf anywhere: NaN everywhere; all -Inf: NaN
	for _, x := range [][]float64{{1, math.NaN(), 3}, {1, math.Inf(1)}, {inf, inf}} {
		for _, f := range []func(dst, x []float64) []float64{Softmax, LogSoftmax} {
			for i, v := range f(nil, x) {
				if !math.IsNaN(v) {
					t.Errorf("%v: result[%d] = %v, want NaN", x, i, v)
				}
			}
		}
	}
	if v := LogSumExp([]float64{1, math.NaN()}); !math.IsNaN(v) {
		t.Errorf("LogSumExp(1, NaN) = %v", v)
	}
	if v := LogSumExp([]float64{1, math.Inf(1)}); !math.IsInf(v, 1) {
		t.Errorf("LogSumExp(1, +Inf) = %v", v)
	}
	if v := LogSumExp([]float64{inf, inf}); !math.IsInf(v, -1) {
		t.Errorf("LogSumExp(-Inf, -Inf) = %v", v)
	}
}

// Every length around the unrolling of the sum (4) and the chunks (64, 512).
func TestSoftmaxLengths(t *testing.T) {
	for n := 1; n <= 70; n++ {
		smCheckLength(t, n)
	}
	for _, n := range []int{127, 128, 255, 510, 511, 512, 513, 514, 515, 1023, 1024, 1025, 1535, 1536, 1537} {
		smCheckLength(t, n)
	}
}

func smCheckLength(t *testing.T, n int) {
	t.Helper()
	x := make([]float64, n)
	for i := range x {
		x[i] = math.Mod(float64(float64(i)*7.3), 40) - 25
	}
	got := Softmax(nil, x)
	ref := softmaxRef(x)
	lg := LogSoftmax(nil, x)
	lr := logSoftmaxRef(x)
	for i := range x {
		if rel, _ := relErr(got[i], ref[i]); rel > softmaxErr {
			t.Fatalf("n=%d: Softmax[%d] = %v, oracle %v, relative error %g", n, i, got[i], ref[i].Hi, rel)
		}
		if rel, _ := relErr(lg[i], lr[i]); rel > logSoftmaxErr {
			t.Fatalf("n=%d: LogSoftmax[%d] = %v, oracle %v, relative error %g", n, i, lg[i], lr[i].Hi, rel)
		}
	}
	// the sum of the probabilities is 1 to the bound
	s := 0.0
	for _, v := range got {
		s += v
	}
	if math.Abs(s-1) > softmaxErr*2+float64(float64(n)*0x1p-53) {
		t.Fatalf("n=%d: Σ Softmax = %v", n, s)
	}
}

// The exact shift is what separates the result from a plain x - m: 18 to 199 u
// of error on the inputs below when exp is taken of the rounded difference.
func TestSoftmaxExactShiftMatters(t *testing.T) {
	x := make([]float64, 1000)
	for i := range x {
		x[i] = -700 * float64(i) / 999 // differences far from exact
	}
	x[0] = 3.14159265358979
	m := floatMax(x)
	plain := make([]float64, len(x))
	for i, v := range x {
		plain[i] = math.Exp(v - m)
	}
	ref := softmaxRef(x)
	worstPlain, worst := 0.0, 0.0
	got := Softmax(nil, x)
	for i := range x {
		if ref[i].Hi < 0x1p-1022 {
			continue
		}
		p := plain[i] / floatSum(plain)
		rp, _ := relErr(p, ref[i])
		rg, _ := relErr(got[i], ref[i])
		worstPlain, worst = math.Max(worstPlain, rp), math.Max(worst, rg)
	}
	if worst > softmaxErr || worstPlain < 8*softmaxErr {
		t.Errorf("exact shift: error %g u, plain x - m: %g u (the plain form must be well above the bound)", inU(worst), inU(worstPlain))
	}
}

func floatSum(x []float64) float64 {
	s := 0.0
	for _, v := range x {
		s += v
	}
	return s
}
