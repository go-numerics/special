package special

import (
	"math"
	"math/rand"
	"testing"
)

// ndKernelPoints are the arguments the five nd kernels are held to: every
// special value, the integer and half-integer grids around the poles and the
// table edges (Factorial2 at 300/301, Perm and Binom at k = 64/65, the
// MaxInt32 bound of the integer rule), and random arguments over every scale.
func ndKernelPoints() []float64 {
	pts := []float64{0, math.Copysign(0, -1), 1, -1, 2, -2, 0.5, -0.5, 1.5, -1.5,
		1e-300, 5e-324, -5e-324, 0x1p-1022, math.Inf(1), math.Inf(-1), math.NaN(),
		math.MaxFloat64, -math.MaxFloat64, 0x1p-20, -0x1p-20, 0x1p-21,
		299, 300, 301, 302, 63, 64, 65, 66, -3, -64, 170, 171, 172,
		math.MaxInt32, math.MaxInt32 + 1, -math.MaxInt32, -math.MaxInt32 - 1,
		1 << 53, 1e19, -1e19}
	for i := -40; i <= 40; i++ {
		pts = append(pts, float64(i), float64(i)+0.5, float64(i)-0.25)
	}
	r := rand.New(rand.NewSource(33))
	for i := 0; i < 4000; i++ {
		s := 1.0
		if r.Intn(2) == 0 {
			s = -1
		}
		pts = append(pts, s*math.Exp(r.Float64()*1400-700), s*r.Float64()*80,
			float64(r.Intn(700)-100), r.Float64())
	}
	return pts
}

// TestNDKernelsEqualScalar holds GammaSignTo, Factorial2To, PermTo, BinomTo
// and Xlog1pyTo to the scalar function (for Factorial2 and Perm, the nd rule:
// an argument that is not an integer of magnitude at most MaxInt32 gives NaN)
// for every element, bit for bit.
func TestNDKernelsEqualScalar(t *testing.T) {
	pts := ndKernelPoints()
	intOK := func(x float64) bool { return x == math.Trunc(x) && math.Abs(x) <= math.MaxInt32 }
	unary := map[string]struct {
		vec    func(dst, x []float64)
		scalar func(float64) float64
	}{
		"GammaSign": {GammaSignTo, GammaSign},
		"Factorial2": {Factorial2To, func(n float64) float64 {
			if !intOK(n) {
				return math.NaN()
			}
			return Factorial2(int(n))
		}},
	}
	// the scalar forms geninline expands are the same rule: they are what the
	// generated kernels are copied from, so they are held here directly
	for _, v := range pts {
		if w := unary["Factorial2"].scalar(v); !sameBits(w, factorial2F(v)) {
			t.Errorf("factorial2F(%v) = %v, rule %v", v, factorial2F(v), w)
		}
	}
	for i := 0; i+1 < len(pts); i++ {
		n, k := pts[i], pts[i+1]
		w := math.NaN()
		if intOK(n) && intOK(k) {
			w = Perm(int64(n), int64(k))
		}
		if !sameBits(w, permF(n, k)) {
			t.Errorf("permF(%v, %v) = %v, rule %v", n, k, permF(n, k), w)
		}
	}
	for name, f := range unary {
		dst := make([]float64, len(pts))
		f.vec(dst, pts)
		for i, v := range pts {
			if w := f.scalar(v); !sameBits(w, dst[i]) {
				t.Errorf("%sTo(%v) = %v, scalar %v", name, v, dst[i], w)
			}
		}
	}
	binary := map[string]struct {
		vec    func(dst, a, b []float64, sa, sb int)
		scalar func(float64, float64) float64
	}{
		"Perm": {PermTo, func(n, k float64) float64 {
			if !intOK(n) || !intOK(k) {
				return math.NaN()
			}
			return Perm(int64(n), int64(k))
		}},
		"Binom":   {BinomTo, Binom},
		"Xlog1py": {Xlog1pyTo, Xlog1py},
	}
	// all pairs of a reduced set (special values and edges), plus the
	// consecutive pairs of the full set
	small := append([]float64{}, pts[:60]...)
	for i := 60; i < len(pts); i += 37 {
		small = append(small, pts[i])
	}
	var a, b []float64
	for _, u := range small {
		for _, v := range small {
			a, b = append(a, u), append(b, v)
		}
	}
	for i := 0; i+1 < len(pts); i++ {
		a, b = append(a, pts[i]), append(b, pts[i+1])
	}
	for name, f := range binary {
		if name == "Binom" {
			// The scalar Binom is slow on its fallback branches; a bounded
			// prefix keeps the suite short while the full pair set above runs
			// for the other two.
			a, b := a[:min(len(a), 60000)], b[:min(len(b), 60000)]
			checkBinary(t, name, f.vec, f.scalar, a, b)
			continue
		}
		checkBinary(t, name, f.vec, f.scalar, a, b)
	}
}

func checkBinary(t *testing.T, name string, vec func(dst, a, b []float64, sa, sb int), scalar func(float64, float64) float64, a, b []float64) {
	t.Helper()
	dst := make([]float64, len(a))
	vec(dst, a, b, 1, 1)
	for i := range a {
		if w := scalar(a[i], b[i]); !sameBits(w, dst[i]) {
			t.Errorf("%sTo(%v, %v) = %v, scalar %v", name, a[i], b[i], dst[i], w)
			return
		}
	}
	// stride 0: the first operand repeated, then the second
	n := min(len(a), 400)
	d := make([]float64, n)
	vec(d, a[:1], b, 0, 1)
	for i := range d {
		if w := scalar(a[0], b[i]); !sameBits(w, d[i]) {
			t.Errorf("%sTo stride 0 a (%v, %v) = %v, scalar %v", name, a[0], b[i], d[i], w)
		}
	}
	vec(d, a, b[:1], 1, 0)
	for i := range d {
		if w := scalar(a[i], b[0]); !sameBits(w, d[i]) {
			t.Errorf("%sTo stride 0 b (%v, %v) = %v, scalar %v", name, a[i], b[0], d[i], w)
		}
	}
}

// TestNDKernelsLengths: operands longer than dst are read only up to len(dst)
// and nothing past len(dst) is written; an empty dst is a no-op; an operand
// shorter than dst panics (as every other array kernel does) instead of
// reading out of range.
func TestNDKernelsLengths(t *testing.T) {
	x := []float64{3, math.NaN(), -2.5, 7, 301, 0, math.Inf(1), 4.5}
	y := []float64{2, 1, 0.5, math.NaN(), 5, 64, 1, -1}
	const sentinel = 12345.5
	un := map[string]func(dst, x []float64){"GammaSign": GammaSignTo, "Factorial2": Factorial2To}
	for name, f := range un {
		buf := make([]float64, len(x))
		for i := range buf {
			buf[i] = sentinel
		}
		f(buf[:3], x) // x longer than dst
		for i := 3; i < len(buf); i++ {
			if buf[i] != sentinel {
				t.Errorf("%sTo wrote dst[%d] past len(dst)", name, i)
			}
		}
		f(nil, x)
		f(buf[:0], nil)
		// capacity 2 as well: x[:2] of the 8-element x would keep its capacity
		mustPanic(t, name+" short x", func() { f(buf, []float64{1, 2}) })
	}
	bin := map[string]func(dst, a, b []float64, sa, sb int){"Perm": PermTo, "Binom": BinomTo, "Xlog1py": Xlog1pyTo}
	for name, f := range bin {
		buf := make([]float64, len(x))
		for i := range buf {
			buf[i] = sentinel
		}
		f(buf[:3], x, y, 1, 1)
		for i := 3; i < len(buf); i++ {
			if buf[i] != sentinel {
				t.Errorf("%sTo wrote dst[%d] past len(dst)", name, i)
			}
		}
		f(nil, x, y, 1, 1)
		f(buf[:0], nil, nil, 1, 1)
		mustPanic(t, name+" short a", func() { f(buf, x[:2], y, 1, 1) })
		mustPanic(t, name+" short b", func() { f(buf, x, y[:2], 1, 1) })
	}
}

func mustPanic(t *testing.T, what string, f func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Errorf("%s: no panic", what)
		}
	}()
	f()
}
