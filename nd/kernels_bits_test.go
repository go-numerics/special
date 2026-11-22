package nd

import (
	"math"
	"math/rand"
	"slices"
	"testing"

	"github.com/go-ndarray/ndarray"
	"github.com/go-numerics/special"
)

// The nd functions that run on the generated kernels (GammaSignTo,
// Factorial2To, PermTo, BinomTo, Xlog1pyTo) are held, bit for bit and NaN
// payload included, to the generic Map1/Map2 path they replaced, over every
// dtype and broadcast case.

func mapGammaSign(x *ndarray.Array) (*ndarray.Array, error) { return Map1(special.GammaSign, x) }
func mapBinom(a, b *ndarray.Array) (*ndarray.Array, error)  { return Map2(special.Binom, a, b) }
func mapXlog1py(a, b *ndarray.Array) (*ndarray.Array, error) {
	return Map2(special.Xlog1py, a, b)
}
func mapPerm(n, k *ndarray.Array) (*ndarray.Array, error) {
	return Map2(func(n, k float64) float64 {
		if !isInt(n) || !isInt(k) {
			return math.NaN()
		}
		return special.Perm(int64(n), int64(k))
	}, n, k)
}
func mapFactorial2(n *ndarray.Array) (*ndarray.Array, error) {
	return Map1(func(n float64) float64 {
		if !isInt(n) {
			return math.NaN()
		}
		return special.Factorial2(int(n))
	}, n)
}

func bitsVals(n int, seed int64) []float64 {
	base := []float64{0, math.Copysign(0, -1), 1, -1, 2, -2, 0.5, -0.5, 3, -3, 5e-324, -5e-324,
		math.Inf(1), math.Inf(-1), math.NaN(), math.Float64frombits(0x7ff8000000000123),
		math.Float64frombits(0xfff8000000000001), math.MaxFloat64, -math.MaxFloat64,
		math.MaxInt32, math.MaxInt32 + 1, -math.MaxInt32, -math.MaxInt32 - 1, 1 << 53, 1 << 63,
		-(1 << 63), 1e300, 64, 65, 300, 301, 170.5, -0.999999, 1e-17, -1e-17, 2147483646.5,
		-9.223372036854776e18, 1.7760895506199026e9, 6.707e72, 2.53e-115, 7.7e57}
	r := rand.New(rand.NewSource(seed))
	v := make([]float64, n)
	for i := range v {
		switch i % 5 {
		case 0:
			v[i] = base[r.Intn(len(base))]
		case 1:
			v[i] = float64(r.Intn(400) - 150)
		case 2:
			v[i] = float64((r.Float64() - 0.3) * 200)
		case 3:
			v[i] = math.Copysign(math.Exp(float64(r.Float64()*1400)-700), r.Float64()-0.5)
		default:
			v[i] = float64(r.Intn(130)) + float64(float64(r.Intn(3))*0.5)
		}
	}
	return v
}

func bitsArr(t *testing.T, v []float64, dt string, shape ...int) *ndarray.Array {
	t.Helper()
	a, err := ndarray.FromData(slices.Clone(v), shape...)
	if err != nil {
		t.Fatal(err)
	}
	if dt != "float64" {
		d, err := ndarray.ParseDType(dt)
		if err != nil {
			t.Fatal(err)
		}
		a = a.AsType(d)
	}
	return a
}

func bitsSame(t *testing.T, name string, got, want *ndarray.Array, gerr, werr error) {
	t.Helper()
	if (gerr == nil) != (werr == nil) {
		t.Fatalf("%s: err %v vs %v", name, gerr, werr)
	}
	if gerr != nil {
		return
	}
	if got.DType() != want.DType() || !slices.Equal(got.Shape(), want.Shape()) {
		t.Fatalf("%s: dtype/shape %v%v vs %v%v", name, got.DType(), got.Shape(), want.DType(), want.Shape())
	}
	g, _ := ndarray.Data[float64](got.AsType(ndarray.Float64))
	w, _ := ndarray.Data[float64](want.AsType(ndarray.Float64))
	bad := 0
	for i := range g {
		if math.Float64bits(g[i]) != math.Float64bits(w[i]) {
			if bad < 5 {
				t.Errorf("%s[%d]: kernel %v (%#x), Map %v (%#x)", name, i, g[i], math.Float64bits(g[i]), w[i], math.Float64bits(w[i]))
			}
			bad++
		}
	}
	if bad > 0 {
		t.Errorf("%s: %d mismatches of %d", name, bad, len(g))
	}
}

func TestKernelsEqualMapPath(t *testing.T) {
	const n = 20000 // above parallelMin, so the chunks have lo > 0
	for _, dt := range []string{"float64", "float32", "int64"} {
		x, y := bitsVals(n, 1), bitsVals(n, 2)
		if dt == "int64" {
			for i := range x {
				x[i], y[i] = float64(int64(i%300)-20), float64(int64(i%90)-10)
			}
		}
		ax := bitsArr(t, x, dt, n)
		g, ge := GammaSign(ax)
		w, we := mapGammaSign(ax)
		bitsSame(t, dt+" GammaSign", g, w, ge, we)
		g, ge = Factorial2(ax)
		w, we = mapFactorial2(ax)
		bitsSame(t, dt+" Factorial2", g, w, ge, we)

		cases := []struct {
			name   string
			sa, sb []int
			xa, yb []float64
		}{
			{"same", []int{n}, []int{n}, x, y},
			{"same2d", []int{100, n / 100}, []int{100, n / 100}, x, y},
			{"scalarA", []int{1}, []int{n}, x[7:8], y},
			{"scalarB", []int{n}, []int{1}, x, y[3:4]},
			{"scalarA2d", []int{1, 1}, []int{n}, x[11:12], y},
			{"scalarB2d", []int{n}, []int{1, 1}, x, y[9:10]},
			{"outer", []int{200, 1}, []int{1, 100}, x[:200], y[:100]},
			{"rowbcast", []int{100, 200}, []int{200}, x, y[:200]},
			{"rankdiff", []int{1, n}, []int{n}, x, y},
			{"empty", []int{0}, []int{1}, nil, y[:1]},
			{"empty2", []int{1, 0}, []int{3, 1}, nil, y[:3]},
		}
		fs := []struct {
			name     string
			kern, mp func(a, b *ndarray.Array) (*ndarray.Array, error)
		}{{"Perm", Perm, mapPerm}, {"Binom", Binom, mapBinom}, {"Xlog1py", Xlog1py, mapXlog1py}}
		for _, c := range cases {
			a, b := bitsArr(t, c.xa, dt, c.sa...), bitsArr(t, c.yb, dt, c.sb...)
			for _, f := range fs {
				g, ge := f.kern(a, b)
				w, we := f.mp(a, b)
				bitsSame(t, dt+" "+c.name+" "+f.name, g, w, ge, we)
			}
		}
		// mixed dtypes: float32 with float64
		a, b := bitsArr(t, x, "float32", n), bitsArr(t, y, "float64", n)
		for _, f := range fs {
			g, ge := f.kern(a, b)
			w, we := f.mp(a, b)
			bitsSame(t, "mixed "+f.name, g, w, ge, we)
		}
	}
}
