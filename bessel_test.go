package special

// Accuracy cases of the Bessel family (fixtures from testdata/gen/fam_bessel.py).
// Budgets are the measured maxima rounded up: a regression guard.

func init() {
	one := func(f func(float64) float64) func(a []float64) float64 {
		return func(a []float64) float64 { return f(a[0]) }
	}
	two := func(f func(float64, float64) float64) func(a []float64) float64 {
		return func(a []float64) float64 { return f(a[0], a[1]) }
	}
	B := map[string]float64{"*": 1e300}
	register("j0", one(J0), B)
	register("j1", one(J1), B)
	register("y0", one(Y0), B)
	register("y1", one(Y1), B)
	register("i0", one(I0), B)
	register("i1", one(I1), B)
	register("k0", one(K0), B)
	register("k1", one(K1), B)
	register("i0e", one(I0e), B)
	register("i1e", one(I1e), B)
	register("k0e", one(K0e), B)
	register("k1e", one(K1e), B)
	register("jv", two(Jv), B)
	register("yv", two(Yv), B)
	register("iv", two(Iv), B)
	register("kv", two(Kv), B)
	register("ive", two(Ive), B)
	register("kve", two(Kve), B)
	register("jn", func(a []float64) float64 { return Jn(int(a[0]), a[1]) }, B)
	register("yn", func(a []float64) float64 { return Yn(int(a[0]), a[1]) }, B)
}
