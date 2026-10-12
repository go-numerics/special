package special

// Accuracy cases of the Bessel family (fixtures: testdata/gen/fam_bessel.py).
// Budgets are the measured maxima over arm64 and amd64, times about 1.3,
// rounded up to a clean number: a regression guard.

func init() {
	register("j0", besselOne(J0), map[string]float64{
		"special":    1,
		"small x":    1,
		"moderate":   1.5,
		"large x":    2.5,
		"huge x":     2.5,
		"near zeros": 1,
		"negative x": 2.5,
	})
	register("j1", besselOne(J1), map[string]float64{
		"special":    1.5,
		"small x":    1,
		"moderate":   2,
		"large x":    3,
		"huge x":     3,
		"near zeros": 1,
		"negative x": 3,
	})
	register("y0", besselOne(Y0), map[string]float64{
		"special":    1,
		"small x":    1,
		"moderate":   1.5,
		"large x":    2.5,
		"huge x":     2.5,
		"near zeros": 1,
	})
	register("y1", besselOne(Y1), map[string]float64{
		"special":    1,
		"small x":    1,
		"moderate":   1.5,
		"large x":    3,
		"huge x":     3,
		"near zeros": 1,
	})
	register("i0", besselOne(I0), map[string]float64{
		"special":             1.5,
		"small x":             1,
		"moderate":            4,
		"large x":             2.5,
		"over/underflow edge": 2,
		"huge x":              1,
		"negative x":          2.5,
	})
	register("i1", besselOne(I1), map[string]float64{
		"special":             1,
		"small x":             1,
		"moderate":            5,
		"large x":             2.5,
		"over/underflow edge": 2,
		"huge x":              1,
		"negative x":          5,
	})
	register("k0", besselOne(K0), map[string]float64{
		"special":             1,
		"small x":             1.5,
		"moderate":            4,
		"large x":             2,
		"over/underflow edge": 3,
		"huge x":              1,
	})
	register("k1", besselOne(K1), map[string]float64{
		"special":             2,
		"small x":             1.5,
		"moderate":            4,
		"large x":             2,
		"over/underflow edge": 2,
		"huge x":              1,
	})
	register("i0e", besselOne(I0e), map[string]float64{
		"special":    1.5,
		"small x":    2,
		"moderate":   3,
		"large x":    1,
		"huge x":     1,
		"negative x": 3,
	})
	register("i1e", besselOne(I1e), map[string]float64{
		"special":    1.5,
		"small x":    1.5,
		"moderate":   4,
		"large x":    1,
		"huge x":     1,
		"negative x": 4,
	})
	register("k0e", besselOne(K0e), map[string]float64{
		"special":  1,
		"small x":  3,
		"moderate": 2.5,
		"large x":  1,
		"huge x":   1,
	})
	register("k1e", besselOne(K1e), map[string]float64{
		"special":  1.5,
		"small x":  2,
		"moderate": 4,
		"large x":  1,
		"huge x":   1,
	})
	register("jv", besselTwo(Jv), map[string]float64{
		"special":    2.5,
		"small x":    1,
		"moderate":   2,
		"large x":    2.5,
		"huge x":     3,
		"near zeros": 3,
		// large order and negative v: loong64, ppc64le and s390x measure 2.01
		// and 13.2 ulp, where amd64 and arm64 measure 1.35 and 7.22: Go fuses
		// a*b+c differently per architecture, and the reflection formula for
		// negative v cancels. SciPy is at 6.8e13 and 1572 ulp on these points.
		"large order": 3,
		"v ~ x":       1,
		"negative v":  20,
		"integer v":   2,
	})
	register("yv", besselTwo(Yv), map[string]float64{
		"special":     2,
		"small x":     4,
		"moderate":    2.5,
		"large x":     2.5,
		"huge x":      3,
		"near zeros":  2.5,
		"large order": 2.5,
		"v ~ x":       1,
		"negative v":  6,
		"integer v":   4,
	})
	register("jn", func(a []float64) float64 { return Jn(int(a[0]), a[1]) }, map[string]float64{
		"special":     2.5,
		"small x":     1.5,
		"moderate":    2,
		"large x":     2.5,
		"huge x":      3,
		"near zeros":  8,
		"negative x":  2,
		"large order": 2.5,
		"v ~ x":       1,
		"negative n":  2,
	})
	register("yn", func(a []float64) float64 { return Yn(int(a[0]), a[1]) }, map[string]float64{
		"special":     2,
		"small x":     2.5,
		"moderate":    4,
		"large x":     2.5,
		"huge x":      2.5,
		"near zeros":  10,
		"large order": 2.5,
		"v ~ x":       1,
		"negative n":  5,
	})
	register("iv", besselTwo(Iv), map[string]float64{
		"special":             3,
		"small x":             1.5,
		"moderate":            3,
		"large x":             4,
		"over/underflow edge": 2.5,
		"large order":         3,
		"v ~ x":               3,
		"negative v":          4,
		"integer v":           3,
	})
	register("kv", besselTwo(Kv), map[string]float64{
		"special":             1.5,
		"small x":             2.5,
		"moderate":            3,
		"large x":             2.5,
		"over/underflow edge": 2,
		"large order":         2.5,
		"v ~ x":               2.5,
		"negative v":          5,
		"integer v":           4,
	})
	register("ive", besselTwo(Ive), map[string]float64{
		"special":             6,
		"small x":             1.5,
		"moderate":            3,
		"large x":             2.5,
		"over/underflow edge": 1,
		"huge x":              1,
		"large order":         2,
		"v ~ x":               1.5,
		"negative v":          20,
		"integer v":           2.5,
	})
	register("kve", besselTwo(Kve), map[string]float64{
		"special":             1.5,
		"small x":             2,
		"moderate":            3,
		"large x":             2,
		"over/underflow edge": 1,
		"huge x":              1,
		"large order":         1.5,
		"v ~ x":               1.5,
		"negative v":          6,
		"integer v":           4,
	})
	register("jn_zeros", func(a []float64) float64 { z := JnZeros(int(a[0]), int(a[1])); return z[len(z)-1] }, map[string]float64{
		"n <= 10, k <= 50": 1,
		"n >= 20, k <= 20": 1,
	})
}

func besselOne(f func(float64) float64) func(a []float64) float64 {
	return func(a []float64) float64 { return f(a[0]) }
}

func besselTwo(f func(float64, float64) float64) func(a []float64) float64 {
	return func(a []float64) float64 { return f(a[0], a[1]) }
}
