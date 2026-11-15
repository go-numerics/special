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
	// the corners of jn_yn_fast.go: the stated bounds of its recurrences
	// (Jn 4.5 ulp, Yn 6.5), also where they decline and Steed's method answers
	register("jn_edge", func(a []float64) float64 { return Jn(int(a[0]), a[1]) }, map[string]float64{
		"x near n":   4.5,
		"near zeros": 4.5,
		"x near 100": 4.5,
		"large n":    4.5,
		"small x":    4.5,
		"random":     4.5,
	})
	register("yn_edge", func(a []float64) float64 { return Yn(int(a[0]), a[1]) }, map[string]float64{
		"x near n":      6.5,
		"near zeros":    6.5,
		"x near 100":    6.5,
		"large n":       6.5,
		"small x":       6.5,
		"overflow edge": 6.5,
		"random":        6.5,
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
	register("kn", func(a []float64) float64 { return Kn(int(a[0]), a[1]) }, map[string]float64{
		"special":             1,
		"small x":             2.5,
		"moderate":            3,
		"large x":             3,
		"over/underflow edge": 2,
		"large order":         2.5,
		"v ~ x":               2.5,
		"negative n":          4,
	})
	register("kn_edge", func(a []float64) float64 { return Kn(int(a[0]), a[1]) }, map[string]float64{
		"x near 1":         3,
		"x near 512":       1.5,
		"large n, large x": 1,
		"overflow edge":    1.5,
		"n up to 1024":     1.5,
	})
	// the corners of kv_fast.go: Debye (32 <= v <= 1100), the large-argument
	// chain (x >= 24), the overflow and underflow edges, also where the fast
	// path declines and besselIK answers
	register("kv_edge", besselTwo(Kv), map[string]float64{
		"small x":        3,
		"x near v":       3,
		"large v":        3,
		"v above Debye":  3,
		"overflow edge":  3,
		"underflow edge": 3,
		"x near 24":      3,
		"negative v":     3.5, // besselIK at small x: 3.3 ulp (v = -2.5, x = 0.5)
		"near-integer v": 3,
		"random":         3,
		"Debye root":     3,
		// the corners of kv_cf2_fast.go (0 < v < 32, 1.5 <= x < 24): 0.6 ulp
		// where it answers; "tiny x" (x < 1.5) is besselIK, unchanged (2.64 ulp
		// at v = 0.3, x = 0.3); the first and last rows of "x near edge" are
		// below its range (besselIK) or at its top
		"cf2 u near 1/2":  0.6,
		"cf2 u near 0":    0.6,
		"cf2 x near edge": 3,
		"cf2 tiny x":      3,
		"cf2 v near 32":   0.6,
		"cf2 random":      0.6,
		// the corners of kv_temme_fast.go (0 < v < 32, 2^-1000 <= x < 1.5): 0.6
		// ulp where it answers (K_v below 2^1020); the two "edge" regions hold rows
		// on both sides of 2^1020 (above it, and up to MaxFloat64, it declines and
		// besselIK answers: 1 ulp at worst on those rows)
		"temme u near 0":        0.6,
		"temme u near 1/2":      0.6,
		"temme x near edge":     0.6,
		"temme tiny x":          0.6,
		"temme v near 32":       0.6,
		"temme edge 2^1020":     1,
		"temme edge MaxFloat64": 3,
		"temme random":          0.6,
	})
	register("kve_edge", besselTwo(Kve), map[string]float64{
		"small x":        3,
		"x near v":       3,
		"large v":        3,
		"v above Debye":  3,
		"overflow edge":  3,
		"underflow edge": 3,
		"x near 24":      3,
		"negative v":     3,
		"near-integer v": 3,
		"random":         3,
		"Debye root":     3,

		"cf2 u near 1/2":  0.6,
		"cf2 u near 0":    0.6,
		"cf2 x near edge": 3,
		"cf2 tiny x":      3,
		"cf2 v near 32":   0.6,
		"cf2 random":      0.6,

		"temme u near 0":        0.6,
		"temme u near 1/2":      0.6,
		"temme x near edge":     0.6,
		"temme tiny x":          0.6,
		"temme v near 32":       0.6,
		"temme edge 2^1020":     1,
		"temme edge MaxFloat64": 3,
		"temme random":          0.6,
	})
	// the corners of i_fast.go: Debye (32 <= v <= 1100), the large-argument
	// expansion (x >= 24), the recurrence (x < 512), the overflow of e^x and of
	// I_v, also where the fast path declines and besselIK answers (x < 1/8 for
	// integer orders, negative non-integer orders, x >= 512 out of reach)
	register("i_edge", besselTwo(Iv), map[string]float64{
		"small x":                 1.5, // integer order below the I0e table: besselIK, 1.05 ulp
		"x near v":                0.6,
		"large v":                 0.6,
		"v above Debye":           0.6,
		"overflow of e^x":         1.5, // x > 700: besselIK, 1.25 ulp
		"overflow edge":           2,   // Debye declines in the band: besselIK, 1.76 ulp
		"underflow edge":          0.6,
		"x near 24":               0.6,
		"x near 512":              0.6,
		"negative v":              3, // the reflection, besselIK: 2.56 ulp
		"negative near-integer v": 3.5,
		"near-integer v":          0.6,
		"integer v":               0.6,
		"random":                  0.6,
	})
	register("ive_edge", besselTwo(Ive), map[string]float64{
		"small x":                 0.6,
		"x near v":                0.6,
		"large v":                 0.8, // besselIK where Debye declines: 0.71 ulp
		"v above Debye":           0.6,
		"overflow of e^x":         0.8,
		"overflow edge":           0.6,
		"underflow edge":          0.6,
		"x near 24":               0.6,
		"x near 512":              0.9, // x = 512 and beyond: besselIK, 0.79 ulp
		"negative v":              3,
		"negative near-integer v": 3.5,
		"near-integer v":          0.6,
		"integer v":               0.6,
		"random":                  0.6,
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
