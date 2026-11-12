package special

import "math"

// Budgets: the measured maximum per region (arm64 and amd64) times 1.3,
// rounded up to a clean value. math.Inf(1): regions where the root is not
// determined by double-double evaluation of the function (see incbeta_inv.go).
func init() {
	register("betainc", func(a []float64) float64 { return BetaInc(a[0], a[1], a[2]) }, map[string]float64{
		"special":                        1.5,
		"a,b<1e-3":                       1.5,
		"a<1e-3<=b":                      1,
		"b<1e-3<=a":                      4,
		"a,b in [1e-3,1)":                3,
		"a,b in [1,100)":                 8,
		"a,b in [100,1e4), x near mean":  15,
		"a,b in [1e4,1e15], x near mean": 10,
		"a small, b large":               8,
		"x tiny (I to subnormal)":        5,
		"x near 1 (I^c to subnormal)":    1,
		// 9 regions around the old prefix threshold, each budget the maximum
		// measured on today's code (docs/accuracy-table.md: 0.98, 2.12, 0.71,
		// 0.54, 1.79, 0.90, 0.74, 1.47, 2.23) times 1.3, rounded up.
		"a,b in [0.01,0.5)":            1.5,
		"a,b in [0.5,3)":               3,
		"a,b in [3,10)":                1,
		"a,b in [10,85]":               1,
		"a,b in [0.01,3), x near 1":    2.5,
		"a,b in [0.5,3), x near 1":     1.5,
		"a,b in [2^-20,85] log, x any": 3,
		"a,b in [5,20)":                1,
		"a,b in [0.5,10), x any":       2,
	})
	register("betaincc", func(a []float64) float64 { return BetaIncC(a[0], a[1], a[2]) }, map[string]float64{
		"special":                        1,
		"a,b<1e-3":                       2,
		"a<1e-3<=b":                      8,
		"b<1e-3<=a":                      1,
		"a,b in [1e-3,1)":                5,
		"a,b in [1,100)":                 6,
		"a,b in [100,1e4), x near mean":  15,
		"a,b in [1e4,1e15], x near mean": 10,
		"a small, b large":               10,
		"x tiny (I to subnormal)":        1,
		"x near 1 (I^c to subnormal)":    4,
		// 9 regions, same rule (measured 0.91, 2.30, 0.74, 0.73, 0.84, 1.74,
		// 0.62, 1.65, 2.07).
		"a,b in [0.01,0.5)":            1.5,
		"a,b in [0.5,3)":               3,
		"a,b in [3,10)":                1,
		"a,b in [10,85]":               1,
		"a,b in [0.01,3), x near 1":    1.5,
		"a,b in [0.5,3), x near 1":     2.5,
		"a,b in [2^-20,85] log, x any": 3,
		"a,b in [5,20)":                1,
		"a,b in [0.5,10), x any":       2.5,
	})
	register("betaincinv", func(a []float64) float64 { return BetaIncInv(a[0], a[1], a[2]) }, map[string]float64{
		"special":                1,
		"a,b<1e-3":               math.Inf(1),
		"a<1e-3<=b or b<1e-3<=a": math.Inf(1),
		"a,b in [1e-3,1)":        20,
		"a,b in [1,100)":         5,
		"a,b in [100,3000)":      1.5,
		"a,b in [3000,1e15]":     1,
		"a small, b large":       3,
		"p tiny":                 3,
		"p near 1":               1,
	})
	register("betainccinv", func(a []float64) float64 { return BetaIncCInv(a[0], a[1], a[2]) }, map[string]float64{
		"special":                1,
		"a,b<1e-3":               math.Inf(1),
		"a<1e-3<=b or b<1e-3<=a": math.Inf(1),
		"a,b in [1e-3,1)":        80,
		"a,b in [1,100)":         5,
		"a,b in [100,3000)":      1.5,
		"a,b in [3000,1e15]":     1,
		"a small, b large":       6,
		"p tiny":                 1,
		"p near 1":               2,
	})
}
