package special

import "math"

// Budgets: the measured maximum per region (arm64 and amd64) times 1.3,
// rounded up to a clean value.
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
	})
	register("betaincinv", func(a []float64) float64 { return BetaIncInv(a[0], a[1], a[2]) }, map[string]float64{
		"special":                1,
		"a,b<1e-3":               1.5e+09,
		"a<1e-3<=b or b<1e-3<=a": 1e+16,
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
		"a<1e-3<=b or b<1e-3<=a": 1.5e+16,
		"a,b in [1e-3,1)":        80,
		"a,b in [1,100)":         5,
		"a,b in [100,3000)":      1.5,
		"a,b in [3000,1e15]":     1,
		"a small, b large":       6,
		"p tiny":                 1,
		"p near 1":               2,
	})
}
