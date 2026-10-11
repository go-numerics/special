package special

// Budgets: the measured maximum per region (arm64 and amd64) times 1.3,
// rounded up to a clean value. math.Inf(1): regions where the root is not
// determined by double-double evaluation of the function (see incbeta_inv.go).
func init() {
	register("gammainc", func(a []float64) float64 { return GammaP(a[0], a[1]) }, map[string]float64{
		"special":                   1,
		"a<1e-3":                    1,
		"a in [1e-3,1)":             1,
		"a in [1,30)":               3,
		"integer, half-integer a":   2,
		"a in [30,1e4), x near a":   1.5,
		"a in [1e4,1e15], x near a": 1,
		"x tiny (P to subnormal)":   1,
		"x large (Q to subnormal)":  1,
		"x huge":                    1,
	})
	register("gammaincc", func(a []float64) float64 { return GammaQ(a[0], a[1]) }, map[string]float64{
		"special":                   1.5,
		"a<1e-3":                    1.5,
		"a in [1e-3,1)":             1.5,
		"a in [1,30)":               2.5,
		"integer, half-integer a":   2,
		"a in [30,1e4), x near a":   1.5,
		"a in [1e4,1e15], x near a": 1,
		"x tiny (P to subnormal)":   1.5,
		"x large (Q to subnormal)":  2,
		"x huge":                    1,
	})
	register("gammaincinv", func(a []float64) float64 { return GammaPInv(a[0], a[1]) }, map[string]float64{
		"special":         1,
		"a<1e-3":          1,
		"a in [1e-3,1)":   1.5,
		"a in [1,100)":    2,
		"a in [100,1e4)":  1,
		"a in [1e4,1e15]": 1,
		"p tiny":          1,
		"p near 1":        1,
	})
	register("gammainccinv", func(a []float64) float64 { return GammaQInv(a[0], a[1]) }, map[string]float64{
		"special":         1,
		"a<1e-3":          1,
		"a in [1e-3,1)":   1,
		"a in [1,100)":    2,
		"a in [100,1e4)":  1,
		"a in [1e4,1e15]": 1,
		"p tiny":          1,
		"p near 1":        1.5,
	})
}
