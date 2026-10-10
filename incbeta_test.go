package special

func init() {
	register("betainc", func(a []float64) float64 { return BetaInc(a[0], a[1], a[2]) }, map[string]float64{"*": 1e9})
	register("betaincc", func(a []float64) float64 { return BetaIncC(a[0], a[1], a[2]) }, map[string]float64{"*": 1e9})
}
