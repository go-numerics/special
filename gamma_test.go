package special

func init() {
	register("gamma", func(a []float64) float64 { return Gamma(a[0]) }, map[string]float64{"*": 1e9})
	register("loggamma", func(a []float64) float64 { return LogGamma(a[0]) }, map[string]float64{"*": 1e9})
}
