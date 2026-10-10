package special

func init() {
	register("gamma", func(a []float64) float64 { return Gamma(a[0]) }, map[string]float64{"*": 1e9})
	register("loggamma", func(a []float64) float64 { return LogGamma(a[0]) }, map[string]float64{"*": 1e9})
}

func init() {
	register("digamma", func(a []float64) float64 { return Digamma(a[0]) }, map[string]float64{"*": 1e9})
}

func init() {
	register("beta", func(a []float64) float64 { return Beta(a[0], a[1]) }, map[string]float64{"*": 1e9})
	register("logbeta", func(a []float64) float64 { return LogBeta(a[0], a[1]) }, map[string]float64{"*": 1e9})
}
