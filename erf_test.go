package special

func init() {
	register("erf", func(a []float64) float64 { return Erf(a[0]) }, map[string]float64{"*": 1e9})
	register("erfc", func(a []float64) float64 { return Erfc(a[0]) }, map[string]float64{"*": 1e9})
	register("erfcx", func(a []float64) float64 { return Erfcx(a[0]) }, map[string]float64{"*": 1e9})
	register("erfinv", func(a []float64) float64 { return ErfInv(a[0]) }, map[string]float64{"*": 1e9})
	register("erfcinv", func(a []float64) float64 { return ErfcInv(a[0]) }, map[string]float64{"*": 1e9})
	register("ndtr", func(a []float64) float64 { return Ndtr(a[0]) }, map[string]float64{"*": 1e9})
	register("ndtri", func(a []float64) float64 { return Ndtri(a[0]) }, map[string]float64{"*": 1e9})
	register("log_ndtr", func(a []float64) float64 { return LogNdtr(a[0]) }, map[string]float64{"*": 1e9})
}
