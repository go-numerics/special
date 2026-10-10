package special

func init() {
	register("erf", func(a []float64) float64 { return Erf(a[0]) }, nil)
	register("erfc", func(a []float64) float64 { return Erfc(a[0]) }, nil)
	register("erfcx", func(a []float64) float64 { return Erfcx(a[0]) }, nil)
	register("erfinv", func(a []float64) float64 { return ErfInv(a[0]) }, nil)
	register("erfcinv", func(a []float64) float64 { return ErfcInv(a[0]) }, nil)
	register("ndtr", func(a []float64) float64 { return Ndtr(a[0]) }, nil)
	register("ndtri", func(a []float64) float64 { return Ndtri(a[0]) }, nil)
	register("log_ndtr", func(a []float64) float64 { return LogNdtr(a[0]) }, nil)
}
