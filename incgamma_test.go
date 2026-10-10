package special

func init() {
	register("gammainc", func(a []float64) float64 { return GammaP(a[0], a[1]) }, map[string]float64{"*": 1e9})
	register("gammaincc", func(a []float64) float64 { return GammaQ(a[0], a[1]) }, map[string]float64{"*": 1e9})
	register("gammaincinv", func(a []float64) float64 { return GammaPInv(a[0], a[1]) }, map[string]float64{"*": 1e9})
	register("gammainccinv", func(a []float64) float64 { return GammaQInv(a[0], a[1]) }, map[string]float64{"*": 1e9})
}
