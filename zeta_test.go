package special

func init() {
	register("zeta", func(a []float64) float64 { return Zeta(a[0]) }, nil)
	register("zeta_hurwitz", func(a []float64) float64 { return HurwitzZeta(a[0], a[1]) }, nil)
	register("polygamma", func(a []float64) float64 { return Polygamma(int(a[0]), a[1]) }, nil)
}
