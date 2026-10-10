package special

func init() {
	register("zeta", func(a []float64) float64 { return Zeta(a[0]) }, map[string]float64{"*": 1e9})
	register("zeta_hurwitz", func(a []float64) float64 { return HurwitzZeta(a[0], a[1]) }, map[string]float64{"*": 1e9})
	register("polygamma", func(a []float64) float64 { return Polygamma(int(a[0]), a[1]) }, map[string]float64{"*": 1e9})
}
