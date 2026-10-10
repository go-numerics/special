package special

func init() {
	register("factorial", func(a []float64) float64 { return Factorial(a[0]) }, nil)
	register("factorial2", func(a []float64) float64 { return Factorial2(int(a[0])) }, nil)
	register("log_factorial", func(a []float64) float64 { return LogFactorial(a[0]) }, nil)
	register("comb", func(a []float64) float64 { return Comb(int(a[0]), int(a[1])) }, nil)
	register("perm", func(a []float64) float64 { return Perm(int(a[0]), int(a[1])) }, nil)
	register("binom", func(a []float64) float64 { return Binom(a[0], a[1]) }, nil)
	register("expit", func(a []float64) float64 { return Expit(a[0]) }, nil)
	register("log_expit", func(a []float64) float64 { return LogExpit(a[0]) }, nil)
	register("log1pexp", func(a []float64) float64 { return Log1pExp(a[0]) }, nil)
	register("logit", func(a []float64) float64 { return Logit(a[0]) }, nil)
	register("xlogy", func(a []float64) float64 { return Xlogy(a[0], a[1]) }, nil)
	register("xlog1py", func(a []float64) float64 { return Xlog1py(a[0], a[1]) }, nil)
	register("logsumexp4", func(a []float64) float64 { return LogSumExp(a) }, nil)
	register("softmax4", func(a []float64) float64 { return Softmax(nil, a)[0] }, nil)
	register("log_softmax4", func(a []float64) float64 { return LogSoftmax(nil, a)[0] }, nil)
	register("ellipk", func(a []float64) float64 { return EllipK(a[0]) }, nil)
	register("ellipkm1", func(a []float64) float64 { return EllipKm1(a[0]) }, nil)
	register("ellipe", func(a []float64) float64 { return EllipE(a[0]) }, nil)
}
