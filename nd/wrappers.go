package nd

import (
	"math"

	"github.com/go-ndarray/ndarray"
	"github.com/go-numerics/special"
)

// Every function below takes its cost class from the costs table (nd.go), keyed
// by its own name.

// Gamma applies special.Gamma element-wise.
func Gamma(x *ndarray.Array) (*ndarray.Array, error) {
	return apply1(costs["Gamma"], special.GammaTo, x)
}

// LogGamma applies special.LogGamma element-wise.
func LogGamma(x *ndarray.Array) (*ndarray.Array, error) {
	return apply1(costs["LogGamma"], special.LogGammaTo, x)
}

// GammaSign applies special.GammaSign element-wise.
func GammaSign(x *ndarray.Array) (*ndarray.Array, error) {
	return apply1(costs["GammaSign"], special.GammaSignTo, x)
}

// Digamma applies special.Digamma element-wise.
func Digamma(x *ndarray.Array) (*ndarray.Array, error) {
	return apply1(costs["Digamma"], special.DigammaTo, x)
}

// Erf applies special.Erf element-wise.
func Erf(x *ndarray.Array) (*ndarray.Array, error) {
	return apply1(costs["Erf"], special.ErfTo, x)
}

// Erfc applies special.Erfc element-wise.
func Erfc(x *ndarray.Array) (*ndarray.Array, error) {
	return apply1(costs["Erfc"], special.ErfcTo, x)
}

// Erfcx applies special.Erfcx element-wise.
func Erfcx(x *ndarray.Array) (*ndarray.Array, error) {
	return apply1(costs["Erfcx"], special.ErfcxTo, x)
}

// ErfInv applies special.ErfInv element-wise.
func ErfInv(x *ndarray.Array) (*ndarray.Array, error) {
	return apply1(costs["ErfInv"], special.ErfInvTo, x)
}

// ErfcInv applies special.ErfcInv element-wise.
func ErfcInv(x *ndarray.Array) (*ndarray.Array, error) {
	return apply1(costs["ErfcInv"], special.ErfcInvTo, x)
}

// Ndtr applies special.Ndtr element-wise.
func Ndtr(x *ndarray.Array) (*ndarray.Array, error) {
	return apply1(costs["Ndtr"], special.NdtrTo, x)
}

// Ndtri applies special.Ndtri element-wise.
func Ndtri(x *ndarray.Array) (*ndarray.Array, error) {
	return apply1(costs["Ndtri"], special.NdtriTo, x)
}

// LogNdtr applies special.LogNdtr element-wise.
func LogNdtr(x *ndarray.Array) (*ndarray.Array, error) {
	return apply1(costs["LogNdtr"], special.LogNdtrTo, x)
}

// Factorial applies special.Factorial element-wise.
func Factorial(x *ndarray.Array) (*ndarray.Array, error) {
	return map1(costs["Factorial"], special.Factorial, x)
}

// LogFactorial applies special.LogFactorial element-wise.
func LogFactorial(x *ndarray.Array) (*ndarray.Array, error) {
	return map1(costs["LogFactorial"], special.LogFactorial, x)
}

// Expit applies special.Expit element-wise.
func Expit(x *ndarray.Array) (*ndarray.Array, error) {
	return apply1(costs["Expit"], special.ExpitTo, x)
}

// LogExpit applies special.LogExpit element-wise.
func LogExpit(x *ndarray.Array) (*ndarray.Array, error) {
	return map1(costs["LogExpit"], special.LogExpit, x)
}

// Log1pExp applies special.Log1pExp element-wise.
func Log1pExp(x *ndarray.Array) (*ndarray.Array, error) {
	return apply1(costs["Log1pExp"], special.Log1pExpTo, x)
}

// Logit applies special.Logit element-wise.
func Logit(x *ndarray.Array) (*ndarray.Array, error) {
	return apply1(costs["Logit"], special.LogitTo, x)
}

// Zeta applies special.Zeta element-wise.
func Zeta(x *ndarray.Array) (*ndarray.Array, error) {
	return apply1(costs["Zeta"], special.ZetaTo, x)
}

// EllipK applies special.EllipK element-wise.
func EllipK(x *ndarray.Array) (*ndarray.Array, error) {
	return map1(costs["EllipK"], special.EllipK, x)
}

// EllipKm1 applies special.EllipKm1 element-wise.
func EllipKm1(x *ndarray.Array) (*ndarray.Array, error) {
	return map1(costs["EllipKm1"], special.EllipKm1, x)
}

// EllipE applies special.EllipE element-wise.
func EllipE(x *ndarray.Array) (*ndarray.Array, error) {
	return map1(costs["EllipE"], special.EllipE, x)
}

// Beta applies special.Beta element-wise to the broadcast arguments.
func Beta(a, b *ndarray.Array) (*ndarray.Array, error) {
	return apply2(costs["Beta"], special.BetaTo, special.Beta, a, b)
}

// LogBeta applies special.LogBeta element-wise to the broadcast arguments.
func LogBeta(a, b *ndarray.Array) (*ndarray.Array, error) {
	return apply2(costs["LogBeta"], special.LogBetaTo, special.LogBeta, a, b)
}

// Binom applies special.Binom element-wise to the broadcast arguments.
func Binom(a, b *ndarray.Array) (*ndarray.Array, error) {
	return apply2(costs["Binom"], special.BinomTo, special.Binom, a, b)
}

// Xlogy applies special.Xlogy element-wise to the broadcast arguments.
func Xlogy(a, b *ndarray.Array) (*ndarray.Array, error) {
	return apply2(costs["Xlogy"], special.XlogyTo, special.Xlogy, a, b)
}

// Xlog1py applies special.Xlog1py element-wise to the broadcast arguments.
func Xlog1py(a, b *ndarray.Array) (*ndarray.Array, error) {
	return apply2(costs["Xlog1py"], special.Xlog1pyTo, special.Xlog1py, a, b)
}

// HurwitzZeta applies special.HurwitzZeta element-wise to the broadcast arguments.
func HurwitzZeta(a, b *ndarray.Array) (*ndarray.Array, error) {
	return apply2(costs["HurwitzZeta"], special.HurwitzZetaTo, special.HurwitzZeta, a, b)
}

// Polygamma applies special.Polygamma element-wise to the broadcast arguments; an
// n that is not an integer gives NaN.
func Polygamma(n, x *ndarray.Array) (*ndarray.Array, error) {
	if n.Size() == 1 && len(n.Shape()) <= len(x.Shape()) && (x.DType() != ndarray.Float32 || n.DType() == ndarray.Float32) {
		// A single order 1 (the shape and dtype of the result are x's):
		// ψ' through the generated loop.
		if v, _, err := operand(n); err == nil && v[0] == 1 {
			return apply1(costs["Polygamma"], special.TrigammaTo, x)
		}
	}
	return map2(costs["Polygamma"], func(n, x float64) float64 {
		if n != math.Trunc(n) || math.Abs(n) > math.MaxInt32 {
			return math.NaN()
		}
		return special.Polygamma(int(n), x)
	}, n, x)
}

// Comb applies special.Comb element-wise to the broadcast arguments;
// arguments that are not integers give NaN.
func Comb(n, k *ndarray.Array) (*ndarray.Array, error) {
	return map2(costs["Comb"], func(n, k float64) float64 {
		if !isInt(n) || !isInt(k) {
			return math.NaN()
		}
		return special.Comb(int64(n), int64(k))
	}, n, k)
}

// Perm applies special.Perm element-wise to the broadcast arguments;
// arguments that are not integers give NaN.
func Perm(n, k *ndarray.Array) (*ndarray.Array, error) {
	return apply2(costs["Perm"], special.PermTo, func(n, k float64) float64 {
		if !isInt(n) || !isInt(k) {
			return math.NaN()
		}
		return special.Perm(int64(n), int64(k))
	}, n, k)
}

// Factorial2 applies special.Factorial2 element-wise; non-integers give NaN.
func Factorial2(n *ndarray.Array) (*ndarray.Array, error) {
	return apply1(costs["Factorial2"], special.Factorial2To, n)
}

func isInt(x float64) bool { return x == math.Trunc(x) && math.Abs(x) <= math.MaxInt32 }

// GammaP applies special.GammaP element-wise to the broadcast arguments.
func GammaP(a, x *ndarray.Array) (*ndarray.Array, error) {
	return map2(costs["GammaP"], special.GammaP, a, x)
}

// GammaQ applies special.GammaQ element-wise to the broadcast arguments.
func GammaQ(a, x *ndarray.Array) (*ndarray.Array, error) {
	return map2(costs["GammaQ"], special.GammaQ, a, x)
}

// GammaPInv applies special.GammaPInv element-wise to the broadcast arguments.
func GammaPInv(a, p *ndarray.Array) (*ndarray.Array, error) {
	return map2(costs["GammaPInv"], special.GammaPInv, a, p)
}

// GammaQInv applies special.GammaQInv element-wise to the broadcast arguments.
func GammaQInv(a, q *ndarray.Array) (*ndarray.Array, error) {
	return map2(costs["GammaQInv"], special.GammaQInv, a, q)
}

// BetaInc applies special.BetaInc element-wise to the broadcast arguments.
func BetaInc(a, b, x *ndarray.Array) (*ndarray.Array, error) {
	return map3(costs["BetaInc"], special.BetaInc, a, b, x)
}

// BetaIncC applies special.BetaIncC element-wise to the broadcast arguments.
func BetaIncC(a, b, x *ndarray.Array) (*ndarray.Array, error) {
	return map3(costs["BetaIncC"], special.BetaIncC, a, b, x)
}

// Bessel functions.
// J0 applies special.J0 element-wise.
func J0(x *ndarray.Array) (*ndarray.Array, error) { return map1(costs["J0"], special.J0, x) }

// J1 applies special.J1 element-wise.
func J1(x *ndarray.Array) (*ndarray.Array, error) { return map1(costs["J1"], special.J1, x) }

// Y0 applies special.Y0 element-wise.
func Y0(x *ndarray.Array) (*ndarray.Array, error) { return map1(costs["Y0"], special.Y0, x) }

// Y1 applies special.Y1 element-wise.
func Y1(x *ndarray.Array) (*ndarray.Array, error) { return map1(costs["Y1"], special.Y1, x) }

// I0 applies special.I0 element-wise.
func I0(x *ndarray.Array) (*ndarray.Array, error) { return map1(costs["I0"], special.I0, x) }

// I1 applies special.I1 element-wise.
func I1(x *ndarray.Array) (*ndarray.Array, error) { return map1(costs["I1"], special.I1, x) }

// I0e applies special.I0e element-wise.
func I0e(x *ndarray.Array) (*ndarray.Array, error) { return map1(costs["I0e"], special.I0e, x) }

// I1e applies special.I1e element-wise.
func I1e(x *ndarray.Array) (*ndarray.Array, error) { return map1(costs["I1e"], special.I1e, x) }

// K0 applies special.K0 element-wise.
func K0(x *ndarray.Array) (*ndarray.Array, error) { return map1(costs["K0"], special.K0, x) }

// K1 applies special.K1 element-wise.
func K1(x *ndarray.Array) (*ndarray.Array, error) { return map1(costs["K1"], special.K1, x) }

// K0e applies special.K0e element-wise.
func K0e(x *ndarray.Array) (*ndarray.Array, error) { return map1(costs["K0e"], special.K0e, x) }

// K1e applies special.K1e element-wise.
func K1e(x *ndarray.Array) (*ndarray.Array, error) { return map1(costs["K1e"], special.K1e, x) }

// Jv applies special.Jv element-wise to the broadcast arguments.
func Jv(v, x *ndarray.Array) (*ndarray.Array, error) { return map2(costs["Jv"], special.Jv, v, x) }

// Yv applies special.Yv element-wise to the broadcast arguments.
func Yv(v, x *ndarray.Array) (*ndarray.Array, error) { return map2(costs["Yv"], special.Yv, v, x) }

// Iv applies special.Iv element-wise to the broadcast arguments.
func Iv(v, x *ndarray.Array) (*ndarray.Array, error) { return map2(costs["Iv"], special.Iv, v, x) }

// Ive applies special.Ive element-wise to the broadcast arguments.
func Ive(v, x *ndarray.Array) (*ndarray.Array, error) { return map2(costs["Ive"], special.Ive, v, x) }

// Kv applies special.Kv element-wise to the broadcast arguments.
func Kv(v, x *ndarray.Array) (*ndarray.Array, error) { return map2(costs["Kv"], special.Kv, v, x) }

// Kve applies special.Kve element-wise to the broadcast arguments.
func Kve(v, x *ndarray.Array) (*ndarray.Array, error) { return map2(costs["Kve"], special.Kve, v, x) }

// Jn applies special.Jn element-wise; an order that is not an integer gives NaN.
func Jn(n, x *ndarray.Array) (*ndarray.Array, error) {
	return map2(costs["Jn"], func(n, x float64) float64 {
		if !isInt(n) {
			return math.NaN()
		}
		return special.Jn(int(n), x)
	}, n, x)
}

// Yn applies special.Yn element-wise; an order that is not an integer gives NaN.
func Yn(n, x *ndarray.Array) (*ndarray.Array, error) {
	return map2(costs["Yn"], func(n, x float64) float64 {
		if !isInt(n) {
			return math.NaN()
		}
		return special.Yn(int(n), x)
	}, n, x)
}

// In applies special.In element-wise; an order that is not an integer gives NaN.
func In(n, x *ndarray.Array) (*ndarray.Array, error) {
	return map2(costs["In"], func(n, x float64) float64 {
		if !isInt(n) {
			return math.NaN()
		}
		return special.In(int(n), x)
	}, n, x)
}

// Kn applies special.Kn element-wise; an order that is not an integer gives NaN.
func Kn(n, x *ndarray.Array) (*ndarray.Array, error) {
	return map2(costs["Kn"], func(n, x float64) float64 {
		if !isInt(n) {
			return math.NaN()
		}
		return special.Kn(int(n), x)
	}, n, x)
}

// BetaIncInv applies special.BetaIncInv element-wise to the broadcast arguments.
func BetaIncInv(a, b, p *ndarray.Array) (*ndarray.Array, error) {
	return map3(costs["BetaIncInv"], special.BetaIncInv, a, b, p)
}

// BetaIncCInv applies special.BetaIncCInv element-wise to the broadcast arguments.
func BetaIncCInv(a, b, q *ndarray.Array) (*ndarray.Array, error) {
	return map3(costs["BetaIncCInv"], special.BetaIncCInv, a, b, q)
}
