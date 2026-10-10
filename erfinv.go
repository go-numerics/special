package special

import (
	"math"

	"github.com/go-numerics/special/internal/dd"
)

// erfInvGuess returns an approximation of erfinv(p) for 0 <= p < 1 with
// q = 1 - p supplied exactly by the caller (it carries the information when
// p is near 1). These are the rational approximations of Boost.Math
// erf_inv_imp for 64-bit precision (boost/math/special_functions/detail/
// erf_inv.hpp, John Maddock, Boost Software License 1.0), accurate to about
// 1e-19 relative before float64 rounding; the callers then take one Halley
// step against a double-double residual.
func erfInvGuess(p, q float64) float64 {
	if p <= 0.5 {
		const y = 0.0891314744949340820313
		g := p * (p + 10)
		r := polevl(p, erfInvP1[:]) / polevl(p, erfInvQ1[:])
		return g*y + g*r
	}
	if q >= 0.25 {
		const y = 2.249481201171875
		g := math.Sqrt(-2 * math.Log(q))
		xs := q - 0.25
		r := polevl(xs, erfInvP2[:]) / polevl(xs, erfInvQ2[:])
		return g / (y + r)
	}
	x := math.Sqrt(-math.Log(q))
	var y, xs float64
	var p0, q0 []float64
	switch {
	case x < 3:
		y, xs, p0, q0 = 0.807220458984375, x-1.125, erfInvP3[:], erfInvQ3[:]
	case x < 6:
		y, xs, p0, q0 = 0.93995571136474609375, x-3, erfInvP4[:], erfInvQ4[:]
	case x < 18:
		y, xs, p0, q0 = 0.98362827301025390625, x-6, erfInvP5[:], erfInvQ5[:]
	default:
		// x < 27.3 for every positive float64 q.
		y, xs, p0, q0 = 0.99714565277099609375, x-18, erfInvP6[:], erfInvQ6[:]
	}
	r := polevl(xs, p0) / polevl(xs, q0)
	return y*x + r*x
}

// Coefficients in descending order (Boost lists them ascending).
var (
	erfInvP1 = [...]float64{-0.00538772965071242932965, 0.00822687874676915743155, 0.0219878681111168899165, -0.0365637971411762664006, -0.0126926147662974029034, 0.0334806625409744615033, -0.00836874819741736770379, -0.000508781949658280665617}
	erfInvQ1 = [...]float64{0.000886216390456424707504, -0.00233393759374190016776, 0.0795283687341571680018, -0.0527396382340099713954, -0.71228902341542847553, 0.662328840472002992063, 1.56221558398423026363, -1.56574558234175846809, -0.970005043303290640362, 1.0}
	erfInvP2 = [...]float64{-3.67192254707729348546, 21.1294655448340526258, 17.445385985570866523, -44.6382324441786960818, -18.8510648058714251895, 17.6447298408374015486, 8.37050328343119927838, 0.105264680699391713268, -0.202433508355938759655}
	erfInvQ2 = [...]float64{1.72114765761200282724, -22.6436933413139721736, 10.8268667355460159008, 48.5609213108739935468, -20.1432634680485188801, -28.6608180499800029974, 3.9713437953343869095, 6.24264124854247537712, 1.0}
	erfInvP3 = [...]float64{-0.681149956853776992068e-9, 0.285225331782217055858e-7, -0.679465575181126350155e-6, 0.00214558995388805277169, 0.0290157910005329060432, 0.142869534408157156766, 0.337785538912035898924, 0.387079738972604337464, 0.117030156341995252019, -0.163794047193317060787, -0.131102781679951906451}
	erfInvQ3 = [...]float64{0.01105924229346489121, 0.152264338295331783612, 0.848854343457902036425, 2.59301921623620271374, 4.77846592945843778382, 5.38168345707006855425, 3.46625407242567245975, 1.0}
	erfInvP4 = [...]float64{0.266339227425782031962e-11, -0.230404776911882601748e-9, 0.460469890584317994083e-5, 0.000157544617424960554631, 0.00187123492819559223345, 0.00950804701325919603619, 0.0185573306514231072324, -0.00222426529213447927281, -0.0350353787183177984712}
	erfInvQ4 = [...]float64{0.764675292302794483503e-4, 0.00263861676657015992959, 0.0341589143670947727934, 0.220091105764131249824, 0.762059164553623404043, 1.3653349817554063097, 1.0}
	erfInvP5 = [...]float64{0.99055709973310326855e-16, -0.281128735628831791805e-13, 0.462596163522878599135e-8, 0.449696789927706453732e-6, 0.149624783758342370182e-4, 0.000209386317487588078668, 0.00105628862152492910091, -0.00112951438745580278863, -0.0167431005076633737133}
	erfInvQ5 = [...]float64{0.282243172016108031869e-6, 0.275335474764726041141e-4, 0.000964011807005165528527, 0.0160746087093676504695, 0.138151865749083321638, 0.591429344886417493481, 1.0}
	erfInvP6 = [...]float64{-0.116765012397184275695e-17, 0.145596286718675035587e-11, 0.411632831190944208473e-9, 0.396341011304801168516e-7, 0.162397777342510920873e-5, 0.254723037413027451751e-4, -0.779190719229053954292e-5, -0.0024978212791898131227}
	erfInvQ6 = [...]float64{0.509761276599778486139e-9, 0.144437756628144157666e-6, 0.145007359818232637924e-4, 0.000690538265622684595676, 0.0169410838120975906478, 0.207123112214422517181, 1.0}
)

// erfInvSmall returns erfinv(p) for 0 < p <= 1/2: the Boost guess refined
// by one Halley step on f(x) = erf(x) - p, f”/f' = -2x, with the residual
// erf(x) - p in double-double.
func erfInvSmall(p float64) float64 {
	// float64(...) forces the rounding the residual below is computed
	// for: without it Go may keep the guess unrounded inside a fused
	// multiply-add of the final update, and the step corrects a value that
	// was never returned.
	x := float64(erfInvGuess(p, 1-p))
	f := dd.AddF(erfSmallDD(x), -p).Float()
	d := f / (twoOverSqrtPi.Hi * math.Exp(-x*x)) // f/f'
	return x - d/(1+x*d)
}

// erfcInvLarge returns erfcinv(q) for 0 < q < 1/2 (x > 0.4769): the Boost
// guess refined by one Halley step on f(x) = erfc(x) - q. The relative
// residual r = erfc(x)/q - 1 is formed from erfc's scaled mantissa so it
// neither underflows nor loses accuracy for q down to the smallest
// subnormal, and f/f' = -r q / (2/sqrt(pi) exp(-x²)) =
// -r/(1+r) erfcx(x) sqrt(pi)/2.
func erfcInvLarge(q float64) float64 {
	x := float64(erfInvGuess(1-q, q)) // rounding forced, see erfInvSmall
	var r float64
	var ex float64
	if x >= 0.5 {
		m, k := erfcMant(dd.F(x))
		r = dd.AddF(dd.Div(m, dd.F(math.Ldexp(q, -k))), -1).Float()
		ex = erfcxDD(x).Hi
	} else {
		r = dd.AddF(dd.DivF(erfcDD(dd.F(x)), q), -1).Float()
		ex = Erfcx(x)
	}
	d := -r / (1 + r) * ex * sqrtPiHalf.Hi // f/f'
	// Halley: f''/f' = -2x
	return x - d/(1+x*d)
}

// ErfInv returns the inverse error function: ErfInv(Erf(x)) = x.
//
// Special values: ErfInv(±1) = ±Inf, ErfInv(±0) = ±0; NaN for |y| > 1 or
// NaN.
//
// Algorithm: Boost.Math's rational initial approximations (erf_inv_imp)
// followed by one Halley step whose residual erf(x) - y (or erfc(x) - (1-y)
// for |y| > 1/2, with 1-y exact) is computed in double-double, which takes
// the result to about half an ulp.
func ErfInv(y float64) float64 {
	p := math.Abs(y)
	switch {
	case math.IsNaN(y) || p > 1:
		return math.NaN()
	case p == 1:
		return math.Copysign(math.Inf(1), y)
	case y == 0:
		return y
	case p <= 0.5:
		return math.Copysign(erfInvSmall(p), y)
	}
	return math.Copysign(erfcInvLarge(1-p), y) // 1-p exact for p >= 1/2
}

// ErfcInv returns the inverse complementary error function:
// ErfcInv(Erfc(x)) = x, for 0 <= q <= 2.
//
// Special values: ErfcInv(0) = +Inf, ErfcInv(2) = -Inf, ErfcInv(1) = 0;
// NaN outside [0, 2] or for NaN.
//
// Algorithm: as ErfInv, refining against erfc(x) - q directly, so that
// small q (down to the smallest subnormal) keep full relative accuracy.
func ErfcInv(q float64) float64 {
	switch {
	case math.IsNaN(q) || q < 0 || q > 2:
		return math.NaN()
	case q == 0:
		return math.Inf(1)
	case q == 2:
		return math.Inf(-1)
	case q > 1:
		return -ErfcInv(2 - q) // exact
	case q >= 0.5:
		return erfInvSmall(1 - q) // exact
	}
	return erfcInvLarge(q)
}

// Ndtri returns the inverse of the standard normal cdf: Ndtri(Ndtr(x)) = x,
// for 0 <= p <= 1.
//
// Special values: Ndtri(0) = -Inf, Ndtri(1) = +Inf, Ndtri(1/2) = 0; NaN
// outside [0, 1] or for NaN.
//
// Algorithm: the guess -sqrt(2) erfcinv(2p) from Boost.Math's
// approximations, refined by one Halley step on Φ(x) - p with the relative
// residual from Ndtr's double-double evaluation (so p in the far lower
// tail, down to subnormals, keeps full relative accuracy). For p > 1/2,
// Ndtri(p) = -Ndtri(1-p) with 1-p exact.
func Ndtri(p float64) float64 {
	switch {
	case math.IsNaN(p) || p < 0 || p > 1:
		return math.NaN()
	case p == 0:
		return math.Inf(-1)
	case p == 1:
		return math.Inf(1)
	case p == 0.5:
		return 0
	case p > 0.5:
		return -Ndtri(1 - p) // exact
	}
	q := 2 * p                                      // exact (or exactly representable subnormal doubling)
	x := float64(-math.Sqrt2 * erfInvGuess(1-q, q)) // rounding forced, see erfInvSmall
	z := ndtrZ(x)
	var r, ratio float64
	if z.Hi >= 0.5 {
		m, k := erfcMant(z)
		r = dd.AddF(dd.Div(m, dd.F(math.Ldexp(p, 1-k))), -1).Float()
		ratio = erfcxCorrected(z).Hi * sqrtPiOver2.Hi // Φ(x)/φ(x)
	} else {
		phi := dd.MulF(erfcDD(z), 0.5)
		r = dd.AddF(dd.DivF(phi, p), -1).Float()
		ratio = phi.Hi / (invSqrt2Pi.Hi * math.Exp(-0.5*x*x))
	}
	d := r / (1 + r) * ratio // f/f'
	// Halley: f''/f' = -x
	return x - d/(1+0.5*x*d)
}
