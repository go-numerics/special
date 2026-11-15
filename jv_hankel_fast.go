package special

import (
	"math"
	"math/big"

	"github.com/go-numerics/special/dd"
)

// A certified fast path of Jv and Yv for x > 20 (besselJYPos, besselJYRefl):
// Hankel's expansion in its P, Q form (DLMF 10.17.3, 10.17.4),
//
//	J_v(x) = a (P cos w - Q sin w),   Y_v(x) = a (P sin w + Q cos w),
//	a = sqrt(2/(pi x)),  w = x - (v/2 + 1/4) pi,
//	P = sum (-1)^k a_2k(v)/x^2k,  Q = sum (-1)^k a_2k+1(v)/x^(2k+1),
//	a_k(v) = (4v^2-1)(4v^2-9)...(4v^2-(2k-1)^2) / (k! 8^k)      (DLMF 10.17.1),
//
// valid for either sign of v (a_k depends on v^2 only, and H1_-v = e^(i v pi)
// H1_v turns w into x + v pi/2 - pi/4 for -v, which is the same formula with
// the signed v). It replaces besselAsymJY's modulus-phase form where that is
// the dearer: there the series of M^2 and of the phase cost O(K^2) (the
// convolution that inverts 1 + S) and are summed down to 2^-110; here the
// terms u_k = a_k/x^k come from one multiplication each, O(K).
//
// What it relies on, and what it proves.
//
// Truncation (rigorous). DLMF 10.17(iii) (opened 2026-11-15): for v >= 0 and
// x > 0 the remainder of the P sum cut after l terms, and that of the Q sum,
// do not exceed the first neglected term in absolute value provided
// l >= max(v/2 - 1/4, 1) (P), l >= max(v/2 - 3/4, 1) (Q). The path takes
// l >= max(ceil(|v|/2), 2), so both hold, and bounds the truncation by
// |u_2l| + |u_2l+1|. It stops at the first l where that pair is below 2^-64 or
// where it stops decreasing; the acceptance test below then decides.
//
// Rounding. u_1 and u_2 (the only terms that matter at 2^-56: u_1 is up to
// 0.3 for v = 3, x = 20) are double-double; u_k, k >= 3, are float64 products,
// each u_k with a relative error below (6k + 2) 2^-53 (a chain of k factors,
// each of the three roundings of a factor, and 1/(8x), 1/k rounded once),
// summed in float64 (additions of the order of 2^-53 of the partial sums,
// both inside hankelTermErr = (k+2) 2^-50 |u_k|, which is above that). The
// angle is reduced exactly (besselReduce, Payne-Hanek) and the sine and
// cosine of the reduced angle are formed as hi + lo by hankelSinCos, whose
// error is below hankelTrigErr (TestHankelSinCosOracle holds it against
// math/big on 10^6 angles). The products P c, Q s and their sum are
// double-double (2^-98 of the larger product), and the result is a.hi F.hi
// with the low parts folded in before ONE rounding.
//
// Contract. With E the sum of those bounds (in units of a) the path answers
// only if E <= 2^-56 |F|: then the pair it rounds is within E |a F| of the
// exact value, at most 1/8 ulp for |F| < 2^53 ulp(F) (always), so the result
// is within 0.5 + 0.125 = 0.63 ulp. Next to a zero of J or Y |F| is small and
// the test fails (the path declines; there is no relative bound there, and
// the absolute bound is the amplitude's: E a). A declined call is answered by
// the code that was there, bit for bit.
const (
	hankelMinX     = 20.0
	hankelMaxX     = 0x1p500
	hankelMaxOrder = 40.0
	hankelMaxL     = 40
	hankelTol      = 0x1p-64
	hankelAccept   = 0x1p-56
	hankelTrigErr  = 0x1p-59
	hankelDDErr    = 0x1p-98

	// hankelDDFloor: the terms below it are carried in float64;
	// hankelTermDDErr bounds the relative error of a double-double term and of
	// its addition to the sum (k <= 2 hankelMaxL + 1 steps of three products
	// and a sum, each below 2^-103).
	hankelDDFloor   = 0x1p-18
	hankelTermDDErr = 0x1p-94
)

// hankelInvK8[k] = 1/(8k) as a double-double.
var hankelInvK8 [2*hankelMaxL + 4]dd.DD

// hankelNextDD returns u_k = u_{k-1} (4v^2 - (2k-1)^2)/(8 k x) in
// double-double: (2v - o)(2v + o) from two TwoSums and a TwoProd (no
// cancellation: each factor is rounded once, and its error is kept), then
// three double-double products.
func hankelNextDD(u dd.DD, k int, two float64, invx dd.DD) dd.DD {
	o := float64(2*k - 1)
	ah, al := dd.TwoSum(two, -o)
	bh, bl := dd.TwoSum(two, o)
	p, pe := dd.TwoProd(ah, bh)
	num := dd.DD{Hi: p, Lo: pe + float64(ah*bl) + float64(al*bh)}
	return dd.Mul(dd.Mul(dd.Mul(u, num), invx), hankelInvK8[k])
}

// hankelInvK[k] = 1/k rounded; hankelSinC, hankelCosC: the Taylor coefficients
// of sin z beyond z^3 (index i: (-1)^i / (2i+5)!) and of cos z beyond z^4
// (index j: (-1)^j / (2j+6)!), rounded; the double-double constants -1/6 and
// 1/24.
var (
	hankelInvK         [2*hankelMaxL + 4]float64
	hankelSinC         [9]float64
	hankelCosC         [8]float64
	hankelC3, hankelC4 dd.DD
)

func init() {
	for k := range hankelInvK {
		if k > 0 {
			hankelInvK[k] = 1 / float64(k)
		}
	}
	inv := func(n int) *big.Rat {
		f := big.NewInt(1)
		for i := 2; i <= n; i++ {
			f.Mul(f, big.NewInt(int64(i)))
		}
		return new(big.Rat).SetFrac(big.NewInt(1), f)
	}
	signed := func(i int, r *big.Rat) *big.Rat {
		if i%2 == 1 {
			return new(big.Rat).Neg(r)
		}
		return r
	}
	for i := range hankelSinC {
		hankelSinC[i], _ = signed(i, inv(2*i+5)).Float64()
	}
	for j := range hankelCosC {
		hankelCosC[j], _ = signed(j+1, inv(2*j+6)).Float64()
	}
	pair := func(r *big.Rat) dd.DD {
		hi, _ := r.Float64()
		lo, _ := new(big.Rat).Sub(r, new(big.Rat).SetFloat64(hi)).Float64()
		return dd.DD{Hi: hi, Lo: lo}
	}
	for k := range hankelInvK8 {
		if k > 0 {
			hankelInvK8[k] = pair(new(big.Rat).SetFrac64(1, int64(8*k)))
		}
	}
	hankelC3 = pair(new(big.Rat).Neg(inv(3)))
	hankelC4 = pair(inv(4))
}

// hankelSinCos returns sin z and cos z as hi + lo for a double-double z with
// |z| <= pi/4 + 2^-20, each within hankelTrigErr of the true value:
//
//	sin z = zh + (-zh^3/6 in double-double) + zh^5 P(zh^2) + zl cos zh,
//	cos z = 1 - zh^2/2 (exact) + zh^4/24 (double-double) + zh^6 Q(zh^2) - zl sin zh,
//
// the Taylor series to z^21 and z^20 (the next terms are below 2^-70).
func hankelSinCos(z dd.DD) (sh, sl, ch, cl float64) {
	zh, zl := z.Hi, z.Lo
	zz, ze := dd.TwoProd(zh, zh)
	zzl := ze // zh*zh = zz + zze exactly; zl enters through the derivative terms only
	// sin
	p, pe := dd.TwoProd(zz, zh)
	z3l := pe + float64(zzl*zh)
	t, te := dd.TwoProd(p, hankelC3.Hi)
	tl := te + float64(p*hankelC3.Lo) + float64(z3l*hankelC3.Hi)
	r := hankelSinC[len(hankelSinC)-1]
	for i := len(hankelSinC) - 2; i >= 0; i-- {
		r = math.FMA(r, zz, hankelSinC[i])
	}
	tail := float64(float64(float64(zh*zz)*zz) * r)
	lin := float64(zl * (1 - float64(zz*(0.5-float64(zz/24)))))
	s, se := dd.TwoSum(zh, t)
	sh, sl = dd.TwoSum(s, se+tl+tail+lin)
	// cos
	h2 := float64(-0.5 * zz)
	c, ce := dd.TwoSum(1, h2)
	q4, q4e := dd.TwoProd(zz, zz)
	q4l := q4e + float64(2*float64(zz*zzl))
	u, ue := dd.TwoProd(q4, hankelC4.Hi)
	ul := ue + float64(q4*hankelC4.Lo) + float64(q4l*hankelC4.Hi)
	rc := hankelCosC[len(hankelCosC)-1]
	for j := len(hankelCosC) - 2; j >= 0; j-- {
		rc = math.FMA(rc, zz, hankelCosC[j])
	}
	ctail := float64(float64(float64(zz*zz)*zz) * rc)
	clin := float64(-zl * zh * (1 - float64(zz/6)))
	c2, c2e := dd.TwoSum(c, u)
	ch, cl = dd.TwoSum(c2, ce+c2e+ul+float64(-0.5*zzl)+ctail+clin)
	return sh, sl, ch, cl
}

// hankelFast returns J_v(x) (Y_v(x) with wantY) and the certified bound e of
// the absolute error of the double-double value it rounds (see above); ok is
// false where the path does not apply or cannot certify.
func hankelFast(v, x float64, wantY bool) (r, e float64, ok bool) {
	r, e, _, ok = hankelFastL(v, x, wantY, hankelMaxL)
	return r, e, ok
}

// hankelCertified is the acceptance test: the bound eF of the error of F
// (in units of a) is at most 2^-56 |F|, which is at most 1/8 ulp of F.
func hankelCertified(eF, f float64) bool {
	return eF <= float64(hankelAccept*math.Abs(f))
}

// hankelFastL is hankelFast with the number of pairs of terms (P and Q each)
// capped at maxL <= hankelMaxL (a test lowers it to hold the cap from both
// sides) and l, the number of pairs the answer used.
func hankelFastL(v, x float64, wantY bool, maxL int) (r, e float64, l int, ok bool) {
	av := math.Abs(v)
	if !(x > hankelMinX && x < hankelMaxX && av <= hankelMaxOrder) {
		return 0, 0, 0, false
	}
	two := float64(2 * av) // exact
	invx := dd.Div(dd.F(1), dd.F(x))
	inv8x := 1 / float64(8*x)
	lmin := int(math.Ceil(av / 2))
	if lmin < 2 {
		lmin = 2
	}
	// The terms u_k = u_{k-1} (4v^2 - (2k-1)^2)/(8 k x) are carried in
	// double-double while |u_k| >= hankelDDFloor (there the float64 rounding
	// of a term would be seen) and in float64 below it, where a term's
	// relative error (k+2) 2^-50 is still below 2^-60 of F.
	cur, curf, inDD := dd.F(1), 0.0, true
	idx := 0
	var errT float64
	adv := func() (t dd.DD, isDD bool) {
		idx++
		if inDD {
			cur = hankelNextDD(cur, idx, two, invx)
			if math.Abs(cur.Hi) < hankelDDFloor {
				inDD, curf = false, cur.Hi
			}
			return cur, true
		}
		o := float64(2*idx - 1)
		curf = float64(float64(float64(curf*float64((two-o)*(two+o)))*inv8x) * hankelInvK[idx])
		return dd.DD{Hi: curf}, false
	}
	pd, qd := dd.F(1), dd.F(0)
	pf, qf := 0.0, 0.0
	// sum adds the signed term t (of index idx) to the double-double or the
	// float64 accumulator and its rounding bound to errT
	sum := func(acc *dd.DD, accf *float64, t dd.DD, isDD bool, sg float64, k int) {
		if isDD {
			*acc = dd.Add(*acc, dd.MulF(t, sg))
			errT += float64(hankelTermDDErr * math.Abs(t.Hi))
		} else {
			*accf += float64(sg * t.Hi)
			errT += float64(float64(float64(k+4)*0x1p-50) * math.Abs(t.Hi))
		}
	}
	t1, d1 := adv()
	sum(&qd, &qf, t1, d1, 1, 1)
	prev := math.Inf(1)
	var trunc float64
	for l = 1; ; l++ {
		if l > maxL {
			return 0, 0, l, false
		}
		ue, de := adv()
		uo, do := adv()
		s := math.Abs(ue.Hi) + math.Abs(uo.Hi)
		if l >= lmin && (s <= hankelTol || s >= prev) {
			trunc = s
			break
		}
		prev = s
		sg := 1.0
		if l&1 == 1 {
			sg = -1
		}
		sum(&pd, &pf, ue, de, sg, 2*l)
		sum(&qd, &qf, uo, do, sg, 2*l+1)
	}
	// (no test of trunc here: eF >= trunc below, and a NaN fails it as well)
	P := dd.AddF(pd, pf)
	Q := dd.AddF(qd, qf)
	// angle: w = (pi/2) (n + s), (2/pi) w = (2/pi) x - (v + 1/2) mod 4
	q, f := besselReduce(x)
	gh, gl := dd.TwoSum(math.Mod(v, 4), 0.5)
	T := dd.Sub(dd.AddF(f, float64(q)), dd.DD{Hi: gh, Lo: gl})
	n := math.Floor(T.Hi + 0.5)
	s := dd.AddF(T, -n)
	ni := int(math.Mod(n, 4))
	if ni < 0 {
		ni += 4
	}
	sh, sl, ch, cl := hankelSinCos(dd.Mul(s, besselPiOver2DD))
	// (sin w, cos w) = rotation of (sin z, cos z) by ni quarter turns
	var sw, cw [2]float64
	switch ni {
	case 0:
		sw, cw = [2]float64{sh, sl}, [2]float64{ch, cl}
	case 1:
		sw, cw = [2]float64{ch, cl}, [2]float64{-sh, -sl}
	case 2:
		sw, cw = [2]float64{-sh, -sl}, [2]float64{-ch, -cl}
	default:
		sw, cw = [2]float64{-ch, -cl}, [2]float64{sh, sl}
	}
	// F = P c - Q s (J) or P s + Q c (Y), double-double
	pc, qs := cw, sw
	sgn := -1.0
	if wantY {
		pc, qs = sw, cw
		sgn = 1
	}
	a1h, a1e := dd.TwoProd(P.Hi, pc[0])
	a1l := a1e + float64(P.Hi*pc[1]) + float64(P.Lo*pc[0])
	b1h, b1e := dd.TwoProd(Q.Hi, qs[0])
	b1l := b1e + float64(Q.Hi*qs[1]) + float64(Q.Lo*qs[0])
	fh, fe := dd.TwoSum(a1h, float64(sgn*b1h))
	fl := fe + a1l + float64(sgn*b1l)
	fh, fl = dd.TwoSum(fh, fl)
	// the bound of the error of F, in units of a
	eF := float64(trunc*(1+0x1p-40)) + errT + float64(hankelTrigErr*(math.Abs(P.Hi)+math.Abs(Q.Hi))) +
		float64(hankelDDErr*(math.Abs(a1h)+math.Abs(b1h)))
	if !hankelCertified(eF, fh) {
		return 0, 0, l, false
	}
	// a = sqrt(2/(pi x)) in double-double, then ONE rounding of a F
	a := dd.Sqrt(dd.DivF(besselTwoOverPiDD, x))
	ph, pe2 := dd.TwoProd(a.Hi, fh)
	lo := pe2 + float64(a.Hi*fl) + float64(a.Lo*fh)
	return ph + lo, float64(a.Hi * eF * (1 + 0x1p-30)), l, true
}
