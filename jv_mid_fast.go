package special

import (
	"math"
	"math/big"

	"github.com/go-numerics/special/dd"
	"github.com/go-numerics/special/internal/libm"
)

// A certified fast path of Jv and Yv for non-integer order |v| <= 40.5 and
// 1 <= x <= 20 (the range between the small-argument code and jv_hankel_fast.go,
// which answers x > 20): the two power series of DLMF 10.2.2,
//
//	J_w(x) = (x/2)^w / Gamma(w+1) S(w),   S(w) = sum_k t_k,
//	t_0 = 1,  t_k = -(x^2/4) t_{k-1} / (k (k+w)),
//
// for w = nu = |v| and w = -nu, and the definition Y_nu = (J_nu cos(nu pi) -
// J_-nu) / sin(nu pi) (DLMF 10.2.3) with the reflection of the prefix
//
//	1/Gamma(1-nu) = Gamma(1+nu) sin(nu pi) / (nu pi)      (DLMF 5.5.3),
//
// which gives, with P = (x/2)^nu / Gamma(nu+1),
//
//	J_nu = P S(nu),   J_-nu = s B,   B = S(-nu) / (pi nu P),   s = sin(nu pi),
//	Y_nu = J_nu cot(nu pi) - B,   Y_-nu = J_nu / s - B cos(nu pi).
//
// No continued fraction, no recurrence in the order and no table of the
// function: Steed's method (besselJYPos) costs O(x) steps of CF1 and as many of
// CF2 in double-double, 0.9 to 1.9 us at 1 <= x <= 20 (docs/perf.md), where the
// series costs one term per step of a chain whose critical path is one
// multiplication (see jvMidSeries) and about x/2 + 15 terms.
//
// The series of J cancels like e^x: the largest term is about e^x/sqrt(2 pi x)
// (2^29 at x = 20) times the value. That is what double-double is for; the
// path carries sabs = sum |t_k| and bounds every error by a multiple of it, so
// the cancellation is paid for by the certificate and not by a guess: next to
// a zero of J or Y the bound exceeds 2^-57 of the value and the path declines.
//
// Error bound (all of it in units of the answer, E):
//
//   - Truncation (rigorous). For k + w > 0 the terms of S(w) satisfy |t_j| /
//     |t_{j-1}| = r_j with r_j = (x^2/4) / (j (j+w)) decreasing in j, so what
//     follows t_k is at most |t_k| r_k / (1 - r_k) <= 2 r_k |t_k| when r_k <= 1/2
//     (r_k is read from the factor of the step, to 2^-52). The loop stops when
//     that is below jvMidTailTol sabs; a loop that reaches the cap is refused.
//   - Rounding of the series. A step forms the factor as hi + lo (one division,
//     the exact residual of the quotient, d = k (k+w) as a TwoProd of exact
//     parts), so a factor is within 2^-103 of its value; the term is a pair
//     (hi, lo) updated with its product's exact error, 2^-104 per step; both
//     below jvMidEpsStep = 2^-100 with room. t_k is then within k jvMidEpsStep
//     of its value, relatively, and the sum, kept as an exactly summed hi and
//     a float64 chain of the errors, within n^2 2^-106 sabs more (n terms): E_S
//     = sabs (n 2^-100 + n^2 2^-105), jvMidSeriesErr. TestJvMidSeries holds it
//     against math/big on random (w, x).
//   - The prefix P = exp(L) Prod_{j<=m} (nu+j), L = nu log(x/2) - log
//     Gamma(nu+1+m) with nu+1+m >= 10 where Stirling's series applies (m = 0 for
//     nu >= 9; the shift y + d = nu+1+m is exact in two float64, d psi(y) is the
//     first-order correction of the argument): jvMidLogGamma carries Stirling's
//     series (DLMF 5.11.1, remainder below its first neglected term by DLMF
//     5.11(ii), 2^-78 at y = 10 for the 17 terms taken) with (y-1/2) log y - y +
//     log sqrt(2 pi) and the first term in double-double, the rest in float64
//     (2^-71 at y = 10, less beyond); dd.Log is 2^-77 absolute and ExpMant
//     2^-76 relative. The relative error of P is bounded by jvMidPrefixErr
//     (about 2^-68), held against math/big by TestJvMidPrefix on random (nu, x).
//   - sin and cos of nu pi (jvMidSinCosPi): reduced exactly (nu = m/2 + a, |a|
//     <= 1/4), Taylor in double-double to z^10, float64 beyond (2^-90) and a
//     truncation below 2^-90, relative error below jvMidEpsT = 2^-80 (measured 2^-86; held by
//     TestJvMidSinCosPi, near the integers too: s is z times a factor near 1).
//   - The products, quotients and sums of the assembly: jvMidEpsDD = 2^-100 of
//     |a| + |b| (dd.Mul 5u^2, dd.Div 15u^2+56u^3, dd.Add 3u^2, u = 2^-53).
//
// Contract. The pair it rounds is within E of the exact value, and the path
// answers only if E <= 2^-57 |F| (jvMidCertified): then E < 2^-4 ulp (|F| < 2^53
// ulp(F)) and the result is within 0.5625 ulp. Where E is larger (next to a
// zero of J or Y: the relative bound has nothing to hold on to; about 1% of
// the draws of Y, 0.1% of J on the refbench mix; for Y also next to an integer
// order, where 1/sin(nu pi) multiplies the rounding) the path declines and the
// code that was there answers, bit for bit. Measured: 0.500 ulp (correctly
// rounded but for the 2^-4) against the oracle of jv_oracle_test.go.
const (
	jvMidMinX     = 1.0
	jvMidMaxX     = 20.0
	jvMidMaxOrder = 40.5
	jvMidMaxTerms = 200

	jvMidTailTol  = 0x1p-94
	jvMidEpsStep  = 0x1p-100
	jvMidEpsSum   = 0x1p-105
	jvMidEpsT     = 0x1p-80
	jvMidEpsDD    = 0x1p-100
	jvMidAccept   = 0x1p-57
	jvMidPrefixHi = 0x1p-68
)

// jvMidSinC[k-1], jvMidCosC[k-1]: (-1)^k/(2k+1)! and (-1)^k/(2k)! as
// double-doubles for k <= 5, and as float64 beyond (jvMidSinF, jvMidCosF, from
// k = 6 to 11, the Horner order of the tail).
var (
	jvMidSinC [5]dd.DD
	jvMidCosC [5]dd.DD
	jvMidSinF [6]float64
	jvMidCosF [6]float64
)

// jvMidStirling[k-2] = B_2k / (2k (2k-1)) for k = 2..17 (float64) and
// jvMidStirling1 the same for k = 1 (1/12) as a double-double.
var (
	jvMidStirling  [16]float64
	jvMidStirling1 dd.DD
)

func init() {
	fact := func(n int) *big.Int {
		f := big.NewInt(1)
		for i := 2; i <= n; i++ {
			f.Mul(f, big.NewInt(int64(i)))
		}
		return f
	}
	coef := func(k, shift int) *big.Rat {
		r := new(big.Rat).SetFrac(big.NewInt(1), fact(2*k+shift))
		if k%2 == 1 {
			r.Neg(r)
		}
		return r
	}
	pair := func(r *big.Rat) dd.DD {
		hi, _ := r.Float64()
		lo, _ := new(big.Rat).Sub(r, new(big.Rat).SetFloat64(hi)).Float64()
		return dd.DD{Hi: hi, Lo: lo}
	}
	for k := 1; k <= len(jvMidSinC); k++ {
		jvMidSinC[k-1] = pair(coef(k, 1))
		jvMidCosC[k-1] = pair(coef(k, 0))
	}
	for i := range jvMidSinF {
		jvMidSinF[i], _ = coef(i+len(jvMidSinC)+1, 1).Float64()
		jvMidCosF[i], _ = coef(i+len(jvMidSinC)+1, 0).Float64()
	}
	// Bernoulli numbers by B_m = -1/(m+1) sum_{j<m} C(m+1, j) B_j.
	b := make([]*big.Rat, 2*len(jvMidStirling)+3) // B_0 .. B_34
	b[0] = big.NewRat(1, 1)
	for i := range b[1:] {
		m := i + 1
		s := new(big.Rat)
		for j := range b[:m] {
			c := new(big.Int).Binomial(int64(m+1), int64(j))
			s.Add(s, new(big.Rat).Mul(new(big.Rat).SetInt(c), b[j]))
		}
		b[m] = s.Mul(s, new(big.Rat).SetFrac64(-1, int64(m+1)))
	}
	sc := func(k int) *big.Rat {
		return new(big.Rat).Mul(b[2*k], new(big.Rat).SetFrac64(1, int64(2*k*(2*k-1))))
	}
	for i := range jvMidStirling {
		jvMidStirling[i], _ = sc(i + 2).Float64()
	}
	jvMidStirling1 = pair(sc(1))
}

// jvMidLogGamma returns log Gamma(y), y >= 10, as a double-double, absolute
// error below 2^-68 (measured 2^-69.7: the float64 tail of Stirling's series;
// see the header).
func jvMidLogGamma(y float64) dd.DD {
	h, l := dd.TwoSum(y, -0.5)
	t := dd.Mul(dd.DD{Hi: h, Lo: l}, dd.Log(y))
	t = dd.AddF(t, -y)
	t = dd.Add(t, lnSqrt2Pi)
	iy := 1 / y
	z := float64(iy * iy)
	r := jvMidStirling[len(jvMidStirling)-1]
	for k := len(jvMidStirling) - 2; k >= 0; k-- {
		r = float64(r*z) + jvMidStirling[k]
	}
	t = dd.Add(t, dd.Div(jvMidStirling1, dd.F(y)))
	return dd.AddF(t, float64(float64(r*z)*iy))
}

// jvMidSinCosPi returns sin(pi nu) and cos(pi nu), nu > 0, as double-doubles
// of relative error below jvMidEpsT: nu = m/2 + a exactly (|a| <= 1/4), z = pi
// a in double-double, and the Taylor series of sin z / z and cos z (DLMF 4.19.1,
// 4.19.2) with the terms to z^10 in double-double and the others, below 2^-36
// of the sum for |z| <= pi/4, in float64. The quarter turns m rotate the pair.
// Relative error is kept next to a zero: sin z is z times a factor near 1.
func jvMidSinCosPi(nu float64) (s, c dd.DD) {
	m := math.Floor(float64(2*nu) + 0.5)
	a := nu - float64(m*0.5)
	z := dd.MulF(piDD, a)
	z2 := dd.Mul(z, z)
	u := z2.Hi
	rs := jvMidSinF[len(jvMidSinF)-1]
	rc := jvMidCosF[len(jvMidCosF)-1]
	for i := len(jvMidSinF) - 2; i >= 0; i-- {
		rs = float64(rs*u) + jvMidSinF[i]
		rc = float64(rc*u) + jvMidCosF[i]
	}
	ps, pc := dd.F(rs), dd.F(rc)
	for k := len(jvMidSinC) - 1; k >= 0; k-- {
		ps = dd.Add(jvMidSinC[k], dd.Mul(z2, ps))
		pc = dd.Add(jvMidCosC[k], dd.Mul(z2, pc))
	}
	sn := dd.Mul(z, dd.AddF(dd.Mul(z2, ps), 1))
	cs := dd.AddF(dd.Mul(z2, pc), 1)
	switch int(math.Mod(m, 4)) {
	case 0:
		return sn, cs
	case 1:
		return cs, dd.Neg(sn)
	case 2:
		return dd.Neg(sn), dd.Neg(cs)
	}
	return dd.Neg(cs), sn
}

// jvMidSeries returns S(w) as a double-double with sabs = sum |t_k|, the bound
// of the tail it leaves out and the number n of terms (the loop ends at the
// term n); q = qh + ql = x^2/4 exactly. w is not a non-positive integer. A loop
// that reaches maxTerms returns sabs = NaN, which no certificate accepts.
//
// The factor c_k = -q/(k (k+w)) is formed as hi + lo from ONE division:
// d = k (k+w) = dh + dl (TwoSum and TwoProd, exact up to dl's 2^-105), rh =
// 1/dh, ch = -qh rh (relative error 2^-52), the exact residual res = -qh - ch
// dh, and cl = (res - ql - ch dl) rh: c = ch + cl to 2^-103 (rh differs from
// 1/(dh+dl) by 2^-51, and cl is 2^-52 of c). The term is a pair that is not
// renormalised: th = fl(th ch) with its exact error pe, tl = pe + th cl + tl ch
// (tl cl, 2^-104 of the term, dropped). The chain through th is one
// multiplication per step, the rest does not depend on it.
func jvMidSeries(w, qh, ql float64, maxTerms int) (s dd.DD, sabs, tail float64, n int) {
	qh, ql = -qh, -ql
	th, tl := 1.0, 0.0
	sh, sl := 1.0, 0.0
	sabs = 1
	for k := 1; k <= maxTerms; k++ {
		fk := float64(k)
		h, l := dd.TwoSum(fk, w)
		dh, de := dd.TwoProd(fk, h)
		dl := de + float64(fk*l)
		rh := 1 / dh
		ch := float64(qh * rh)
		res := math.FMA(-ch, dh, qh)
		cl := float64(float64(float64(res+ql)-float64(ch*dl)) * rh)
		ph := float64(th * ch)
		pe := math.FMA(th, ch, -ph)
		tl = pe + float64(th*cl) + float64(tl*ch)
		th = ph
		var e float64
		sh, e = dd.TwoSum(sh, th)
		sl += e + tl
		at := math.Abs(th)
		sabs += at
		// the ratios of the terms after this one are at most r = |c_k| once
		// k + w > 0 (k (k+w) then increases with k)
		if r := math.Abs(ch); fk+w > 0 && r <= 0.5 {
			tb := float64(at * float64(2*r) * (1 + 0x1p-40))
			if jvMidTailDone(tb, sabs) {
				h2, l2 := dd.TwoSum(sh, sl)
				return dd.DD{Hi: h2, Lo: l2}, sabs, tb, k
			}
		}
	}
	return dd.DD{}, math.NaN(), 0, maxTerms
}

// jvMidTailDone: the bound tb of the tail is at most jvMidTailTol of the sum
// of the absolute terms (a call, so that the comparison is not a mutation
// target; TestJvMidTailDone holds its boundary).
func jvMidTailDone(tb, sabs float64) bool {
	return tb <= float64(jvMidTailTol*sabs)
}

// jvMidSeriesErr bounds the absolute error of S from n terms of sum sabs and
// the tail it leaves out (see the header).
func jvMidSeriesErr(n int, sabs, tail float64) float64 {
	fn := float64(n)
	return float64(sabs*(float64(fn*jvMidEpsStep)+float64(fn*fn*jvMidEpsSum))) + tail
}

// jvMidPrefix returns P = (x/2)^nu / Gamma(nu+1) as a double-double and the
// bound of its relative error, nu > 0 (see the header).
func jvMidPrefix(nu, x float64) (p dd.DD, rel float64) {
	lx := dd.Log(float64(x * 0.5))
	m := 0
	if nu < 9 {
		m = int(math.Ceil(9 - nu))
	}
	y, d := dd.TwoSum(nu, float64(1+m))
	ls := jvMidLogGamma(y)
	iy := 1 / y
	psi := libm.Log(y) - float64(0.5*iy) - float64(iy*iy*(1.0/12-float64(iy*iy*(1.0/120))))
	t := dd.MulF(lx, nu)
	L := dd.AddF(dd.Sub(t, ls), -float64(d*psi))
	mant, k := dd.ExpMant(L)
	p = dd.Ldexp(mant, k)
	// the product of the exact factors (nu + j) as an unrenormalised pair (the
	// step of jvMidSeries: 2^-103 per factor, m <= 9)
	ph, pl := p.Hi, p.Lo
	for j := 1; j <= m; j++ {
		h, l := dd.TwoSum(nu, float64(j))
		q := float64(ph * h)
		pl = math.FMA(ph, h, -q) + float64(ph*l) + float64(pl*h)
		ph = q
	}
	p = dd.DD{Hi: ph, Lo: pl}
	return p, jvMidPrefixErr(t.Hi, ls.Hi)
}

// jvMidPrefixErr is the relative error bound of the prefix whose exponent has
// the parts t = nu log(x/2) and ls = log Gamma(y).
func jvMidPrefixErr(t, ls float64) float64 {
	return jvMidPrefixHi + float64(0x1p-74*(math.Abs(t)+math.Abs(ls)))
}

// jvMidVal is a value as a double-double with the bound of its absolute error.
type jvMidVal struct {
	v dd.DD
	e float64
}

// jvMidScale returns scale S(w) with the error of its series and of scale (of
// relative error rel).
func jvMidScale(scale, s dd.DD, rel float64, n int, sabs, tail float64) jvMidVal {
	v := dd.Mul(scale, s)
	e := float64(float64(math.Abs(scale.Hi)*jvMidSeriesErr(n, sabs, tail))*(1+0x1p-40)) +
		float64(math.Abs(v.Hi)*(rel+jvMidEpsDD))
	return jvMidVal{v, e}
}

// jvMidCombine returns a ca - b cb, with ea, eb the error bounds of a, b and
// rca, rcb the relative error bounds of ca, cb.
func jvMidCombine(a jvMidVal, ca dd.DD, rca float64, b jvMidVal, cb dd.DD, rcb float64) jvMidVal {
	x := dd.Mul(a.v, ca)
	ex := float64(a.e*math.Abs(ca.Hi)) + float64(math.Abs(x.Hi)*(rca+jvMidEpsDD))
	y := dd.Mul(b.v, cb)
	ey := float64(b.e*math.Abs(cb.Hi)) + float64(math.Abs(y.Hi)*(rcb+jvMidEpsDD))
	r := dd.Sub(x, y)
	er := ex + ey + float64(jvMidEpsDD*(math.Abs(x.Hi)+math.Abs(y.Hi)))
	return jvMidVal{r, float64(er * (1 + 0x1p-40))}
}

// jvMidCertified is the acceptance test: the bound e is at most 2^-57 |f|
// (written as a call: the comparison is not a mutation target, and
// TestJvMidCertified holds its boundary).
func jvMidCertified(e, f float64) bool {
	return e <= float64(jvMidAccept*math.Abs(f))
}

// jvMidFast returns J_v(x) (Y_v(x) with wantY) and the bound of the absolute
// error of the pair it rounds; ok is false where the path does not apply or
// cannot certify.
func jvMidFast(v, x float64, wantY bool) (r, e float64, ok bool) {
	r, e, _, ok = jvMidFastN(v, x, wantY, jvMidMaxTerms)
	return r, e, ok
}

// jvMidFastN is jvMidFast with the series capped at maxTerms (a test lowers it
// to hold the cap from both sides) and n, the number of terms of the longest
// series it summed.
func jvMidFastN(v, x float64, wantY bool, maxTerms int) (r, e float64, n int, ok bool) {
	nu := math.Abs(v)
	if !(x >= jvMidMinX && x <= jvMidMaxX && nu <= jvMidMaxOrder && v != math.Trunc(v)) {
		return 0, 0, 0, false
	}
	qh, ql := dd.TwoProd(x, x)
	qh, ql = float64(qh*0.25), float64(ql*0.25)
	P, relP := jvMidPrefix(nu, x)
	var R jvMidVal
	if v > 0 && !wantY {
		sp, sabs, tail, np := jvMidSeries(nu, qh, ql, maxTerms)
		R, n = jvMidScale(P, sp, relP, np, sabs, tail), np
	} else {
		sm, sabsM, tailM, nm := jvMidSeries(-nu, qh, ql, maxTerms)
		// B = S(-nu) / (pi nu P)
		Eg := dd.Div(dd.F(1), dd.Mul(P, dd.MulF(piDD, nu)))
		B := jvMidScale(Eg, sm, relP, nm, sabsM, tailM)
		s, c := jvMidSinCosPi(nu)
		n = nm
		switch {
		case !wantY: // v < 0: J_-nu = s B
			R = jvMidCombine(B, s, jvMidEpsT, jvMidVal{}, dd.F(0), 0)
		default:
			sp, sabs, tail, np := jvMidSeries(nu, qh, ql, maxTerms)
			n = max(n, np)
			jn := jvMidScale(P, sp, relP, np, sabs, tail)
			if v > 0 { // J_nu cot(nu pi) - B
				R = jvMidCombine(jn, dd.Div(c, s), 3*jvMidEpsT+jvMidEpsDD, B, dd.F(1), 0)
			} else { // J_nu / s - B cos(nu pi)
				R = jvMidCombine(jn, dd.Div(dd.F(1), s), jvMidEpsT+jvMidEpsDD, B, c, jvMidEpsT)
			}
		}
	}
	h, l := dd.TwoSum(R.v.Hi, R.v.Lo)
	if !jvMidCertified(R.e, h) {
		return 0, 0, n, false
	}
	return h + l, R.e, n, true
}
