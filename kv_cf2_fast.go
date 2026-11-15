package special

import (
	"math"

	"github.com/go-numerics/special/dd"
)

// Path 3 of kv_fast.go: K_v e^x for 0 < v < 32 and kvCF2MinX <= x < 24, the
// region that used to fall back to besselIK (Steed's CF2 in double-double all
// the way, 0.5 to 5 us). With n the nearest integer to v and u = v - n, |u| <=
// 1/2 (exact), the pair K_u e^x and K_{u+1} e^x from the continued fraction CF2
// (kvCF2Pair), then the hi + lo upward recurrence of the other paths
// (kvChainUp). Bessels.jl (besselk.jl at 009b615, read 2026-11-15) takes a Levin
// transform of the asymptotic series for x > 1.5 and the power series below;
// neither is certifiable to 2^-58 and neither is ported here.
//
// The fraction is Steed's algorithm for CF2 (Temme 1975; Thompson and Barnett
// 1986, as in Boost's CF2_ik and bessel_ik.go; neither paper was reopened for
// this change: the formulas below are held by the math/big oracle of
// kv_oracle_test.go and by an independent evaluation of the fraction at 320
// bits, TestKvCF2FBracket), reorganised so that no step divides by a_k:
//
//	a_k = u^2 - (k - 1/2)^2,   b_k = 2 (x + k),
//	D_k = 1 / (b_k + a_k D_(k-1)),   delta_k = delta_(k-1) (-a_k) D_(k-1) D_k,
//	delta_1 = D_1 = 1/b_1,   f = sum_k delta_k,
//	t_1 = -a_1,   t_k = (a_(k-1) t_(k-2) / (k-1) + b_(k-1) t_(k-1)) / k   (t_0 = 0),
//	Q_k = sum_(j <= k) t_j,   S = 1 + sum_k Q_k delta_k,
//
// and then K_u e^x = sqrt(pi/(2x)) / S, K_(u+1) / K_u = (x + 1/2 + u + a_1 f) / x.
// t_k is the C_k q_k of Steed's variables (C_k = C_(k-1) (-a_k) / k, q_k =
// (q_(k-2) - b_(k-1) q_(k-1)) / a_k): the division by a_k cancels.
//
// Signs, for |u| < 1/2 and k >= 2 (TestKvCF2Signs checks them on a grid of the
// region; a_k < 0, D_k > 0 and delta_k > 0 follow from the recurrences, t_k > 0
// and Q_k increasing are observed): every term of f and of S is positive, so
// nothing cancels and the partial sums rise to their limits. Q_k grows without
// bound (like exp(2 sqrt(2 x k))) while delta_k falls like exp(-4 sqrt(2 x k)):
// the terms Q_k delta_k fall like exp(-2 sqrt(2 x k)), which is why S needs
// about 120 steps at x = 2 and 20 at x = 24. For u = +-1/2, a_1 = 0, S = 1 and f
// is not needed.
//
// Precision. b_1 = 2x + 2 is carried as the exact pair; the first
// kvCF2Terms(x) steps (4, 3, 2 by x) run in double-double (the sloppy
// additions of bessel_ik.go: 2^-104 each); the rest in float64, in the order
// above. The terms after the double-double ones weigh at most 9.2e-4 (x < 3),
// 3.3e-4 (x < 8) and 1.0e-4 of S (TestKvCF2TermWeights, on a grid of u and x),
// so their float64 rounding (a few 2^-53 relative each, summed in float64
// without compensation and added to the double-double S once) is below 2^-60
// of S. The tail of f enters K_(u+1) / K_u only through a_1 f / (x + 1/2 +
// u) <= 5% and is summed in float64: below 2^-58 of the ratio. K_u e^x and the
// ratio are then formed in double-double (2^-104 per operation). Measured
// against the oracle (TestKvCF2PairOracle; the pull request quotes 1.2e5
// points): K_u within 2^-58.6, K_(u+1) within 2^-57.5 relative at worst.
//
// Truncation. For f the bound is rigorous: with c_k = -a_k, the tail tau_k =
// c_(k+1) / (b_(k+1) - tau_(k+1)) of the fraction after level k is at most w_k
// = k + 1/2, because (k + 1/2)(2x + k + 1/2) >= c_(k+1) = (k + 1/2)^2 - u^2
// (induction from the truncation depth, where tau = 0; the limit exists because
// the fraction converges). The value is increasing in the tail, and the
// determinant formula of continued fractions gives f(tau) - f_k = tau delta_k
// D_k / (1 - tau D_k), so f - f_k <= w_k delta_k D_k / (1 - w_k D_k)
// (kvCF2FTail, checked against the 320-bit fraction in TestKvCF2FBracket). The
// loop stops when a_1 times this is below 2^-58. For S there is no such proof:
// it stops when a term Q_k delta_k is below 2^-64 and below 0.9 of the previous
// one (the ratio of consecutive terms rises towards 1 with k, so the tail is a
// few terms at x near kvCF2MinX and one at x = 24), and TestKvCF2Truncation
// compares the stopped pair with the loop run to 2^-84 on 1e5 points: 2^-61.5
// relative at worst, at x just above kvCF2MinX. NOT PROVEN for all (u, x). The
// cap kvCF2MaxIters declines (the answer is then besselIK's).
//
// Total: 0.5 ulp for the rounding, the start pair 2^-57.5 (0.045 ulp), the
// recurrence R_n <= R_0 + n 2^-91 (kn_fast.go's derivation, n <= 32) and 2^-76
// for e^-x: below 0.55 ulp; measured 0.52 on the sweeps, 0.50 on the fixtures.
const (
	// kvCF2MinX is where the double-double steps (4 below 3) still hold the
	// start pair at 2^-57.5; at x = 1.4 it reaches 2^-56 (a_1 f weighs 5%).
	kvCF2MinX = 1.5
	// kvCF2Stop, kvCF2FStop: the stopping tolerances for S and for a_1 f.
	kvCF2Stop  = 0x1p-64
	kvCF2FStop = 0x1p-58
	// kvCF2MaxIters caps the float64 loop (kvInv has this many entries).
	kvCF2MaxIters = 1000
)

// kvInv[k] = 1/k rounded to float64 (kvInv[0] is unused, +Inf): the float64
// phase multiplies by it where Steed's variables divide, a dependent chain of
// two divisions per step otherwise.
var kvInv = kvInvTable()

func kvInvTable() (t [kvCF2MaxIters + 1]float64) {
	for k := 1; k <= kvCF2MaxIters; k++ {
		t[k] = 1 / float64(k)
	}
	return
}

// kvCF2FTail is the rigorous bound on f - f_k described above, after step k
// with delta_k = dl and D_k = Df: w dl Df / (1 - w Df), w = k + 1/2 (+Inf if
// w Df >= 1, where the bracket is not established).
func kvCF2FTail(k int, dl, Df float64) float64 {
	w := float64(k) + 0.5
	den := 1 - float64(w*Df)
	if !(den > 0) {
		return math.Inf(1)
	}
	return float64(w*dl) * Df / den
}

// kvCF2Terms is the number of terms run in double-double at x.
func kvCF2Terms(x float64) int {
	switch {
	case x < 3:
		return 4
	case x < 8:
		return 3
	}
	return 2
}

// kvCF2Pair returns K_u(x) e^x and K_(u+1)(x) e^x as double-doubles for |u| <=
// 1/2 and kvCF2MinX <= x < 24, ok false if the iteration cap is reached.
func kvCF2Pair(u, x float64) (k0, k1 dd.DD, ok bool) {
	return kvCF2PairTol(u, x, kvCF2Stop, kvCF2FStop)
}

// kvCF2PairTol is kvCF2Pair with the two stopping tolerances as parameters
// (the tests run it far below the production ones to measure the truncation).
func kvCF2PairTol(u, x, stop, fstop float64) (k0, k1 dd.DD, ok bool) {
	a1 := besselAddFS(dd.MulFF(u, u), -0.25)
	bh, bl := dd.TwoSum(float64(2*x), 2)
	b := dd.DD{Hi: bh, Lo: bl}
	D := besselRecipDD(b)
	f, delta := D, D
	tm2 := dd.DD{}
	tm1 := dd.Neg(a1)
	Q := tm1
	S := besselAddS(dd.F(1), dd.Mul(Q, delta))
	a := a1
	m := kvCF2Terms(x)
	k := 2
	for ; k <= m; k++ {
		ak1, bprev := a, b
		a = besselAddFS(a, float64(-2*(k-1)))
		b = besselAddFS(b, 2)
		Dp := D
		D = besselRecipDD(besselAddS(b, dd.Mul(a, D)))
		delta = dd.Mul(dd.Mul(dd.Neg(a), delta), dd.Mul(Dp, D))
		f = besselAddS(f, delta)
		tk := dd.Mul(bprev, tm1)
		if k > 2 {
			tk = besselAddS(tk, igamDivF(dd.Mul(ak1, tm2), float64(k-1)))
		}
		tk = igamDivF(tk, float64(k))
		tm2, tm1 = tm1, tk
		Q = besselAddS(Q, tk)
		S = besselAddS(S, dd.Mul(Q, delta))
	}
	// the rest in float64
	ak, bk, Df, dl := a.Hi, b.Hi, D.Hi, delta.Hi
	t2, t1, Qf := tm2.Hi, tm1.Hi, Q.Hi
	var tail, ftail float64
	aa := math.Abs(a1.Hi) // f enters K_(u+1) / K_u only through a_1 f: u = +-1/2 needs none of it
	prev := math.Inf(1)
	converged := false
	for ; k < kvCF2MaxIters; k++ {
		ak1, bprev, Dp := ak, bk, Df
		ak -= float64(2 * (k - 1))
		bk += 2
		Df = 1 / math.FMA(ak, Dp, bk)
		dl = float64(dl * (float64(float64(-ak*Dp) * Df)))
		tk := float64(math.FMA(float64(ak1*t2), kvInv[k-1], float64(bprev*t1)) * kvInv[k])
		t2, t1 = t1, tk
		Qf += tk
		term := float64(Qf * dl)
		tail += term
		ftail += dl
		if term < stop && term < float64(0.9*prev) && !(float64(aa*kvCF2FTail(k, dl, Df)) >= fstop) {
			converged = true
			break
		}
		prev = term
	}
	if !converged {
		return dd.DD{}, dd.DD{}, false
	}
	S = dd.AddF(S, tail)
	fsum := float64(f.Hi + ftail)
	c := dd.Sqrt(igamDivF(besselPiOver2DD, x))
	k0 = dd.Mul(c, besselRecipDD(S))
	p, pe := dd.TwoSum(0.5, u)
	s, se := dd.TwoSum(x, p)
	nh, ne := dd.TwoSum(s, float64(a1.Hi*fsum))
	nh, nl := dd.FastTwoSum(nh, ne+(se+pe)+float64(a1.Lo*fsum))
	k1 = dd.Mul(k0, igamDivF(dd.DD{Hi: nh, Lo: nl}, x))
	return k0, k1, true
}
