package special

import (
	"math"
	"math/bits"

	"github.com/go-numerics/special/internal/dd"
)

// Double-double constants (hi + lo to about 107 bits).
var (
	besselPiOver2DD   = dd.DD{Hi: 1.5707963267948966, Lo: 6.123233995736766e-17}
	besselTwoOverPiDD = dd.DD{Hi: 0.6366197723675814, Lo: -3.935735335036497e-17}
)

// besselTwoOverPiWord returns the 64 binary digits of 2/pi that start at
// digit s (digit 1 is the first after the point; digits s <= 0 are zero).
func besselTwoOverPiWord(s int) uint64 {
	if s < 1 {
		sh := 1 - s
		if sh >= 64 {
			return 0
		}
		return besselTwoOverPiWord(1) >> uint(sh)
	}
	off := s - 1
	w, b := off/64, uint(off%64)
	if b == 0 {
		return besselTwoOverPiBits[w]
	}
	return besselTwoOverPiBits[w]<<b | besselTwoOverPiBits[w+1]>>(64-b)
}

// besselReduce returns q and f with x*(2/pi) = 4N + q + f for some integer
// N, q in {0, 1, 2, 3} and f in [-1/2, 1/2) as a double-double accurate to
// about 2^-150 absolutely, for any finite x > 0.
//
// This is the Payne-Hanek reduction (M. H. Payne and R. N. Hanek, "Radian
// reduction for trigonometric functions", SIGNUM Newsletter 18(1), 1983):
// with x = M*2^E, only the digits of 2/pi from position E-1 on contribute to
// x*(2/pi) mod 4, and 256 of them, multiplied exactly by the 53-bit M,
// leave more than 190 bits after the point. The closest a double comes to a
// multiple of pi/2 is about 2^-62 (Kahan, McDonald), so f keeps its full
// double-double precision.
func besselReduce(x float64) (q int, f dd.DD) {
	b := math.Float64bits(x)
	e := int(b>>52) & 0x7ff
	m := b & (1<<52 - 1)
	if e == 0 {
		e = 1
	} else {
		m |= 1 << 52
	}
	E := e - 1075 // x = m * 2^E
	i0 := E - 1
	w0 := besselTwoOverPiWord(i0)
	w1 := besselTwoOverPiWord(i0 + 64)
	w2 := besselTwoOverPiWord(i0 + 128)
	w3 := besselTwoOverPiWord(i0 + 192)
	// R = m * (w0:w1:w2:w3); x*(2/pi) mod 4 = R * 2^-254 mod 4.
	h3, l3 := bits.Mul64(m, w3)
	h2, l2 := bits.Mul64(m, w2)
	h1, l1 := bits.Mul64(m, w1)
	_, l0 := bits.Mul64(m, w0)
	r0 := l3
	r1, c := bits.Add64(h3, l2, 0)
	r2, c := bits.Add64(h2, l1, c)
	r3, _ := bits.Add64(h1, l0, c)
	q = int(r3 >> 62)
	r3 &= 1<<62 - 1
	neg := false
	if r3>>61 != 0 { // f >= 1/2: use f - 1
		neg = true
		q = (q + 1) & 3
		// 2^254 - F over the 254-bit fraction
		var bw uint64
		r0, bw = bits.Sub64(0, r0, 0)
		r1, bw = bits.Sub64(0, r1, bw)
		r2, bw = bits.Sub64(0, r2, bw)
		r3, _ = bits.Sub64(1<<62, r3, bw)
	}
	// normalise the 256-bit number r3:r2:r1:r0 (value F*2^-254)
	w := [4]uint64{r3, r2, r1, r0}
	sh := 0
	for w[0] == 0 && sh < 192 {
		w[0], w[1], w[2], w[3] = w[1], w[2], w[3], 0
		sh += 64
	}
	lz := uint(bits.LeadingZeros64(w[0]))
	if lz > 0 {
		w[0] = w[0]<<lz | w[1]>>(64-lz)
		w[1] = w[1]<<lz | w[2]>>(64-lz)
	}
	sh += int(lz)
	// f = w0*2^(-62-sh) + w1*2^(-126-sh)
	hi := math.Ldexp(float64(w[0]>>11), -51-sh)
	lo := math.Ldexp(float64(w[0]&0x7ff), -62-sh) + math.Ldexp(float64(w[1]), -126-sh)
	hi, lo = dd.FastTwoSum(hi, lo)
	if neg {
		hi, lo = -hi, -lo
	}
	return q, dd.DD{Hi: hi, Lo: lo}
}

// besselSinCosQuarter returns sin and cos of (pi/2)*(n + s) for an integer
// n and a double-double s with |s| <= 1/2 (about), to full relative
// precision: the angle s*pi/2 is formed in double-double and its low part
// enters through the first-order term.
func besselSinCosQuarter(n int, s dd.DD) (sn, cs float64) {
	a := dd.Mul(s, besselPiOver2DD)
	sh, ch := math.Sincos(a.Hi)
	sa := sh + a.Lo*ch
	ca := ch - a.Lo*sh
	switch n & 3 {
	case 0:
		return sa, ca
	case 1:
		return ca, -sa
	case 2:
		return -sa, -ca
	default:
		return -ca, sa
	}
}

// besselAsymTol is the size below which the terms of the asymptotic
// expansions must fall for the expansion to be used.
const besselAsymTol = 0x1p-58

// besselAsymJY returns J_v(x) and Y_v(x) for large x by the modulus-phase
// form J = M cos(theta), Y = M sin(theta) of Hankel's expansion (DLMF
// 10.18.17 for M^2, and theta from DLMF 10.18.8, theta' = 2/(pi x M^2),
// integrated term by term; its first terms are DLMF 10.18.18). ok is false
// when the expansions do not reach besselAsymTol before their terms start
// to grow, or when a needed function is so close to a zero that the error
// of the phase would show in its relative error (j and y are still the
// best values this method gives); v may be of either sign.
//
// Accuracy near the zeros of J and Y is what this form is for: the phase
// x - (v/2 + 1/4)pi + (mu-1)/(8x) + (mu-1)(mu-25)/(384x^3) + ... is formed
// in double-double (x reduced mod pi/2 exactly by besselReduce, the first
// two correction terms in double-double), and then reduced to the nearest
// multiple of pi/2, so cos(theta) keeps its relative accuracy where it is
// small. (Hankel's P cos(chi) - Q sin(chi) form, as in Cephes and Boost,
// loses it there.)
func besselAsymJY(v, x float64, needJ, needY bool) (j, y float64, ok bool) {
	mu := 4 * v * v
	// S = sum_{k>=1} T_k, T_k = c_k/x^(2k): M^2 = 2/(pi x) (1 + S).
	// D_k: 1/(1+S) = sum D_k (D_0 = 1); phase terms e_k = x D_k/(1-2k).
	// Both series are asymptotic: they are summed while their terms
	// decrease (optimal truncation), down to 2^-110.
	var t [80]float64
	var d [80]float64
	d[0] = 1
	ix2 := 1 / (x * x)
	S := 0.0
	phase := 0.0    // sum of e_k for k >= 3
	phaseAbs := 0.0 // sum of |e_k| for k >= 3 (rounding error bound)
	last := math.Inf(1)
	for k := 1; k < len(t); k++ {
		o := float64(2*k - 1)
		if k == 1 {
			t[1] = (mu - 1) / 8 * ix2
		} else {
			t[k] = t[k-1] * (o / float64(2*k)) * (mu - o*o) / 4 * ix2
		}
		s := 0.0
		for i := 1; i <= k; i++ {
			s -= t[i] * d[k-i]
		}
		d[k] = s
		ek := x * d[k] / (1 - float64(2*k))
		size := math.Max(math.Abs(t[k]), math.Abs(ek))
		if size > last {
			break
		}
		S += t[k]
		if k >= 3 {
			phase += ek
			phaseAbs += math.Abs(ek)
		}
		last = size
		if size < 0x1p-110 {
			break
		}
	}
	if !(last < besselAsymTol) {
		return math.NaN(), math.NaN(), false
	}
	// e_1 + e_2 in double-double.
	mu1 := dd.AddF(dd.MulFF(2*v, 2*v), -1)
	var eps dd.DD
	if x < 0x1p300 {
		e1 := dd.DivF(mu1, 8*x)
		x3 := dd.MulF(dd.MulFF(x, x), x)
		e2 := dd.Div(dd.Mul(mu1, dd.AddF(dd.MulFF(2*v, 2*v), -25)), dd.MulF(x3, 384))
		eps = dd.AddF(dd.Add(e1, e2), phase)
	} else {
		// e2 and beyond are below 2^-900; e1 needs no extra precision.
		eps = dd.F(mu1.Hi / 8 / x)
	}
	// theta*(2/pi) = x*(2/pi) - (v + 1/2) + eps*(2/pi), reduced mod 4.
	q, f := besselReduce(x)
	vm := math.Mod(v, 4) // exact
	gh, gl := dd.TwoSum(vm, 0.5)
	T := dd.Sub(dd.AddF(f, float64(q)), dd.DD{Hi: gh, Lo: gl})
	T = dd.Add(T, dd.Mul(eps, besselTwoOverPiDD))
	n := math.Floor(T.Hi + 0.5)
	s := dd.AddF(T, -n)
	ni := int(math.Mod(n, 4))
	if ni < 0 {
		ni += 4
	}
	sn, cs := besselSinCosQuarter(ni, s)
	// The phase is known to about errPh (radians): truncation, plus the
	// rounding of the terms summed in float64. J = M cos(theta) is then
	// relatively accurate only where |cos(theta)| >> errPh; next to a zero
	// the caller is told to use another method.
	errPh := last + phaseAbs*0x1p-52 + 0x1p-100
	if (needJ && errPh > 0x1p-56*math.Abs(cs)) || (needY && errPh > 0x1p-56*math.Abs(sn)) {
		ok = false
	} else {
		ok = true
	}
	// M = sqrt(2/(pi x) (1+S)), scaled to stay normal for huge x.
	scale := 1.0
	xs := x
	if x > 0x1p1000 {
		xs = x * 0x1p-200
		scale = 0x1p-100
	}
	m2 := dd.Mul(dd.DivF(besselTwoOverPiDD, xs), dd.AddF(dd.F(1), S))
	M := dd.Sqrt(m2)
	j = (M.Hi*cs + M.Lo*cs) * scale
	y = (M.Hi*sn + M.Lo*sn) * scale
	return j, y, ok
}
