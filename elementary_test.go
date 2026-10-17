package special

import (
	"math"
	"testing"

	"github.com/go-numerics/special/internal/dd"
)

// Unit checks of the float64 fast paths (kernels.go, fastpoly.go) that the
// fixtures alone do not reach; the accuracy itself is held by TestAccuracy.

func TestEvalShortDegrees(t *testing.T) {
	for name, p := range map[string][]polyPiece{
		"erfSmall": erfSmallPieces[:], "erfcxTail": erfcxTailPieces[:],
		"logErfcxTail": logErfcxTailPieces[:], "zeta": zetaPieces[:],
	} {
		if d := maxDegree(p); d > 10 {
			t.Errorf("%s: degree %d, evalShort reads 10 coefficients", name, d)
		}
	}
	for name, p := range map[string][]polyPiece{
		"erfcx": erfcxPieces[:], "gamma1p[0,1]": gamma1pPieces[2:], "digamma1p": digamma1pPieces[:],
		"digammaMid": digammaMidPieces[:], "erfInvSmall": erfInvSmallPieces[:], "erfcInv": erfcInvPieces[:],
		"logErfcx": logErfcxPieces[:], "zetaTail[1:]": zetaTailPieces[1:],
	} {
		if d := maxDegree(p); d > 16 {
			t.Errorf("%s: degree %d, estPiece holds 16 coefficients", name, d)
		}
	}
}

// TestHurwitzShiftBound checks the claim behind hurwitzFast's shift
// w0 = 7.3 + s/2: the Bernoulli series, evaluated as there, reaches its
// 2^-64 stopping test within the 25 tabulated terms for every s in (1, 64]
// (the terms decrease with w, so w = w0 is the worst case).
func TestHurwitzShiftBound(t *testing.T) {
	for s := 1.0 + 1.0/1024; s <= 64; s += 1.0 / 64 {
		w := 7.3 + 0.5*s
		bh := w / (s - 1)
		poch, pw := s*(s+1)*(s+2), math.Pow(w, -3)
		ok := false
		for j := 2; j <= len(bernoulliOverFactorial25); j++ {
			if math.Abs(bernoulliOverFactorial25[j-1]*poch*pw) < 0x1p-64*bh {
				ok = true
				break
			}
			k := float64(2*j - 1)
			poch *= (s + k) * (s + k + 1)
			pw /= w * w
		}
		if !ok {
			t.Fatalf("s = %v: Bernoulli series not converged at w = %v", s, w)
		}
	}
}

func TestErfcDDBranches(t *testing.T) {
	for _, z := range []float64{0.7, 3, 12, -0.7, -3, 0.2, -0.2} {
		got := erfcDD(dd.F(z)).Float()
		want := Erfc(z)
		if math.Abs(got-want) > 2*ulp(want) {
			t.Errorf("erfcDD(%v) = %v, Erfc = %v", z, got, want)
		}
	}
}

func TestGammaPosBranches(t *testing.T) {
	if v := gammaPos(200).Float(); !math.IsInf(v, 1) {
		t.Errorf("gammaPos(200) = %v", v)
	}
	for _, x := range []float64{1.5, 2, 50.5, 0.3, 7.25} {
		if got, want := gammaPos(x).Float(), Gamma(x); math.Abs(got-want) > ulp(want) {
			t.Errorf("gammaPos(%v) = %v, Gamma = %v", x, got, want)
		}
	}
}

func TestXlogySubnormal(t *testing.T) {
	y := 5e-320
	want := 3 * logf(y)
	if got := Xlogy(3, y); math.Abs(got-want) > 2*ulp(want) {
		t.Errorf("Xlogy(3, %v) = %v, want %v", y, got, want)
	}
	if got := logf(2); got != math.Log(2) {
		t.Errorf("logf(2) = %v", got)
	}
}
