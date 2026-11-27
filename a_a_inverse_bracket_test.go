package special

import (
	"math"
	"testing"
)

// bracketTol is the relative slack of the monotone bracket below: the
// forward functions are accurate to a few 1e-16, the slack is 1e-12.
const bracketTol = 1e-12

// TestIncBetaInverseBracket checks, on a grid that crosses every branch bound
// of ibetaInv (a, b around 1e-3, 1, 40; p and q in both tails), that the
// inverse x of I_x(a, b) = p lies between two neighbours whose forward values
// enclose p: I(x-2ulp) <= p <= I(x+2ulp), and the same for 1 - I = q. The
// grid test above only checks the range of the result.
func TestIncBetaInverseBracket(t *testing.T) {
	ab := []float64{4e-4, 9e-4, 1.1e-3, 3e-3, 0.05, 0.4, 0.9, 1, 1.0001, 1.7, 3, 12, 39, 41, 300, 2e4}
	ps := []float64{1e-30, 1e-12, 1e-6, 1e-3, 0.05, 0.3, 0.5, 0.7, 0.95, 0.999, 1 - 1e-9}
	lo := func(x float64) float64 { return math.Nextafter(math.Nextafter(x, 0), 0) }
	hi := func(x float64) float64 { return math.Nextafter(math.Nextafter(x, 1), 1) }
	for _, a := range ab {
		for _, b := range ab {
			for _, p := range ps {
				x := BetaIncInv(a, b, p)
				if x > 0 && x < 1 {
					if l := BetaInc(a, b, lo(x)); l > p*(1+bracketTol) {
						t.Errorf("BetaIncInv(%g, %g, %g) = %g: I(x-2ulp) = %g > p", a, b, p, x, l)
					}
					if h := BetaInc(a, b, hi(x)); h < p*(1-bracketTol) {
						t.Errorf("BetaIncInv(%g, %g, %g) = %g: I(x+2ulp) = %g < p", a, b, p, x, h)
					}
				}
				xc := BetaIncCInv(a, b, p)
				if xc > 0 && xc < 1 {
					// 1 - I = q decreases with x
					if h := BetaIncC(a, b, lo(xc)); h < p*(1-bracketTol) {
						t.Errorf("BetaIncCInv(%g, %g, %g) = %g: Ic(x-2ulp) = %g < q", a, b, p, xc, h)
					}
					if l := BetaIncC(a, b, hi(xc)); l > p*(1+bracketTol) {
						t.Errorf("BetaIncCInv(%g, %g, %g) = %g: Ic(x+2ulp) = %g > q", a, b, p, xc, l)
					}
				}
			}
		}
	}
}
