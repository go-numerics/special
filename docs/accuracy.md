# Accuracy

Accuracy is what this package sells, so it is measured, published and
enforced by the test suite, not asserted.

The per-region table is generated: [accuracy-table.md](accuracy-table.md).
This page explains how it is produced, which algorithm each function uses
and where it comes from, what the negative control showed, and where the
package is not (yet) as good as it should be.

## How the error is measured

**Oracle.** Every fixture value is computed by [mpmath](https://mpmath.org)
1.4.1 twice, at 192 and at 320 bits of working precision (with extra
precision where the formula cancels, e.g. `log Γ(a) + log Γ(b) - log Γ(a+b)`
for huge arguments, which is evaluated with 1200 extra bits). A point whose
two evaluations differ by more than 2^-80 relative is dropped, and the drop
is counted in the fixture header: an oracle that silently lost precision
must not become the truth the tests are held to. (The first runs dropped
247 `logbeta` points this way, which led to the extra-precision fix.)

**Exact reference, not a rounded one.** Each row stores the mpmath value as a
double-double `hi + lo`, `hi` being the correctly rounded float64. The error
of a result `y` is `|y - (hi + lo)| / ulp(hi)`, so a correctly rounded
result scores at most 0.5, and the half-ulp of the reference's own rounding
is not added to anyone's error.

**SciPy on the same points.** The generator also stores SciPy 1.18.1's
result for the same float64 arguments, and the Go test measures it exactly
as it measures this package. SciPy is a point of comparison, never the
reference.

**Regions.** Each function's domain is cut into regions — tails, the
neighbourhood of zeros and poles, huge and tiny arguments, subnormals,
special values (NaN, ±Inf, ±0) — sampled with a fixed seed (log-uniform
where the scale matters). About 1000 points per region; 160 000 points
over all fixtures. A result that should be NaN or ±Inf and is not (or the
reverse) is reported as `wrong`.

**Budgets.** `TestAccuracy` fails if any region's maximum error exceeds its
budget (`budgets_test.go`: 1.5 times the error measured when the budgets
were generated, at least 1 ulp). The suite runs on amd64, arm64, 386, arm,
riscv64, loong64, ppc64le and s390x in CI: architectures where Go fuses
multiply-adds and ones where it does not must all meet the same budgets.

Regenerate:

```
uv run --with mpmath==1.4.1 --with scipy==1.18.1 --with numpy==2.5.4 \
    python testdata/gen/gen.py            # fixtures (minutes)
go test -run TestAccuracyTable -accuracy.table=docs/accuracy-table.md
go test -run TestAccuracyBudgets -accuracy.budgets=budgets_test.go
```

## Headline

SUMMARY_PLACEHOLDER

## Algorithms and sources

Two building blocks run through everything:

- **Double-double arithmetic** (`internal/dd`): error-free transformations
  (Knuth's TwoSum, Dekker's Fast2Sum, FMA-based TwoProd) and the double-word
  algorithms of Joldes, Muller & Popescu (ACM TOMS 44(2), 2017). `dd.Exp`
  and `dd.Log` are table-driven after Tang (ACM TOMS 15(2), 1989 and 16(4),
  1990), carried to double-double (about 2^-76 and 2^-72 relative), with
  300-bit mpmath tables.
- **Fitted polynomials** (`testdata/gen/fit.py`, `zz_*_coeffs.go`): Chebyshev
  interpolants of a well-chosen *regular part* of a function, fitted to
  mpmath at 200 bits on short pieces, degree chosen so that the
  interpolation error is below 2^-64, constant term kept in double-double.
  The function's zeros, poles and logarithmic singularities are factored
  out analytically, so the fitted part is smooth and the relative accuracy
  holds where the function vanishes.

| function | algorithm | source |
|---|---|---|
| `Gamma` | Γ(1+t) = 1 + t(t-1)F(t) on [-1/2, 1] with F fitted; exact-factor recurrence (x - k is exact) carried in double-double for 0 < x < 10 and -34 < x < 0; Stirling's series (DLMF 5.11.1) with (x-1/2)log x - x in double-double above 10; reflection with sin(πx) reduced exactly below -34; integers from a correctly rounded factorial table | DLMF 5.5, 5.11; core fitted to mpmath |
| `LogGamma`, `LogGammaSign`, `GammaSign` | log1p of Γ(1+t)-1 near the zeros at 1 and 2; Stirling in double-double above 10; exact factors for -34 < x < 0; reflection below | DLMF 5.11.1, 5.5.3 |
| `Digamma` | ψ(1+t) = (t - r)D(t) with the positive zero r in double-double and D fitted (the form of Boost.Math's `digamma_imp_1_2`); recurrence with double-double reciprocals; asymptotic series above 10 (DLMF 5.11.2); for -40 < x < 0, exact-factor recurrence, redone with a 2^-100 evaluation when the terms cancel (near the negative zeros); reflection with cot(πx) in double-double below | DLMF 5.5, 5.11; Boost.Math digamma.hpp |
| `Polygamma` | (-1)^(n+1) n! ζ(n+1, x) | DLMF 25.11.12 |
| `Beta`, `LogBeta` | below 10, Γ(a)Γ(b)/Γ(a+b) from the double-double core with the rounding of a+b corrected to first order; above, log Γ(b) - log Γ(a+b) as one cancellation-free Stirling expression (-a log b - b log1pmx(a/b) - (a-1/2) log1p(a/b) + S(b) - S(a+b)) added to log Γ(a) | DLMF 5.11.1, 5.12.1 |
| `Erf`, `Erfc`, `Erfcx` | x E(x²) for \|x\| <= 1/2; erfcx fitted on [1/2, 8] and as x·erfcx(x) in 1/x² beyond (the Gaussian factor removed, as in Cody 1969); exp(-x²) from x² in double-double, multiplied in before the single rounding | Cody, Math. Comp. 23 (1969); DLMF 7.12 |
| `Ndtr`, `LogNdtr` | erfc of -x/√2 carried in double-double (the rounding of x/√2 is what costs SciPy up to 1e12 ulp in the lower tail); log Φ as log erfcx(z) - z² - log 2 in the lower tail and log1p(-Φ(-x)) in the upper | DLMF 7.2, 7.12 |
| `ErfInv`, `ErfcInv`, `Ndtri` | Boost.Math's 64-bit rational approximations (`erf_inv_imp`) as the guess, then one Halley step whose relative residual comes from the double-double erfc (so subnormal arguments keep full accuracy) | Boost.Math detail/erf_inv.hpp (J. Maddock, BSL-1.0) |
| `Zeta` | 1/(s-1) + R(s) with R entire and fitted on [-1, 6] (exact pole); 1 + g(s)2^-s with g = (ζ(s)-1)2^s fitted on (6, 64); reflection with sin(πs/2) reduced exactly below -1 (trivial zeros exact) | DLMF 25.4.1 |
| `HurwitzZeta` | Euler–Maclaurin summation (the arrangement of Cephes `zeta.c`) with q+k exact in double-double, powers through double-double log/exp and terms summed with a common exponent (no intermediate overflow); the number of direct terms chosen so 15 Bernoulli terms reach 2^-64 | DLMF 25.11.5; Moshier, Cephes |
| `Comb`, `Perm`, `Binom`, `Factorial`, `Factorial2`, `LogFactorial` | products in double-double up to 64 factors (exact below 2^53, correctly rounded above); beyond, through `LogBeta`'s cancellation-free logarithm | — |
| `Expit`, `LogExpit`, `Log1pExp`, `Logit`, `Xlogy`, `Xlog1py`, `LogSumExp`, `Softmax`, `LogSoftmax` | the exponential and logarithm in double-double; log1p of a tiny argument by its series | — |
| `EllipK`, `EllipKm1`, `EllipE` | K(1-p) = A_K(p) - log(p)K(p)/π and E(1-p) = A_E(p) - log(p)(K(p)-E(p))/π (the structure of DLMF 19.12, of Cephes `ellpk`/`ellpe` and of Hastings' approximations) with A_K, A_E, K, E and (K-E)/p fitted; m < -1 by the imaginary-modulus transformation | DLMF 19.7.5, 19.12; A&S 17.3.34–36 |
| `GammaP`, `GammaQ`, `GammaPInv`, `GammaQInv` | INCOMPLETE_GAMMA_PLACEHOLDER | |
| `BetaInc`, `BetaIncC`, `BetaIncInv`, `BetaIncCInv` | INCOMPLETE_BETA_PLACEHOLDER | |
| Bessel functions | BESSEL_PLACEHOLDER | |

Ported code keeps its attribution in a comment at the function: Cephes
(Stephen L. Moshier, BSD via SciPy) and Boost.Math (Boost Software
License 1.0).

## Negative control

A test suite that cannot fail proves nothing. `testdata/negative_control.sh`
perturbs one coefficient, runs `TestAccuracy`, and restores the file.

NEGATIVE_CONTROL_PLACEHOLDER

## Where SciPy's result is not a number it should be

Some SciPy "wrong" entries in the table are conventions, not errors; they are
listed so the table is not misread:

- `gammaln(5e-324)` and `gammaln(-5e-324)` return `inf` (exact: 744.44).
- `erfcinv(5e-324)` returns `inf` (exact: 27.2133).
- `betaln(a, b)` returns `nan` for 528 of the 1000 points with a, b in
  (100, 1e300] (exact values are finite, e.g. betaln(5.08e223, 7.77e195) =
  -5.05e197).
- Conventions: `gammaln(-inf)` is `-inf` in SciPy and `+inf` here (C99
  `lgamma`); `log_factorial(-1)` is compared with `gammaln(0) = inf` where
  this package returns NaN (n! is undefined for n < 0); `polygamma(1, -inf)`
  is `inf` in SciPy and NaN here.

## Known limitations

LIMITATIONS_PLACEHOLDER
