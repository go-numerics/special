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

**Budgets.** `TestAccuracyBound` fails if any region's maximum error exceeds its
budget (`budgets_test.go`: 1.5 times the error measured when the budgets
were generated, at least 1 ulp). The suite runs on amd64, arm64, 386, arm,
riscv64, loong64, ppc64le and s390x in CI.

**Same bits on every architecture.** Go may fuse `x*y + z` into one
multiply-add, and does on arm64, ppc64le, s390x, riscv64 and loong64 but
not on amd64, 386 or arm; package math's Exp, Log, Pow, Sin, ... also
differ by architecture (assembly versions, or pure Go compiled with
fusion). Every product here is therefore rounded explicitly
(`float64(a*b)`) or fused on purpose with `math.FMA` — `tools/fmalint`
checks the source, and its `--objdump` mode checks the arm64 binaries —
and those math functions come from `internal/libm`, Go's pure-Go
implementations with the same treatment. `TestArchHash` compares the
output bits of every public function on a fixed grid with
`testdata/archhash.golden` in every CI lane.

Regenerate:

```
uv run --with mpmath==1.4.1 --with scipy==1.18.1 --with numpy==2.5.4 \
    python testdata/gen/gen.py            # fixtures (minutes)
go test -run TestAccuracyTable -accuracy.table=docs/accuracy-table.md
go test -run TestAccuracyBudgets -accuracy.budgets=budgets_test.go
```

## Headline

Over 62 fixtures (419 regions, about 250 000 points), the maximum error of
this package is at or below SciPy's in **418 of 419 regions**; the one
exception is `k1` at small x, 1.05 ulp against SciPy's 1.05 (both use
Boost's K1 rational; ours is larger in the third digit).

- **About half an ulp** (correctly rounded or nearly) for the gamma
  family, erf/erfc/erfcx and their inverses, Ndtr/Ndtri/LogNdtr, zeta and
  Hurwitz zeta, polygamma, combinatorics, logistic helpers and elliptic
  integrals — including the tails, the zeros of log Γ at 1 and 2, the poles,
  the trivial zeros of ζ and the subnormal range, where SciPy's errors run
  from tens of ulps to 1e15 ulp or a wrong NaN/Inf.
- **Incomplete gamma** and its inverses: at most 2.1 ulp (SciPy: up to
  8.9e15). **Incomplete beta**: at most 9 ulp; its inverses at most 20 ulp
  for parameters >= 1e-3 (SciPy: up to 4.5e15).
- **Bessel functions**: at most about 5 ulp everywhere, 0.5 ulp at the zeros
  of J0, J1, Y0, Y1 (SciPy: 1e10–1e12 ulp there) and under 2 ulp for
  arguments up to 1e300 (SciPy loses every digit).
- **Not solved**: the beta inverses with a or b far below 1e-3, where the
  root depends on digits beyond double-double (both this package and SciPy
  can return a wrong root; the budget is +Inf and documented).

Bold in [accuracy-table.md](accuracy-table.md) marks a region where ours is
at or below SciPy; where both are "wrong" the row is bold too.

## Algorithms and sources

Two building blocks run through everything:

- **Double-double arithmetic** (`github.com/go-numerics/special/dd`, a
  public package since v0.3.0; fuzzed against `math/big` with the error bounds
  of the double-word algorithms: Add 3u², Mul 5u², Div 15u²+56u³, u = 2⁻⁵³):
  error-free transformations
  (Knuth's TwoSum, Dekker's Fast2Sum, FMA-based TwoProd) and the double-word
  algorithms of Joldes, Muller & Popescu (ACM TOMS 44(2), 2017). `dd.Exp`
  and `dd.Log` are table-driven after Tang (ACM TOMS 15(2), 1989 and 16(4),
  1990), carried to double-double (about 2^-76 and 2^-72 relative), with
  300-bit mpmath tables.
- **Float64 kernels** (`kernels.go`): table-driven exp and log after Tang
  in the arrangement of ARM's optimized-routines (exact Cody–Waite / TwoProd
  reductions, 400-bit tables from `go -C tools run ./genkernel`)
  returning an unevaluated hi + lo — exp to about 2^-68 relative, log to
  2^-68 absolute — at a few floating-point operations; the elementary
  families below use them instead of `dd` on their main paths,
  which remains for the tails and for the incomplete gamma/beta functions.
- **Fitted polynomials** (`testdata/gen/fit.py` and, for the tables added
  with the float64 kernels, the Go port `go -C tools run ./genfit`;
  `zz_*_coeffs.go`): Chebyshev
  interpolants of a well-chosen *regular part* of a function, fitted to
  mpmath at 200 bits on short pieces, degree chosen so that the
  interpolation error is below 2^-64, constant term kept in double-double.
  The function's zeros, poles and logarithmic singularities are factored
  out analytically, so the fitted part is smooth and the relative accuracy
  holds where the function vanishes.

| function | algorithm | source |
|---|---|---|
| `Gamma` | Γ(1+t) = 1 + t(t-1)F(t) on [-1/2, 1] with F fitted; exact-factor recurrence (x - k is exact), the product carried as hi + lo for 1 <= x < 10 and in double-double for 0 < x < 1 and -34 < x < 0 (there and in the reflection region, a float64 evaluation with a proven error bound returns the double-double path's float64 bit for bit whenever the bound decides the rounding: [gamma-fastpath.md](gamma-fastpath.md)); Stirling's series (DLMF 5.11.1) above 10, (x-1/2)log x - x + S(x) formed from the log kernel with every leading sum exact and S as hi + lo (about 2^-63 absolute), then the exp kernel; reflection with sin(πx) reduced exactly below -34; integers from a correctly rounded factorial table | DLMF 5.5, 5.11; core fitted to mpmath |
| `LogGamma`, `LogGammaSign`, `GammaSign` | log1p of Γ(1+t)-1 near the zeros at 1 and 2; the same Stirling expression above 10; exact factors for -34 < x < 0; reflection below | DLMF 5.11.1, 5.5.3 |
| `Digamma` | ψ(1+t) = (t - r)D(t) with the positive zero r in double-double and D fitted (the form of Boost.Math's `digamma_imp_1_2`); recurrence with double-double reciprocals; on [2, 10) ψ fitted directly (constant and linear terms exact); asymptotic series above 10 (DLMF 5.11.2) with log x from the log kernel; for -40 < x < 0, exact-factor recurrence, redone with a 2^-100 evaluation when the terms cancel (near the negative zeros); reflection with cot(πx) in double-double below | DLMF 5.5, 5.11; Boost.Math digamma.hpp |
| `Polygamma` | (-1)^(n+1) n! ζ(n+1, x); for n = 1, ψ'(x) directly: the asymptotic series from 10 with its two leading terms as hi + lo, the recurrence below with every 1/(x+k)² as hi + lo | DLMF 25.11.12, 5.15.8 |
| `Beta`, `LogBeta` | below 10, Γ(a)Γ(b)/Γ(a+b) from the double-double core with the rounding of a+b corrected to first order; above, log Γ(b) - log Γ(a+b) as one cancellation-free Stirling expression (-a log b - b log1pmx(a/b) - (a-1/2) log1p(a/b) + S(b) - S(a+b)) added to log Γ(a), or for 10 <= a <= b <= 64a the three Stirling log-gammas combined directly (bounded cancellation); the series of log1p(x) - x below x = 1/64 and S(b) - S(a+b) with the differences of powers factored; float64 kernels throughout. `LogBeta` for 10 <= a <= b <= 64a and a + b < 2048 reads the three log-gammas from the fitted table of `LogGamma` (extended to 2048; hi + lo, about 2^-57 absolute, the low part of the argument through ψ = a1 + 2 a2 s), within the half ulp of a result of at least 7 in absolute value; `Beta` keeps the Stirling evaluation (its exponential would turn that absolute error into a relative one). `LogBeta` output differs from v0.9.0 for 1.8e-5 of the points, 1 ulp at most | DLMF 5.11.1, 5.12.1 |
| `Erf`, `Erfc`, `Erfcx` | x E(x²) for \|x\| <= 1/2; erfcx fitted on [1/2, 8] and as x·erfcx(x) in 1/x² beyond (the Gaussian factor removed, as in Cody 1969); exp(-x²) from x² carried exactly (exp kernel), multiplied in before the single rounding; the pieces evaluated with their constant and linear terms exact. `Erf` (0.5 < \|x\| < 6) and `Erfc` (1/2 <= x <= 25, and -6 < x <= -1/2 through 2 - erfc(-x)) use instead a direct table: erfc(x) itself fitted on 673 pieces uniform in x² (step 1/4 below x = 4, 1 beyond; index min(int(4x²), int(x²)+48)), no exponential or erfcx, the constant to cubic terms exact (hi + lo) and only the quartic remainder rounded. Output differs from v0.7.0 for 3e-4 (`Erf`) and 1.3e-3 (`Erfc`) of the points, 1 ulp at most | Cody, Math. Comp. 23 (1969); DLMF 7.12 |
| `Ndtr`, `LogNdtr` | erfc of -x/√2 carried in double-double (the rounding of x/√2 is what costs SciPy up to 1e12 ulp in the lower tail); log Φ as log erfcx(z) - z² - log 2 in the lower tail, with log erfcx fitted directly (and as log(z erfcx z) in 1/z² beyond 8), and log1p(-Φ(-x)) in the upper. `Ndtr` is now a direct table in x: Φ(x) on pieces of width 1/8 for -3/4 <= x < 8.5, and Φ(-t) on 696 pieces uniform in t²/2 (as for `Erfc`) for 3/4 <= t <= 36, so no division by √2 and no exponential; the same Φ(-t) table gives the upper branch of `LogNdtr`; its lower tail for -2048 < x <= -8 is -x²/2 + L(-x), x² exact and L = log Φ(x) + x²/2 fitted on 8 pieces per octave. Output differs from v0.7.0 for 1.5e-3 (`Ndtr`) and 5e-5 to 2e-3 (`LogNdtr`) of the points, 1 ulp at most | DLMF 7.2, 7.12 |
| `ErfInv`, `ErfcInv` | evaluated directly, no iteration: erfinv(p) = p R(p²) for \|p\| <= 1/2 and erfcinv(q) = X(√(-log q)) on 20 pieces down to the smallest subnormal q, both fitted to mpmath; -log q from the log kernel and its square root carried as hi + lo. For q in [2^-53, 1/2) (all of `ErfInv` beyond 1/2, with q = 1 - \|y\| exact, and `ErfcInv`) erfcinv(q) is instead fitted directly in q on 416 pieces, 8 per octave (the piece is the top 15 bits of q's float64 pattern, degree <= 11), the constant and linear terms as hi + lo and one rounding: no logarithm, square root or division. Output differs from v0.8.0 for 2.5e-4 of the points, 1 ulp at most (the new value is the nearer to mpmath in 83% of the differences) | the variables of Boost.Math's erf_inv_imp and of Giles (GPU Computing Gems, 2011); fits to mpmath |
| `Ndtri` | q = min(p, 1-p) (1-p exact for p >= 1/2); for q in [2^-29, 1/2], (q - 1/2) G(q) with G(q) = ndtri(q)/(q - 1/2) fitted to mpmath on 8 pieces per octave of q (the piece is the top 15 bits of q's float64 pattern; 224 pieces, degree <= 16), G and q - 1/2 carried as hi + lo, no logarithm, square root or division; below 2^-29 -√2 erfcinv(2p) as for `ErfcInv`, with √2 a double-double. Output differs from v0.6.x for 5.9e-4 of the points (1 ulp at most, both kernels within 0.5 ulp of mpmath on the corpus) | same fits to mpmath |
| `Zeta` | 1/(s-1) + R(s) with R entire and fitted on [-1, 6] (exact pole); 1 + g(s)2^-s with g = (ζ(s)-1)2^s fitted on (6, 64); reflection with sin(πs/2) reduced exactly below -1 (trivial zeros exact) | DLMF 25.4.1 |
| `HurwitzZeta` | Euler–Maclaurin summation (the arrangement of Cephes `zeta.c`) with q+k exact in double-double, powers through double-double log/exp and terms summed with a common exponent (no intermediate overflow); the number of direct terms chosen so 15 Bernoulli terms reach 2^-64; for 1 < s <= 64 the float64 kernels with 25 Bernoulli terms and the shorter shift w >= 7.3 + s/2 (bound computed with mpmath, checked by a test) | DLMF 25.11.5; Moshier, Cephes |
| `Comb`, `Perm`, `Binom`, `Factorial`, `Factorial2`, `LogFactorial` | products in double-double up to 64 factors (exact below 2^53, correctly rounded above); beyond, through `LogBeta`'s cancellation-free logarithm | — |
| `Softmax`, `LogSoftmax`, `LogSumExp` (since this change; the three were correctly rounded, in double-double, before) | m = max x; x_i - m exactly as h_i + l_i (Knuth's TwoSum: the rounded difference would cost \|h\| 2^-53, up to 199 u measured, the error of NumPy's and SciPy's softmax); e_i = exp(h_i)(1 + l_i) by `floats.Exp` (Arm's optimized-routines exp, 0.511 ulp) and one FMA; Σ e_i compensated (four TwoSum accumulators started at 0: an earlier Fast2Sum started at 1 added every term below 2^-53 plain once the maximum was left out, 32.7 u at n = 2e4, found by the independent review and fixed; the maximum is counted, not summed, for `LogSoftmax` and `LogSumExp`, so that log1p receives Σ_{x_i < m} e_i in full relative accuracy when one term dominates); log1p in double-double as before. With u = 2^-53: **Softmax** relative error at most **8u** for every result at or above 2^-1022 (e_i 2.03u, the sum 2.03u, its rounding u, the reciprocal u, the product u; absolute 2^-1073 below), **LogSoftmax** at most **5u** ((x_i - m) and log Σ have the same sign, so nothing cancels: the sum's 2.03u acts through log1p with gain at most 1, and \|y_i\| >= L), **LogSumExp** at most **4u** where \|lse\| >= L = log Σ e_i (all m >= 0 and most m < 0); where m + L cancels (m in (-2L, 0)) the double-double evaluation (`LogSumExpExact`) answers, with NO relative bound: its absolute error is below 2^-84 L + 2^-53 \|lse\| (log1pDD is within 2^-85.6 of L), which as a relative error is 4.5e6 u at x = [-ln 2, -ln 2] (lse = 2.3e-17). **Region of all three bounds**: no term that matters is subnormal, i.e. the sum of the others is 0 or at least about n·2^-1000 (below that the terms under 2^-1022, each with an absolute error up to 2^-1075, carry the relative accuracy away: x = [0, 2^20 × -720] gives lse = 2.13e-307 at 26298 u, [0, 2^20 × -715] 707 u); the same two limits hold for the `*Exact` functions, which are NOT correctly rounded there. Both are open (issue linked from the pull request). Measured against a 320-bit math/big oracle (independent of every code path) and the double-double code on 9 shapes (wide, huge, masked, ties, dominant ...) and 18 sizes: 3.9u, 2.1u, 1.0u; the constant tails [0, 999 × -40] and [0, 2e4 × -41.598...] at 0.61u and 0.42u. The independent review's own 320-bit oracle and adversarial search (constant tails up to n = 1e5, the -36.7 cutoff ± 1, a 3000-input worst-tail search) found at most 3.07u for Softmax and 1.87u for LogSoftmax and LogSumExp, inside 8u, 5u and 4u. `SoftmaxExact`, `LogSoftmaxExact`, `LogSumExpExact` are the v0.19.0 double-double code under new names (half an ulp inside the region above, bit-identical on every architecture, about 4.5 times slower), kept for where the last bit or the architecture independence matters. The bits depend on the architecture (`floats.Exp` is left open to fusion), so the three left the `archhash` grid; the bound is asserted on every CI lane. NOT proven: the 0.511 ulp of `floats.Exp` (documented by Arm, measured by go-simd/floats against a 256-bit reference, not derived here). Derivation: `softmax_fast.go` | Arm optimized-routines exp.c (MIT); Knuth TAOCP 2, 4.2.2; Dekker 1971 |
| `Expit`, `LogExpit`, `Log1pExp`, `Logit`, `Xlogy`, `Xlog1py` | the exponential and logarithm from the float64 kernels as hi + lo (double-double in the tails and for the vector functions), quotients corrected by their FMA residual; log1p of a tiny argument by its series. `Logit` is a direct table instead: q = min(p, 1-p), logit(q) = (q - 1/2) G(q) with G fitted to mpmath on 8 pieces per octave of q on [2^-29, 1/2] (as `Ndtri`), G and q - 1/2 as hi + lo, one rounding, no logarithm or division; below 2^-29 the logarithm of p/(1-p) as hi + lo. Output differs from v0.9.0 for 1.1e-4 of the points, 1 ulp at most (the new value is the nearer to mpmath in 311 of 344 differences) | — |
| `EllipK`, `EllipKm1`, `EllipE` | for p = 1 - m in [2^-10, 2^11): degree-11 interpolants on 8 pieces per octave of p, in s = m - c (exact by Sterbenz), constant term in double-double, one rounding (zz_ellipOct_coeffs.go, testdata/gen/fitgo); outside, K(1-p) = A_K(p) - log(p)K(p)/π and E(1-p) = A_E(p) - log(p)(K(p)-E(p))/π (the structure of DLMF 19.12, of Cephes `ellpk`/`ellpe` and of Hastings' approximations) with A_K, A_E, K, E and (K-E)/p fitted; m < -1 by the imaginary-modulus transformation | DLMF 19.7.5, 19.12; A&S 17.3.34–36 |
| `GammaP`, `GammaQ`, `GammaPInv`, `GammaQInv` | method selection of DiDonato & Morris as in Boost.Math's `gamma_incomplete_imp` and Cephes `igam`: power series (terms in light double-double pairs while above 2^-16 of the sum), Legendre's continued fraction (depth from the Wallis recurrences, value summed backwards as P/Q, top level in double-double), Temme's uniform expansion for a > 20 near x = a (its sum generated straight-line, testdata/gen/fitgo), small-a upper-tail series; the complement formed from the double-double tail; the prefactor x^a e^-x/Γ(a) formed as a double-double logarithm (a·log1pmx((x-a)/a) and Stirling, no cancellation for any a) and exponentiated once. Inverses: DiDonato–Morris starting value (Boost `find_inverse_gamma`), then a bracketed Halley iteration in log x on log of the smaller tail against its exact target (one float64 evaluation of the tail per step, g = log1p((T-t)/t) near the root) | DiDonato & Morris, ACM TOMS 12(4) 1986; DLMF 8.7, 8.9, 8.11, 8.12; Boost.Math gamma.hpp, igamma_inverse.hpp (BSL-1.0); Cephes igam.h |
| `BetaInc`, `BetaIncC`, `BetaIncInv`, `BetaIncCInv` | TOMS 708 (BRATIO with BPSER, BUP, BFRAC, BGRAT, BASYM, APSER, FPSER) ported from SciPy's cdflib.c, with every power prefix x^a y^b/B(a,b) formed as a double-double logarithm, BASYM's λ = (a+b)y - b in double-double; BPSER in Euler's all-positive form (DLMF 8.17.8), shared with BUP, in light double-double pairs; BFRAC's fraction summed backwards from the depth its forward pass finds, the top level from the closed forms in double-double. Inverses: Temme's first method or the normal/power-law starting value, the side of 1/2 shown by the iteration; bracketed Halley in log x or log(1-x) on the smaller tail | DiDonato & Morris, ACM TOMS 18(3) 1992; Temme, JCAM 41 (1992); Boost.Math ibeta_inverse.hpp |
| `J0`, `J1`, `Y0`, `Y1` | power series near 0 for Y (log in double-double); up to x = 100, one polynomial per cell centred on a zero of the function, of its derivative or a midpoint between them: f(z+h) = a0 + a1 h + h²R(h), a0, a1 and z in double-double from mpmath, R a degree-11 Chebyshev interpolant (zz_besselCells_coeffs.go, testdata/gen/fitgo), combined with error-free transformations; beyond, Hankel's expansion in modulus–phase form with the phase in double-double and an exact Payne–Hanek reduction of x mod π/2 | Harrison, ARITH 2009; DLMF 10.2, 10.8, 10.18 |
| `Jv`, `Yv`, `Jn`, `Yn` | Temme's series below x = 1/4 (1/Γ Taylor coefficients, A&S 6.1.34); Steed's CF1/CF2 method with both fractions (evaluated backwards in light double-double pairs), the normalisation and the recurrences in double-double, at the order itself when x >= v (no recurrence through the oscillatory region, no reflection); modulus–phase expansion for large x; Jn/Yn avoid the forward recurrence that loses the zeros | Temme 1976; Barnett et al. 1974; Boost.Math bessel_jy.hpp (BSL-1.0) |
| `Jn` (n >= 2), `Yn` (n >= 2), 0 < x < 100 | hi + lo recurrence DLMF 10.6.1 in dd-light form (hi chain with error-free products and sums, linear lo chain) from J0, J1, Y0, Y1 as unrounded pairs: `Yn` upward from Y0, Y1; `Jn` upward for n <= x, Miller's downward recurrence from the CF1 ratio (last 8 levels in double-double) for n > x, normalised by J0 or J1. Each answer carries a certificate: the start errors and step roundings are carried through the Casoratian (DLMF 10.5.2) as A·J_n + B·Y_n, with a priori bounds from DLMF 10.14.1, 10.14.4, and the path answers only where the bound is below 6 ulp (Yn) / 4 ulp (Jn); otherwise the row above answers, bit for bit (about 1% of the refbench inputs). Error <= 6.5 ulp (Yn), 4.5 ulp (Jn); measured 1.40 and 1.04 on the rows answered (derivation: `jn_yn_fast.go`) | DLMF 10.5.2, 10.6.1, 10.14.1, 10.14.4 |
| `I0`, `I1`, `K0`, `K1`, `I0e`, `I1e`, `K0e`, `K1e` | I0e (x in [1/8, 512)) and K0e (x in [1, 512)): degree-11 interpolants on 8 pieces per octave fitted to mpmath (testdata/gen/fitgo), I0 and K0 as e^±x from dd.ExpMant times them, rounded once; I0 below 1/8 by its series. Since v0.5.6 the same for I1e (x in [1/8, 512)) and K1e (x in [1, 512)), with I1 and K1 as e^±x times them; I1 below 1/8 by a double-double series. K0 and K1 on (0, 1]: Boost's 53-bit minimax rationals; DLMF 10.40 asymptotics above 20 (512 for order 0) with the exponential reduced in double-double | Boost.Math bessel_i0/i1/k0/k1.hpp; DLMF 10.40 |
| `Jv`, `Yv` (non-integer v, \|v\| <= 40, 20 < x < 2^500; `jv_hankel_fast.go`) | Hankel's P, Q form DLMF 10.17.3-4, J = a(P cos w - Q sin w), Y = a(P sin w + Q cos w), a = sqrt(2/(πx)), w = x - (v/2 + 1/4)π, both signs of v. The terms u_k = a_k/x^k (DLMF 10.17.1) in double-double while above 2^-18, float64 below, with a running rounding bound; truncation bounded rigorously by the first neglected term (DLMF 10.17(iii), l >= max(⌈\|v\|/2⌉, 2)); w reduced exactly (Payne–Hanek), sin and cos of the reduced angle as hi + lo (`hankelSinCos`, error < 2^-59, math/big oracle on 1e6 angles); one rounding. The path answers only where the total bound E <= 2^-56 \|F\| (F = P cos w - Q sin w): then the result is within 0.63 ulp. Next to a zero of J or Y, \|F\| is small, the test fails and the modulus–phase code of the row above answers, bit for bit: no relative bound is claimed there (the absolute bound is E·a, a the amplitude). Answers 88% of v in [0, 20), x in (20, 50]; measured 0.505 ulp (math/big oracle, 1e5 answers). Derivation: `jv_hankel_fast.go` | DLMF 10.17.1, 10.17.3, 10.17.4, 10.17(iii) |
| `Jv`, `Yv` (non-integer v, \|v\| <= 40.5, 1 <= x <= 20; `jv_mid_fast.go`) | the two power series of DLMF 10.2.2, J_w = (x/2)^w/Γ(w+1) Σ t_k, t_k = -(x²/4) t_(k-1)/(k(k+w)), for w = ±\|v\|, in double-double: the factor of a step is hi + lo from one division and its exact residual, the term a pair updated with the exact error of its product, so the chain through the term is one multiplication per step; Y from DLMF 10.2.3 with 1/Γ(1-ν) = Γ(1+ν) sin(νπ)/(νπ) (Y_ν = J_ν cot νπ - B, Y_-ν = J_ν/sin νπ - B cos νπ, B = S(-ν)/(πν P)), sin and cos of νπ reduced exactly and summed in double-double. The prefix P = (x/2)^ν/Γ(ν+1): Γ shifted to an argument >= 10 by exact factors, the shift's rounding corrected to first order, Stirling's series (DLMF 5.11.1) with its first term in double-double. No table (the Bernoulli numbers and Taylor coefficients are computed at init). Every answer carries a bound: truncation rigorous (ratios of the terms decrease once k + w > 0, tail <= 2r\|t_k\|), series rounding n 2^-100 + n² 2^-105 of Σ\|t_k\| (derived by hand, held against math/big on 2e4 draws), prefix 2^-68 relative (measured 2^-69.4 on 3e4), sin/cos 2^-80 relative (measured 2^-86), the double-double assembly 2^-100. The path answers only where E <= 2^-57 \|F\|, hence within 0.5625 ulp; the series cancels like e^x (up to 2^36 against the value), which the bound pays for. Next to a zero of J or Y, and for Y next to an integer order, it declines and the rows above answer, bit for bit (checked against b4e5a09 = v0.18.0 on 2.4e5 values, `jv_mid_bits_test.go`). Measured 0.5000 ulp on 1.2e5 answers against the math/big oracle (J 59997 of 60000 draws answered, Y 59803) and 0.52 on the mpmath rows (`jv_mid`, `yv_mid`). NOT proven: the per-step 2^-100 and the prefix and sin/cos bounds (held by the oracle, not derived to the last bit), the dd package's operation bounds are taken from its tests. Derivation: `jv_mid_fast.go` | DLMF 10.2.2, 10.2.3, 5.5.3, 5.11.1, 5.11(ii), 4.19 |
| `Kn` (\|n\| >= 2, 1 <= x < 512, while the scaled chain stays finite: no fixed cap on n, about n = 929 at x near 512) | upward recurrence DLMF 10.29.1 on K0e, K1e as hi + lo pairs, hi chain with error-free products and sums plus a linear lo chain, scaled by e^-x at the end; all terms positive, so no cancellation; error <= 0.55 ulp, measured 0.52 (derivation: `kn_fast.go`); elsewhere as the row below | DLMF 10.29.1 |
| `Kv`, `Kve` (v >= 32, x < 1e6; 2 < v < 32 with x >= 24 where 4v² - 1 > 4x; 0 < v < 32 with 1.5 <= x < 24; and 0 < v < 32 with 2^-1000 <= x < 1.5) and integer v of `Kv`/`Kve` through the `Kn` chain (1 <= x < 512) | the scheme of Bessels.jl's `besselk` (MIT) with this package's accuracy. v >= 32: Debye's uniform expansion DLMF 10.41.4 with the exponent v·η = S + v ln(x/(v+S)), S = sqrt(v² + x²), in double-double (float64 would lose about 700 ulp there), the prefactor sqrt(π/2S) in double-double, the polynomials U_k (k <= 16) from the exact recurrence DLMF 10.41.9 (`testdata/gen/kvdebye`), the sum k >= 1 in float64 (below 1/(12v) of the result). 2 < v < 32, x >= 24: the large-argument expansion DLMF 10.40.2 of the orders u and u + 1 (u = frac v; first term in double-double, truncation bounded by the first neglected term, DLMF 10.40(ii)), then the `Kn` hi + lo recurrence DLMF 10.29.1 with the order u + k as an exact pair, scaled by e^-x at the end. Error <= 0.6 ulp (stated), measured 0.51 on every row answered (derivation: `kv_fast.go`; the Debye truncation is checked on the fixtures, not proven: Olver's remainder bound was not opened). 0 < v < 32, 1.5 <= x < 24 (`kv_cf2_fast.go`): u = v - round(v), the pair K_u e^x, K_(u+1) e^x by Steed's CF2 (Thompson-Barnett) in two phases (the first 2-4 steps in double-double, the rest in float64, every term positive: no cancellation), K_(u+1)/K_u = (x + 1/2 + u + a_1 f)/x, then the `Kn` hi + lo recurrence and e^-x. The tail of f is bracketed rigorously (the tail of the fraction is at most k + 1/2 and the value increases with it); the truncation of S is NOT proven (the stopped pair is within 2^-61.5 of the loop run to 2^-84, on 1e5 points). Start pair within 2^-58.9 (K_u) and 2^-57.4 (K_(u+1)) of a math/big oracle on 1.2e5 points; Kv and Kve 0.52 ulp on the same sweep, 0.50 on the fixtures (was 1.9). 0 < v < 32, 2^-1000 <= x < 1.5 (`kv_temme_fast.go`, 0.6 ulp stated, 0.500 measured on the 2845 fixture rows and on the 1.2e5-point oracle sweep below; was up to 3.9 ulp): u = v - round(v), K_u and K_(u+1) by Temme's series (Temme 1975; the recurrences of Boost's `temme_ik` and Bessels.jl `besselk.jl`, MIT) with f_0, p_0, q_0 in double-double, the first 0 to 4 terms (by x) in double-double and the rest in float64, then the `Kn` hi + lo recurrence on the unscaled values. The limit u -> 0 needs no division: Gamma_1 = (1/Gamma(1-u) - 1/Gamma(1+u))/(2u) and Gamma_2 = (1/Gamma(1-u) + 1/Gamma(1+u))/2 are Taylor series in u² (`zz_kvTemme_coeffs.go`, from 1/Gamma(1+u) = sum B_m u^m, `testdata/gen/kvtemme`, 300 bits), sinh(σ)/σ and cosh(σ) are series for σ of absolute value below 2^-6, and sin(πu)/(πu) is 1/(Γ(1+u)Γ(1-u)). The truncation is bounded rigorously: for j > k >= 1, f_j + 4 max(p_j, q_j) falls by a factor k per step (z_j <= z_(j-1)/k), so both tails are geometric (kvTemmeTail), stopped at 2^-64 of the partial sums; the cancellation of the Γ pieces is of absolute size 2^-100 against K >= 0.21, and the logarithm (2^-72) moves σ = u ln(2/x) by at most 2^-63.5. Start pair within 2^-64.7 (K_u) and 2^-62.0 (K_(u+1)) relative of a math/big oracle on 1.2e5 points; Kv and Kve 0.500 ulp on the 105611 of them the path answers. f_k > 0 for k >= 1 is proven in the header of `kv_temme_fast.go` (f_k = (p_k - q_k)/u). NOT proven: the decay of the Gamma coefficients beyond the 14 kept (checked against ln Gamma at 320 bits). The path answers only where K_v < 2^1020 and declines above (it never returns +Inf itself; besselIK answers and returns +Inf where the true value overflows). Declined (exponent in the 700-760 band, result below 2^-1000, x >= 1e6, x < 2^-1000, K_v of 2^1020 or more, or the single series of besselAsymIK at x >= 24): the row below, bit for bit (checked against c267f26, the commit of v0.16.0, on 2.4e5 inputs, `kv_bitcheck_test.go`) | DLMF 10.40, 10.41.4, 10.41.9, 10.29.1; Temme 1975; Thompson & Barnett 1986; Bessels.jl besselk.jl (MIT) |
| `Iv`, `In`, `Ive` (order > 0 or integer, x > 0, positive or integer order; negative non-integer orders and x < 1/8 of integer order stay on the row below) | the scheme of Bessels.jl's `besseli` (MIT) with this package's accuracy, carried as I e^-x in hi + lo pairs and rounded once. v >= 32 (<= 1100, x < 1e6): Debye DLMF 10.41.3, as for `Kv`. x >= 24: the large-argument expansion DLMF 10.40.1, accepted only with a running bound on the float64 rounding of its terms (<= 2^-57 of the sum). Otherwise u = frac v, n = floor v: the downward recurrence DLMF 10.29.1 started at a level N where the bound prod g_j g_{j+1} (g_j = min(1, x/2j), from the monotonicity DLMF 10.37 and the power series DLMF 10.25.2) is below 2^-63, normalised by I_u e^-x (the I0e octave table for u = 0, the series DLMF 10.25.2 for x < 24, the expansion beyond). Error <= 0.6 ulp (stated), measured 0.53 (derivation: `i_fast.go`; the Debye and large-argument truncation are checked against a math/big oracle, not proven). Declined (exponent band, unscaled x > 700, start level beyond 600, chain overflow at tiny x): the row below, bit for bit | DLMF 10.25.2, 10.29.1, 10.37, 10.40.1, 10.41.3; Bessels.jl besseli.jl (MIT) |
| `Iv`, `Ive`, `Kv`, `Kve`, `In`, `Kn` | scaled throughout: Temme 1975 below 0.5, Thompson–Barnett CF2 in double-double (to 2^-64 of the sum), CF1 + Wronskian for I, power series; DLMF 10.27.4 series for K at small x; reflection DLMF 10.27.2 for negative order; overflow detected from Γ(v)(2/x)^v/2 | Temme 1975; Thompson & Barnett 1986; Boost.Math bessel_ik.hpp |
| `JnZeros` | McMahon (n = 0) or Olver's uniform expansion with Airy zeros (n >= 1) as the guess, Halley's iteration | DLMF 10.21.19; A&S 9.5.22, 10.4.94; Boost.Math bessel_jy_zero.hpp |

Ported code keeps its attribution in a comment at the function: Cephes
(Stephen L. Moshier, BSD via SciPy) and Boost.Math (Boost Software
License 1.0).

## Negative control

A test suite that cannot fail proves nothing. `testdata/negative_control.sh`
perturbs one coefficient, runs `TestAccuracyBound`, and restores the file.

Results (run on this branch, arm64):

| perturbation | size | result |
|---|---|---|
| low part of the constant term of the Γ(1+t) core on [0, 1/4] (`c0lo`, zz_gamma1pPieces_coeffs.go) | 3.7e-16 absolute (~1.5 ulp of that coefficient) | **fails**: 12 regions over gamma, beta, binom, factorial, log_factorial (7.4 ulp), logbeta (32 ulp) |
| Stirling coefficient B₆/(6·5) (`c3 = 1.0/1260`, gamma.go) | 8e-7 relative | **fails**: 7 regions, gamma [2,33) 48.9 ulp, beta 70 ulp, binom, factorial, log_factorial |
| same coefficient | 8e-11 relative | passes: its effect (≈ 6e-19 on log Γ(10)) is below every result's last bit — a perturbation the output cannot see is not one a test can catch |
| one coefficient of the erfcx interpolant on [0.5, 1] (zz_erfcxPieces_coeffs.go) | 1e-12 relative | **fails**: erf 14 ulp, erfc 25.6 ulp, erfcx 29.7 ulp, erfcinv 16.3 ulp |

The budgets are tight enough that an error of a few ulps in one coefficient
of one piece is caught, through every function built on it.

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

- **Speed.** Double-double costs a factor of about 3 single-threaded for
  most functions and 10–15 for the Bessel functions of real order, Hurwitz
  zeta and the incomplete beta function ([perf.md](perf.md)).
- **Digamma at its negative zeros**: 31 ulp at 1e-16 relative distance from
  a zero (SciPy: 2.5e17); limited by double-double itself.
- **LogBeta where log B(a, b) ≈ 0** (a, b near 1): 30 ulp; the three
  log-gammas carry ~2^-56 relative errors that the cancellation exposes
  (SciPy: 45 000).
- **Bessel**: Jn/Yn within 5.4 ulp at the zeros of orders >= 2; Jv at
  negative non-integer order and x < |v| 7.2 ulp, Ive at negative order
  12 ulp (both from the reflection formula's cancellation); K1/K1e at small
  x tie with SciPy (1.05/1.49 ulp, Boost's rational unchanged); orders above
  1e7 only where the series or the large-x expansion applies (NaN
  elsewhere); J and Y at x = +Inf return the limit 0 where SciPy returns
  NaN.
- **Incomplete beta**: up to 5 ulp for a, b in [1, 100) and 9 ulp for
  a, b in [100, 1e4) near the mean in this table (SciPy: 447 and 3487);
  BetaIncCInv 20 ulp for a, b in [1e-3, 1) (51 on amd64; SciPy 4.5e15).
  These are the fixtures' measurements before the backward BFRAC and the
  positive BPSER series; a dense re-check (2000 points per region) gives
  1.05 and 3.1 ulp and 1.2 ulp for those three; the table has not been
  regenerated.
- **Beta inverses with a or b far below 1e-3**: I_x(a, b) is flat to beyond
  2^-100 over most of (0, 1), so the float64 root depends on digits past
  double-double; results can be wrong (so are SciPy's). Budget +Inf,
  documented at BetaIncInv.
- **Oracle caveat**: mpmath's own `betainc` returned an identical wrong
  value at both precisions for b = 1e-200 (the self-check cannot catch an
  error that does not depend on precision), so the incomplete-beta oracles
  compute the smaller tail by series or tanh-sinh quadrature instead.
- **Complex arguments** are not implemented (the `nd` layer rejects complex
  dtypes).
- **Not yet implemented**: Struve, Airy, Kelvin, spherical Bessel,
  incomplete elliptic integrals, Lambert W.
