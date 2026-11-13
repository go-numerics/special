<p align="center"><img src="https://raw.githubusercontent.com/go-numerics/brand/main/social/go-numerics.png" alt="go-numerics/special" width="720"></p>

# special

[![Docs](https://img.shields.io/badge/docs-hugo%20%2B%20hextra-7445AD)](https://go-numerics.github.io/docs/latest/special/)
[![pkg.go.dev](https://img.shields.io/badge/pkg.go.dev-special-007d9c?logo=go&logoColor=white)](https://pkg.go.dev/github.com/go-numerics/special)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-7445AD)](LICENSE)

Special mathematical functions in pure Go (no cgo, no assembly): the gamma
family, error functions, incomplete gamma and beta functions and their
inverses, Bessel functions, zeta functions, combinatorics, logistic helpers
and complete elliptic integrals — with an element-wise layer over
[go-ndarray](https://github.com/go-ndarray/ndarray).

🌐 **[Website](https://go-numerics.github.io)** · 📚 **[Documentation](https://go-numerics.github.io/docs/latest/special/)**

It is the numerical foundation of the go-numerics statistics library: every
distribution's cdf and quantile function is computed with these functions,
so their accuracy is the product. Each function is measured in ULPs against
[mpmath](https://mpmath.org) at high precision, over its whole domain —
tails, zeros, poles, huge and tiny arguments, subnormals, NaN and ±Inf — and
SciPy is measured on exactly the same points.

```
go get github.com/go-numerics/special
```

```go
package main

import (
	"fmt"

	"github.com/go-numerics/special"
)

func main() {
	fmt.Println(special.Gamma(4.5))              // 11.631728396567448
	fmt.Println(special.Erfc(10))                // 2.088487583762545e-45
	fmt.Println(special.Ndtr(-30))               // 4.906713927148187e-198
	fmt.Println(special.Ndtri(1e-300))           // -37.0470962993612
	fmt.Println(special.GammaP(2.5, 1.5))        // regularised lower incomplete gamma
	fmt.Println(special.BetaIncInv(2, 3, 0.975)) // a beta quantile
	fmt.Println(special.Jv(2.5, 10))             // Bessel function of real order
}
```

Arrays (`github.com/go-numerics/special/nd`): any real dtype in, Float32 kept
as Float32, everything else Float64; NumPy broadcasting; large arrays are
split across cores.

```go
x, _ := ndarray.Linspace(-5, 5, 1_000_001)
p, _ := nd.Ndtr(x)
```

## Accuracy

Measured on about 250 000 points in 419 regions over 62 functions, this
package's maximum error is at or below SciPy's in 418 regions (the
exception, K1 at small x, is a tie at 1.05 ulp). Typical maxima:

| | this package | SciPy 1.18.1 |
|---|--:|--:|
| Gamma, LogGamma, Beta (all regions) | 0.5–1 ulp | 3–3e15 ulp |
| Digamma near its negative zeros | 31 ulp | 2.5e17 ulp |
| Erfc to the subnormal range | 0.5 ulp | 2e13 ulp |
| Ndtr lower tail, x in [-38.5, -10) | 0.5 ulp | 1.8e12 ulp |
| Ndtri, ErfInv, ErfcInv | 0.5–0.6 ulp | 1.4–4.4 ulp (wrong at subnormals) |
| GammaP / GammaQ, a in [1e4, 1e15] | 0.6 ulp | 8.9e15 ulp |
| BetaInc, a and b in [100, 1e4) | 9 ulp | 3487 ulp |
| J0 / Y0 next to their zeros | 0.5 ulp | 3e10 / 2e12 ulp |
| Jv at x in [1e4, 1e300] | 1.7 ulp | 5e18 ulp |

The one area not solved: inverses of the incomplete beta function with a
or b far below 1e-3 (both packages can be wrong there).

Full method, sources of every algorithm, negative control and limitations:
[docs/accuracy.md](docs/accuracy.md); every region:
[docs/accuracy-table.md](docs/accuracy-table.md).

## Accuracy contract

Bound: for every function and every region, the maximum error in ulps of the result is at most the region's budget (explicit budgets, e.g. `incbeta_test.go`: the measured maximum times 1.3, rounded up; cases without one, `budgets_test.go`: 1.5 times the measured maximum, at least 1 ulp; typically 0.5 to 1 ulp, 15 ulp for BetaInc and BetaIncC at a and b in [100, 1e4), 48 ulp for Digamma next to its negative roots), and NaN, ±Inf and ±0 are returned where the function has them. **Unbounded today** (budget +Inf, the root is not determined by double-double evaluation of the function): BetaIncInv and BetaIncCInv for a, b < 1e-3 and for exactly one of a, b below 1e-3 (four regions); bounding them is open work (Ref #11, and "Beta inverses with a or b far below 1e-3" in `docs/accuracy.md`).
Oracle: mpmath 1.4.1 at 192 and 320 bits (points where the two disagree by more than 2^-80 are dropped and counted), stored as a double-double `hi + lo` and compared exactly, so a correctly rounded result scores 0.5 ulp; SciPy is measured on the same points but is never the reference.
Exact: the bit-exact build stays and is tested: every product is rounded explicitly (`tools/fmalint`), `TestArchHash` pins the output bits of every public function on 8 architectures, and the vector forms (`nd`, `…To`) equal the scalar ones element by element. A faster kernel (FMA, a reordered sum, another algorithm) lands when it is faster and within Bound; the golden hashes are then regenerated as an explicit step, never as a reason to keep a slower kernel.

The bound is asserted by `TestAccuracyBound` in `accuracy_test.go` over the fixtures of every function of the root package, on every CI lane; the complex functions (`cmplx`) keep their own gate, `TestAccuracy` in `cmplx/harness_test.go`, with its own budgets. Bit identity is how a kernel is ported and checked, not what the package promises: the promise is the accuracy table.

## Speed

Accuracy has a price. Measured against SciPy 1.18.1 on one core (refbench
v0.3.0, tag v0.13.0, 2026-11-13, 10 000-point calls, host load 6.5–8.0, valid
run (151 rows, 11 rounds retried above the gate); ratios are Go time over SciPy
time):

- faster than SciPy: Gamma 0.84 (8.5 vs 10.1 ns/element; `nd.Gamma` 0.85),
  LogGamma 0.76, LogBeta 0.58, ErfInv 0.49, ErfcInv 0.62, Ndtri 0.57, LogNdtr
  0.59, Ndtr 0.84, Erf 0.89, Digamma 0.80, Beta 0.84, the order 0 and 1 Bessel
  functions (J0 0.74, I0e 0.30, K1 0.93, K1e 0.37), the elliptic integrals
  (0.37–0.43), Zeta 0.64, Polygamma 0.05, Comb 0.85, Factorial2 0.03, and the
  integer orders `Jn` 0.18 (130 vs 736 ns/element) and `Kn` 0.28 (102 vs 363
  ns);
- ties: Logit 0.90, I1 0.97, Erfc 1.03, `nd.Erfc` 1.03 (`nd.I1` measured 1.52,
  unexplained; rerun needed);
- still slower, 1.08–1.54x: LogExpit 1.08, Xlogy 1.08, GammaQ 1.13, GammaP
  1.18, Expit 1.20, GammaSign 1.22, Perm 1.22, Factorial of non-integers 1.22,
  Yv 1.22, Jv 1.28, LogSumExp 1.47, GammaQInv 1.48, GammaPInv 1.54;
- several times slower, where a double-double continued fraction or series
  produces the rounded result: Iv 7.9, In 5.2, Softmax 4.8, LogSoftmax 4.1,
  BetaIncInv 3.5, Binom 3.0, Yn 2.37 (91 vs 38 ns, from 33 in v0.8.0), Kve
  2.4, Kv 2.3, BetaInc 2.1, Xlog1py 1.9, Ive 1.69 (open defects, #33).

Since v0.5.6, `I1`, `I1e`, `K1` and `K1e` use the same fitted octave
interpolants as `I0e` and `K0e` (and are more accurate: worst case against
mpmath 3.45 to 0.51 ulp for I1); `Comb`, `Factorial2`, `Factorial` and
`Xlog1py` have faster paths that return the same bits as before.

Method and full table: [docs/perf.md](docs/perf.md).

## Functions

Each row gives the Go function and, in brackets, the `scipy.special` function it
is **measured against** — the pairing comes from the harness
(`bench/perf_test.go` and `testdata/gen/fam_*.py`), not from transliterating
a name, which is why `Log1pExp` is scipy's `softplus` and `LogGamma` is
`gammaln`. 64 of the 67 carry such a pairing; `—` marks the three that do not,
and the notes under the table say what they are. `readme_scipy_test.go` reads
the harness and fails if this column drifts from it.

| family | functions (scipy equivalent) |
|---|---|
| gamma | `Gamma` (`gamma`), `LogGamma` (`gammaln`), `LogGammaSign` (—), `GammaSign` (`gammasgn`), `Digamma` (`digamma`), `Polygamma` (`polygamma`), `Beta` (`beta`), `LogBeta` (`betaln`) |
| incomplete gamma / beta | `GammaP` (`gammainc`), `GammaQ` (`gammaincc`), `GammaPInv` (`gammaincinv`), `GammaQInv` (`gammainccinv`), `BetaInc` (`betainc`), `BetaIncC` (`betaincc`), `BetaIncInv` (`betaincinv`), `BetaIncCInv` (`betainccinv`) |
| error function, normal | `Erf` (`erf`), `Erfc` (`erfc`), `Erfcx` (`erfcx`), `ErfInv` (`erfinv`), `ErfcInv` (`erfcinv`), `Ndtr` (`ndtr`), `Ndtri` (`ndtri`), `LogNdtr` (`log_ndtr`) |
| combinatorics | `Comb` (`comb`), `Perm` (`perm`), `Binom` (`binom`), `Factorial` (`factorial`), `Factorial2` (`factorial2`), `LogFactorial` (`gammaln(n+1)`) |
| logistic, log helpers | `Expit` (`expit`), `LogExpit` (`log_expit`), `Logit` (`logit`), `Log1pExp` (`softplus`), `Xlogy` (`xlogy`), `Xlog1py` (`xlog1py`), `LogSumExp` (`logsumexp`), `Softmax` (`softmax`), `LogSoftmax` (`log_softmax`) |
| zeta | `Zeta` (`zeta`), `HurwitzZeta` (`zeta`) |
| Bessel | `J0` (`j0`), `J1` (`j1`), `Jn` (`jv`), `Jv` (`jv`), `Y0` (`y0`), `Y1` (`y1`), `Yn` (`yn`), `Yv` (`yv`), `I0` (`i0`), `I1` (`i1`), `In` (—), `Iv` (`iv`), `K0` (`k0`), `K1` (`k1`), `Kn` (`kn`), `Kv` (`kv`), `I0e` (`i0e`), `I1e` (`i1e`), `Ive` (`ive`), `K0e` (`k0e`), `K1e` (`k1e`), `Kve` (`kve`), `JnZeros` (`jn_zeros`) |
| elliptic | `EllipK` (`ellipk`), `EllipKm1` (`ellipkm1`), `EllipE` (`ellipe`) |

The three without a pairing, and why:

- **`Kn`** — scipy does have `kn`; this package simply does not measure it
  against it. The bracketed name is scipy's, not a measurement.
- **`In`** — scipy has no `in`, and cannot: it is a Python keyword. `Iv` at
  integer order is the equivalent, and `In(n, x)` is `Iv(float64(n), x)`.
  `Kn`/`Kv` and `Jn`/`Jv` stand in the same relation, which is why `Jn` is
  bracketed `jv`: scipy's `jn` is a deprecated alias of `jv`.
- **`LogGammaSign`** — no scipy counterpart in any spelling; scipy splits the
  two results across `gammaln` and `gammasgn`.

The 20 `…To` slice kernels — `GammaTo`, `ErfTo`, `XlogyTo` and the rest — are the
batch form of the function they are named after and compute the same thing, so
they share its scipy equivalent rather than earning rows here. One exception is
worth knowing: **`TrigammaTo` is exported while `Trigamma` is not**, so that
kernel has no scalar form in the public API. It is psi'(x), which scipy spells
`polygamma(1, x)` — the same function `Polygamma` already provides.

`Zeta` and `HurwitzZeta` share one scipy name, `zeta(x, q=None)`, which takes
the second argument optionally. `LogFactorial` is bracketed as a call rather
than a name because the function is `gammaln` with its argument shifted, which
is what the harness measures it against.

## Complex functions

`github.com/go-numerics/special/cmplx` (new in v0.4.0) is a drop-in
superset of the standard library's `math/cmplx`. Changing the import path
is enough: the names and signatures are the same, with C11 Annex G
special values.

- **New functions:** `Mul` and `Div` (the Annex G product and quotient),
  `Proj`, `Log2`, `Log1p` and `Expm1`.
- **Unchanged from `math/cmplx`:** `Abs`, `Phase`, `Conj`, `Polar`,
  `Rect`, `Inf`, `NaN`, `IsInf` and `IsNaN` are one call to a correctly
  specified real function, and are re-exported as they are.
- **Rewritten:** every other function. The algorithms are FreeBSD msun's
  (as NumPy's npymath carries them), carried in double-double where
  float64 loses accuracy.
- **Underlying real functions:** `sin`, `cos` and `tan` use this
  package's Payne–Hanek reduction, and `atan2` is table-driven
  double-double.

```go
import "github.com/go-numerics/special/cmplx"

cmplx.Tanh(400 + 1i)   // (1+0i); math/cmplx: NaN
cmplx.Log(1 + 1e-10i)  // (5e-21+1e-10i); math/cmplx: real part 0
cmplx.Pow(1i, 2)       // (-1+0i), exactly; math/cmplx: (-1+1.2e-16i)
```

Each part is measured in ulps against mpmath at 192 and 320 bits.
The fixtures cover the tiny, subnormal, huge and near-overflow
regions, the branch points, and both sides of every branch cut.
Worst part per function:

| function | this package | math/cmplx (Go 1.27.3) |
|---|--:|--:|
| Mul, Div | 0.5 ulp | 1.2e16 ulp (builtin `*`, `/`) |
| Exp, Cosh, Sinh, Cos, Sin | 1.29 ulp | 439 ulp, Inf near Re z = 710 |
| Tanh, Tan, Cot | 2.55 ulp | NaN for \|Re z\| > 355 |
| Log, Log10, Log2, Sqrt | 1.03 ulp (0.5 outside subnormals) | 8.9e15 ulp near \|z\| = 1, wrong sign of π on the cut |
| Log1p, Expm1 | 0.53 ulp | 2.5e27 ulp (naive formula) |
| Atanh, Atan | 1.44 ulp | 1.4e16 ulp |
| Asinh, Asin, Acos, Acosh | 2.82 ulp ([#26](https://github.com/go-numerics/special/issues/26)) | 1e9 ulp, wrong quadrant above 1e154 |
| Pow | 0.51 ulp; up to 2937 ulp where \|y log x\| reaches 3e5 ([#27](https://github.com/go-numerics/special/issues/27)) | 3.4e20 ulp |

Each region has a CI budget of 1.5 times its measured maximum, and at
least the design's target. Two budgets sit above the target, and both
are open issues:

- **Asinh, Asin, Acos and Acosh:** 3 ulp (#26).
- **Pow, region `unit`:** 4500 ulp (#27), inside the design's
  (2 + |y log x|) ulp bound.

Every region is listed in
[docs/accuracy-complex.md](docs/accuracy-complex.md), and the design is
[docs/design-complex.md](docs/design-complex.md). Timing against NumPy's
complex ufuncs has not been measured yet.

## Double-double arithmetic

`github.com/go-numerics/special/dd` (public since v0.3.0) is the
double-double type the functions above are built on: `dd.DD{Hi, Lo}` with
`Add`, `Sub`, `Mul`, `Div`, `Sqrt`, `Exp`, `Log`, the exact accumulators
`Dot`, `DotDD`, `Sum`, `AddProd` (every product exact, accumulation in
double-double, so an ill-conditioned dot product keeps about 100 bits), and
conversions `FromInt64`, `FromBig`, `Parse` (a decimal string to the nearest
double-double), `(DD).Big`, `(DD).Float`. Relative errors, fuzzed against
`math/big`: Add 3u², Mul 5u², Div 15u²+56u³ (u = 2⁻⁵³); `Sqrt` is measured
(under 8u²). It is the arithmetic behind double-double QR with iterative
refinement (the NIST StRD Filip and Wampler problems to 15 digits).
`ExpMant` and `ExpReductionMargin` expose the exponential's argument
reduction, for callers that must know whether two nearby arguments take the
same table entry (go-statistics/stats uses it; it is the single copy of this
package).

## Portability

Pure Go; CI runs the full accuracy suite on amd64, arm64, 386, arm (32-bit),
riscv64, loong64, ppc64le and s390x, compiles the tests for every
`go tool dist list` target, and holds a 100% statement-coverage gate.

One finding along the way: on amd64, Go's `math.Log` (assembly) returns
about -709.09 for every subnormal argument (it should go down to -744.44);
this package never calls it on a value that can be subnormal.

## Roadmap

[docs/roadmap.md](docs/roadmap.md) lists what Julia's SpecialFunctions and
LogExpFunctions, R's nmath and GSL's specfunc do better or have in addition,
ranked by gain for cost, with an implementation source and an mpmath oracle
for each item.

## License

BSD 3-Clause. Ported algorithms keep their attribution in place: Cephes
(Stephen L. Moshier, distributed under BSD by SciPy), Boost.Math (Boost
Software License 1.0), FreeBSD msun (BSD-2-Clause) as carried by NumPy's
npymath (BSD-3-Clause), and Go's `math` trigReduce (BSD-3-Clause).
