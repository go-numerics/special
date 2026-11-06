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

## Speed

Accuracy has a price: single-threaded, a Go loop is about 3x slower than
SciPy's C for the gamma and error-function families and 10–15x for the
Bessel functions of real order and the incomplete beta function. The `nd`
functions spread large arrays over the cores and beat SciPy for 21 of 31
functions measured (typically 1.5–3x on a 16-core host; e.g. Gamma 6.4 vs
13.4 ns/element, Ndtr 4.7 vs 9.0, GammaP 37 vs 137).

Method and full table: [docs/perf.md](docs/perf.md).

## Functions

Each row gives the Go function and, in brackets, the `scipy.special` function it
is **measured against** — the pairing comes from the harness
(`bench/scipy_server.py` and `testdata/gen/fam_*.py`), not from transliterating
a name, which is why `Log1pExp` is scipy's `softplus` and `LogGamma` is
`gammaln`. 63 of the 67 carry such a pairing; `—` marks the four that do not,
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

The four without a pairing, and why:

- **`GammaSign`** and **`Kn`** — scipy does have `gammasgn` and `kn`; this
  package simply measures neither against them. The bracketed name is scipy's,
  not a measurement.
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

## Portability

Pure Go; CI runs the full accuracy suite on amd64, arm64, 386, arm (32-bit),
riscv64, loong64, ppc64le and s390x, compiles the tests for every
`go tool dist list` target, and holds a 100% statement-coverage gate.

One finding along the way: on amd64, Go's `math.Log` (assembly) returns
about -709.09 for every subnormal argument (it should go down to -744.44);
this package never calls it on a value that can be subnormal.

## License

BSD 3-Clause. Ported algorithms keep their attribution in place: Cephes
(Stephen L. Moshier, distributed under BSD by SciPy), Boost.Math (Boost
Software License 1.0), FreeBSD msun (BSD-2-Clause) as carried by NumPy's
npymath (BSD-3-Clause), and Go's `math` trigReduce (BSD-3-Clause).
