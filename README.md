# special

Special mathematical functions in pure Go (no cgo, no assembly): the gamma
family, error functions, incomplete gamma and beta functions and their
inverses, Bessel functions, zeta functions, combinatorics, logistic helpers
and complete elliptic integrals — with an element-wise layer over
[go-ndarray](https://github.com/go-ndarray/ndarray).

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

ACCURACY_HEADLINE_PLACEHOLDER

Full method, sources of every algorithm, negative control and limitations:
[docs/accuracy.md](docs/accuracy.md); every region:
[docs/accuracy-table.md](docs/accuracy-table.md).

## Speed

PERF_HEADLINE_PLACEHOLDER

Method and full table: [docs/perf.md](docs/perf.md).

## Functions

| family | functions |
|---|---|
| gamma | `Gamma`, `LogGamma`, `LogGammaSign`, `GammaSign`, `Digamma`, `Polygamma`, `Beta`, `LogBeta` |
| incomplete gamma / beta | `GammaP`, `GammaQ`, `GammaPInv`, `GammaQInv`, `BetaInc`, `BetaIncC`, `BetaIncInv`, `BetaIncCInv` |
| error function, normal | `Erf`, `Erfc`, `Erfcx`, `ErfInv`, `ErfcInv`, `Ndtr`, `Ndtri`, `LogNdtr` |
| combinatorics | `Comb`, `Perm`, `Binom`, `Factorial`, `Factorial2`, `LogFactorial` |
| logistic, log helpers | `Expit`, `LogExpit`, `Logit`, `Log1pExp`, `Xlogy`, `Xlog1py`, `LogSumExp`, `Softmax`, `LogSoftmax` |
| zeta | `Zeta`, `HurwitzZeta` |
| Bessel | BESSEL_LIST_PLACEHOLDER |
| elliptic | `EllipK`, `EllipKm1`, `EllipE` |

Conventions: functions return NaN outside their domain and ±Inf at poles,
as the IEEE 754 operations do; nothing panics on a float argument. Edge
values follow SciPy's where SciPy's are mathematically sound; the exceptions
are documented at each function.

## Portability

Pure Go; CI runs the full accuracy suite on amd64, arm64, 386, arm (32-bit),
riscv64, loong64, ppc64le and s390x, compiles the tests for every
`go tool dist list` target, and holds a 100% statement-coverage gate.

One finding along the way: on amd64, Go's `math.Log` (assembly) returns
about -709.09 for every subnormal argument (it should go down to -744.44);
this package never calls it on a value that can be subnormal.

## License

BSD 3-Clause. Ported algorithms keep their attribution in place: Cephes
(Stephen L. Moshier, distributed under BSD by SciPy) and Boost.Math (Boost
Software License 1.0).
