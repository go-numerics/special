# Design: the double-precision complex layer

Status: design, no code yet. This page decides how `go-numerics/special`
handles `complex128`: the elementary functions every complex special function
is built on, the naming of the complex special functions, the error model, how
they are validated and measured, and the order of the pull requests. It is the
first of the "shared foundations" of the go-scikit roadmap
([COVERAGE.md](https://github.com/go-scikit/roadmap/blob/main/COVERAGE.md),
section 1). Its arbitrary-precision counterpart is
[go-bignum/bigfloat docs/design-complex.md](https://github.com/go-bignum/bigfloat/blob/main/docs/design-complex.md).

## Decisions

1. **New package `github.com/go-numerics/special/cmplx`**: a drop-in superset
   of the standard library's `math/cmplx` (same names, same signatures), with
   C11 Annex G special values, accuracy measured per component, and a few
   additions (`Mul`, `Div`, `Proj`, `Log1p`, `Expm1`, `Log2`).
2. **`math/cmplx` is replaced, not wrapped**, for every function except the
   ones that are one call to a correctly specified real function (`Abs` =
   `math.Hypot`, `Phase` = `math.Atan2`, `Conj`, `Inf`, `NaN`, `IsInf`,
   `IsNaN`, `Polar`, `Rect`), which are re-exported unchanged. The evidence is
   below.
3. **The algorithms are FreeBSD msun's** (BSD-2), in the form NumPy carries
   them in `npymath` — the code NumPy itself runs wherever it does not trust
   the platform's C library — corrected with error-free transformations from
   `special/dd` in the few bands where msun loses accuracy.
4. **Complex special functions stay in the root package with a `Z` prefix**:
   `ZLogGamma`, `ZGamma`, `ZRGamma`, later `ZErf`, `ZWofz`, `ZJv`,
   `ZHankel1`, … They share the real kernels (unexported) of the root
   package.
5. **The double-double complex type is internal until a consumer needs it**
   (`special/internal/ddc`, built only on `dd`'s exported primitives); it is
   promoted to `dd.Complex` when a package outside this module asks for it.
6. **Error model**: componentwise error in ulps against an mpmath reference
   carried in double-double, for the elementary functions; normwise relative
   error for the special functions whose reference algorithm only bounds the
   norm (AMOS). Bit-for-bit equality with NumPy or SciPy is **not** a target
   for the elementary layer, because both delegate to the host C library
   (below); it is measured and published, never gated.

### Why `cmplx` and not `complex`

`complex` is a predeclared identifier of Go: a package named `complex` shadows
the builtin `complex(re, im)` in every file that imports it, which breaks the
most common expression in complex code. `cmplx` collides only with the
standard library's package of the same name, and that collision is the point:
the package is a replacement, so a file switches by changing one import path
and never needs both. (`golang.org/x/exp/slices` did the same against
`slices`.) Rejected: `zmath` (no precedent, hides the replacement), `cmath`
(reads as C's `<cmath>`, which is real-valued).

### Why a `Z` prefix for the special functions

- A `C` suffix collides with names that already end in C: `ErfC` next to
  `Erfc`, `BetaIncC` next to the complement `BetaIncC`.
- Generics cannot overload: `func LogGamma[T float64 | complex128](T) T`
  needs a type switch inside, the real and complex algorithms differ
  entirely, and the switch defeats inlining on the real path that the
  current speed table depends on.
- A sub-package (`special/zspecial`) would need the real kernels (Γ, sinπx in
  double-double, erfcx, Bessel recurrences) exported or moved to `internal/`
  first; the root package already has them.
- `Z` is the AMOS and LAPACK convention for double complex (`ZBESJ`,
  `ZGETRF`) that SciPy's sources and users already read.

## Who needs it

From each repository's `docs/coverage.md` on main (2026-11-02):

| repository | rows | what they need from this layer |
|---|---|---|
| go-numerics/special | C1 (cross-cutting), 1 `airy`/`airye`, 10-15 `jv`, `yv`, `iv`, `kv`, `jve`, `yve`, `hankel1/2(e)`, 26 `spherical_*` (complex z), 61 `psi`, 53/63/66 `gamma`, `rgamma`, `loggamma`, 67-72 `erf`, `erfc`, `erfcx`, `erfi`, `wofz`, `dawsn`, 73 `fresnel`, 75 `*_zeros` (complex roots), 76 `voigt_profile` (via `wofz`), 79 `sph_harm_y`, 89 `hyp2f1`, 90 `hyp1f1`, 97 `kelvin`, 104 `lambertw`, `wrightomega`, 108 `exp1`, `expi`, 109 `sici`, `shichi`, 110 `spence`, 115 `xlogy`, `xlog1py`, 116 `log1p`, `expm1`, M1 `PolyLog`, M15-M17 | every elementary function, `Log1p`, `Expm1`, `Mul`, `Div`; complex `sinpi`/`cospi`; `ZLogGamma` first |
| go-fft/fft | 27-29 `fht`, `ifht`, `fhtoffset` | `ZLogGamma` (SciPy v1.18.1 `scipy/fft/_fftlog_backend.py` lines 84-86 and 173-174 call `loggamma` on `xp + 1j*y`), plus the real `poch` (special row 65) |
| go-numerics/integrate | 25 `solve_ivp` complex states, 34 `ode` ZVODE and `complex_ode`, M8 complex waypoints | builtin `complex128` arithmetic, `Abs`, and complex LU from go-ndarray `linalg`; nothing beyond `cmplx.Abs`/`Mul`/`Div` from this layer |
| go-bignum/bigfloat | "Complex numbers" section and almost every other row | not this package: the arbitrary-precision type is designed in bigfloat |
| go-ndarray | complex ufuncs (`typed.go` lines 270-272 call `cmplx.Sqrt`, `Exp`, `Log`, `Sin`, `Cos`, `Tan`, `Pow`) | a later switch from `math/cmplx` to this package (last task; see "Module boundary") |

## Bibliography

Opened for this design:

- ISO/IEC 9899:201x committee draft N1570 (C11), Annex G "IEC 60559-compatible
  complex arithmetic": G.3 (conventions: a value with one infinite part is an
  infinity), G.5.1 (multiplicative operators, the `_Cmultd` and `_Cdivd`
  examples), G.6 (continuity onto both sides of each branch cut "taking into
  account the sign of zero", the derived functions `casin(z) = -i casinh(iz)`
  etc., the conj/odd/even symmetries) and the special-value lists of G.6.1.1,
  G.6.2.1-6, G.6.3.1-2, G.6.4.1-2. HTML rendering at
  <https://port70.net/~nsz/c/c11/n1570.html#G>.
- Go 1.27.3 sources: `src/math/cmplx/{abs,phase,sqrt,log,exp,pow,asin,tan}.go`
  (Cephes c9x-complex, simplified) and `src/runtime/complex.go`
  (`complex128div`: Smith 1962 with Annex G recovery of infinities).
- NumPy main (2026-11-02): `numpy/_core/src/npymath/npy_math_complex.c.src`
  (FreeBSD msun HEAD of 2013-10-04, BSD-2: `cexp` with scaled exponential,
  `clog` with the four problem cases, `csqrt` (CACM Algorithm 312 with
  scaling), `cpow` with an exact integer-power path, `ccosh`/`csinh` (msun
  r226599), `ctanh` (msun r226600, Kahan's algorithm), `cacos`/`casinh`/
  `catanh` (msun r251404, after Hull, Fairgrieve and Tang)),
  `numpy/_core/src/common/npy_config.h` (when the platform library is
  replaced by those fallbacks), `numpy/_core/src/umath/loops.c.src`
  (`CDOUBLE_multiply`: the naive formula; `CDOUBLE_divide`: Smith) and
  `numpy/_core/src/umath/funcs.inc.src` (`nc_log1p`, `nc_expm1` naive).
- xsf at commit `f7b85f50`, the submodule SciPy v1.18.1 pins:
  `include/xsf/loggamma.h` (Wilson's principal-branch log-gamma after Hare
  1997: Stirling for Re z > 7 or |Im z| > 7, Taylor around 1 and 2,
  reflection below Re z = 0.1, recurrence otherwise), `zlog1.h`, `trig.h`
  (complex `sinpi`/`cospi` with overflow-safe scaling), `log.h` (complex
  `log1p` with a double-double branch), `config.h` (all complex elementary
  functions are `std::` ones), `faddeeva.h` (Johnson's Faddeeva package, MIT),
  `amos/amos.h` (SciPy's C++ translation of AMOS, BSD), `hyp2f1.h`.
- SciPy v1.18.1 `scipy/fft/_fftlog_backend.py`.
- This repository: `docs/coverage.md`, `docs/accuracy.md`, `docs/perf.md`,
  `go doc ./dd`; the three other `coverage.md` files listed above.

Not obtained (cited only through the sources above, not read):

- W. Kahan, "Branch cuts for complex elementary functions, or much ado about
  nothing's sign bit", in *The State of the Art in Numerical Analysis*,
  Iserles and Powell eds., 1987 (cited by msun `ctanh`).
- T. E. Hull, T. F. Fairgrieve, P. T. P. Tang, "Implementing the complex
  arcsine and arccosine functions using exception handling", ACM TOMS 23(3),
  1997, doi:10.1145/275323.275324 (cited by msun `catrig`).
- D. E. Amos, Algorithm 644, ACM TOMS 12(3), 1986, and its two remarks
  (cited by xsf `amos.h`).
- M. R. Zaghloul and A. N. Ali, Algorithm 916, ACM TOMS 38(2), 2011;
  G. P. M. Poppe and C. M. J. Wijers, ACM TOMS 16(1), 1990; W. Gautschi,
  SIAM J. Numer. Anal. 7(1), 1970 (cited by `faddeeva.h`).
- D. E. G. Hare, "Computing the principal branch of log-Gamma", J.
  Algorithms 25(2), 1997 (cited by `loggamma.h`).
- The current FreeBSD msun sources (only NumPy's 2013 copy was read).
- glibc's complex functions (LGPL) and libgcc's `__muldc3`/`__divdc3` (GPL
  with runtime exception): deliberately not read. They are what SciPy runs on
  Linux; they are measured as black boxes, never ported.

## `math/cmplx` against Annex G and NumPy

`math/cmplx` says (abs.go) that "special case handling conforms to the C99
standard Annex G". Its algorithms are Cephes' c9x-complex code simplified: the
formula of each function evaluated directly, with special-value branches on
top. A probe on Go 1.27.3 (darwin/arm64), at points whose exact value is known
in closed form:

| call | Go 1.27.3 | exact | defect |
|---|---|---|---|
| `Tanh(400+1i)` | `NaN+0i` | 1 + i·(tiny) | `cosh(2x)` overflows, `sinh/cosh` = Inf/Inf |
| `Tan(1+400i)` | `0+NaNi` | (tiny) + 1i | same, through `cosh(2y)` |
| `Exp(710+1e-300i)` | `+Inf+Inf i` | +Inf + 2.2e8 i | `exp(x)` overflows before the product with `sin y` |
| `Cosh(710.4+1i)` | `+Inf+Inf i` | 9.0e307 + 1.4e308 i | same |
| `Log(1+1e-10i)` | `0+1e-10i` | 5e-21 + 1e-10 i | real part `log(hypot)`: every digit lost near the unit circle |
| `Asin(1e200+1e200i)` | `-0.785-Inf i` | +0.785 + 461.56 i | `z*z` overflows; wrong quadrant |
| `Asin(1e-300+1e-300i)` | `1e-300-0i` | 1e-300 + 1e-300 i | underflow loses the imaginary part |
| `Acos(1+1e-20i)` | `1.0000000827e-10…` | 1.0000000000e-10 - 1.0e-10 i | π/2 - asin: relative error 8e-8 (~4e8 ulp) |
| `Asinh(1e-20+0.5i)` | `0+0.5236i` | 1.1547e-20 + 0.5236 i | real part lost |
| `Pow(1i, 2)` | `-1+1.2e-16i` | -1 + 0i | no integer-exponent path (NumPy has one) |
| `(Inf+NaN i)*(1+1i)` | `NaN+NaN i` | an infinity (G.5.1p4) | builtin `*` is the naive formula |

The special-value branches are mostly right (`Sqrt(-Inf+NaN i)`,
`Log(-0-0i)`, `Atanh(1+0i)`, `Sqrt(-4-0i)` all match G.6); the defects are in
the finite range: overflow of intermediates, cancellation near the unit circle
and near the branch points, and the inverse functions computed as
`log(iz + sqrt(1-z²))`. Wrapping cannot repair these (the bad value is formed
inside), so the functions are replaced. The builtin operators are left alone:
Go's `/` already recovers infinities as in G.5.1, and Go's `*` is the same
naive formula NumPy's `CDOUBLE_multiply` uses (so ufunc parity holds with the
builtin); `cmplx.Mul` is the opt-in Annex G product.

### What NumPy and SciPy actually run

- **NumPy** calls the platform's C99 functions when the build finds them and
  falls back to `npymath` otherwise. `npy_config.h` forces the fallback for
  `casin`, `casinh`, `catan`, `catanh`, `cacos`, `cacosh` (and `csqrt`, `clog`
  under MSVC) on MSVC, on glibc older than 2.18, on non-glibc Linux (musl, plus
  `clog`), and on Cygwin. On glibc >= 2.18 (manylinux wheels) and on macOS it
  runs the platform library. `log1p`, `expm1`, `log10`, `log2`, `exp2` of
  complex arguments are NumPy's own naive formulas (`funcs.inc.src`), and
  `power` takes an exact repeated-multiplication path for integer exponents
  |n| < 100 before falling back to `cpow`.
- **SciPy** (`xsf`) calls `std::log`, `std::exp`, `std::sqrt`, `std::pow`,
  `std::abs` on `std::complex<double>` (`config.h`), that is libstdc++/libc++
  over the host's libm, and complex `*`/`/` through the compiler's runtime.
  Its own algorithms sit on top: `loggamma`, `zlog1`, complex `sinpi`/
  `cospi`, `log1p` (double-double branch), Faddeeva, AMOS.

Consequently the same NumPy version gives different complex results on
Linux, macOS and Windows, and no Go port can be bit-identical to "NumPy" in
general. What can be bit-identical is an *algorithm*: where SciPy's
algorithm is its own (xsf), it is ported exactly, so that the only
differences left are the ones propagated from the elementary layer.

## API

```go
package cmplx // import "github.com/go-numerics/special/cmplx"

// Re-exported unchanged from math/cmplx (one call to an Annex F function):
func Abs(x complex128) float64             // math.Hypot(real, imag)
func Phase(x complex128) float64           // math.Atan2(imag, real)
func Conj(x complex128) complex128
func Polar(x complex128) (r, θ float64)
func Rect(r, θ float64) complex128
func Inf() complex128
func NaN() complex128
func IsInf(x complex128) bool
func IsNaN(x complex128) bool

// Replaced (same signatures as math/cmplx):
func Sqrt(x complex128) complex128
func Exp(x complex128) complex128
func Log(x complex128) complex128
func Log10(x complex128) complex128
func Pow(x, y complex128) complex128
func Sin(x complex128) complex128
func Cos(x complex128) complex128
func Tan(x complex128) complex128
func Cot(x complex128) complex128
func Sinh(x complex128) complex128
func Cosh(x complex128) complex128
func Tanh(x complex128) complex128
func Asin(x complex128) complex128
func Acos(x complex128) complex128
func Atan(x complex128) complex128
func Asinh(x complex128) complex128
func Acosh(x complex128) complex128
func Atanh(x complex128) complex128

// Additions:
func Mul(x, y complex128) complex128       // Annex G G.5.1 product, componentwise accurate
func Div(x, y complex128) complex128       // Annex G G.5.1 quotient with logb scaling
func Proj(x complex128) complex128         // C11 7.3.9.5 cproj
func Log2(x complex128) complex128
func Log1p(x complex128) complex128        // log(1+x), accurate for small x (SciPy xsf log1p)
func Expm1(x complex128) complex128        // exp(x)-1, accurate for small x
```

Root package, first consumers (task 9):

```go
func ZLogGamma(z complex128) complex128    // principal branch, scipy.special.loggamma
func ZGamma(z complex128) complex128       // scipy.special.gamma, complex loop
func ZRGamma(z complex128) complex128      // scipy.special.rgamma, complex loop
```

Conventions for every later `Z` function: argument and result `complex128`;
`NaN` parts propagate as in Annex G; a pole returns the value SciPy returns
(e.g. `ZLogGamma` at a non-positive integer returns NaN+NaN i, as xsf does
with `SF_ERROR_SINGULAR`); no error-state machinery (as for the real
functions).

No function allocates; every function is safe for concurrent use; results are
**bit-identical on all eight CI architectures**: products that must not be
fused are written `float64(a*b)`, and the fused ones use `math.FMA`
explicitly (Go may fuse `x*y+z` on architectures with a fused multiply-add
otherwise — accuracy.md already records the cost for the real package: `Jv`
at negative order measures 7.2 ulp on most architectures and 13.2 ulp on
loong64, ppc64le and s390x).

## Algorithms

Each port keeps the attribution comment of its source at the function
(FreeBSD msun: David Schultz, Stephen Montgomery-Smith, BSD-2; xsf: SciPy
developers, BSD-3).

| function | algorithm | source | change from the source |
|---|---|---|---|
| `Mul` | (ac - bd) and (ad + bc), each as an exact sum of two exact products (`dd.TwoProd`, then `dd.TwoSum` and one rounding); G.5.1 infinity recovery when both parts are NaN | Annex G `_Cmultd` | exact products instead of naive ones: faithful per component |
| `Div` | scale the divisor by 2^-logb(max(\|c\|,\|d\|)), quotient, rescale; G.5.1 recovery of infinities and zeros | Annex G `_Cdivd` | numerators as exact sums of exact products |
| `Proj` | identity, or +Inf ± 0i for any infinite part | C11 7.3.9.5 | — |
| `Exp` | e^x cis y; for x in [710.4759, 1454.9159] (npymath's `SCALED_CEXP_LOWER`/`UPPER`) e^(x - 1799 ln2) and cis y split by `frexp`, multiplied, rescaled by `ldexp`; special values of G.6.3.1 | msun `cexp`, `__ldexp_cexp` (npymath `_npy_scaled_cexp`) | e^x and its scaling from the root package's table-driven kernel when a measured gap requires it |
| `Cosh`, `Sinh`, `Cos`, `Sin` | cosh x cos y + i sinh x sin y with the `CCOSH_BIG` split (e^\|x\|/2 alone above it) and the scaled exponential near overflow; `Cos(z) = Cosh(iz)`, `Sin(z) = -i Sinh(iz)` (G.6p7) | msun `ccosh`, `csinh` | — |
| `Tanh`, `Tan`, `Cot` | Kahan: t = tan y, β = 1 + t², s = sinh x, ρ = √(1+s²); (βρs + it)/(1 + βs²); \|x\| >= `TANH_HUGE`: ±1 + i·4 sin y cos y e^(-\|x\|)e^(-\|x\|) | msun `ctanh`; `Tan(z) = -i Tanh(iz)` | `Cot = 1/Tan` through `Div` |
| `Log`, `Log10`, `Log2` | Re log\|z\| with the four cases of npymath `clog` (rescaling above DBL_MAX/4 and for subnormal \|z\|; the band 0.71 <= \|z\| <= 1.73 as log1p of t/2); Im `math.Atan2` | npymath `clog` | in the band, t = x² + y² - 1 formed **exactly** as an unevaluated sum (`dd.TwoProd` on both squares, `dd.TwoSum`), then log1p(t_hi) + t_lo/(1+t_hi): npymath's `(am-1)*(am+1)+an*an` rounds three times and loses every digit when t is tiny; Log10/Log2 multiply by log10(e)/log2(e) held as `DD` |
| `Sqrt` | CACM Algorithm 312 with scaling above DBL_MAX/(1+√2); the special-value table of G.6.4.2 | npymath `csqrt` | — |
| `Pow` | b = 0 → 1; a = 0 per NumPy; integer b with \|b\| < 100 by binary powering with `Mul`, reciprocal via `Div`; otherwise `Exp(b·Log(a))` | npymath `npy_cpow` (body) | `Mul` instead of the naive product inside the powering; the range check `-100 < b < 100` happens **before** the float-to-int conversion (Go's conversion of an out-of-range float is implementation-defined) |
| `Asinh`, `Asin`, `Acos`, `Acosh`, `Atanh`, `Atan` | Hull–Fairgrieve–Tang: A = (\|z+i\|+\|z-i\|)/2, B = y/A; log1p((A-1)+√((A-1)(A+1))) below A_crossover with A-1 = f(x,1+y)+f(x,1-y), f(a,b) = a²/(hypot(a,b)+b)/2; atan2(y, √((A+y)(A-y))) above B_crossover; `clog_for_large_values` beyond 1/ε; `Atanh` via `sum_squares` and `real_part_reciprocal`; `Asin(z) = -i Asinh(iz)`, `Atan(z) = -i Atanh(iz)` | msun `catrig.c` r251404 (npymath) | — |
| `Log1p` | xsf `log1p(complex)`: \|z\| < 0.707: ½ log1p(\|z\|(\|z\| + 2x/\|z\|)), with the double-double branch where x < 0 and y² ≈ -2x; otherwise Log(1+z) | xsf `log.h` | double-double from `dd` (xsf uses its `double_double`) |
| `Expm1` | Re = expm1(x) cos y - 2 sin²(y/2), Im = e^x sin y (NumPy's `nc_expm1`) | NumPy `funcs.inc.src` | Re formed with exact products and one rounding where the two terms cancel (NumPy's is naive) |
| `ZLogGamma` | xsf `loggamma`: Stirling (8 Bernoulli terms) for Re z > 7 or \|Im z\| > 7; Taylor (23 terms) for \|z-1\| < 0.2; `zlog1(z-1)` + Taylor for \|z-2\| < 0.2; reflection with 2πi·floor(x/2+1/4)·sign(y) and complex sinπz for Re z < 0.1; backward recurrence with sign-flip counting otherwise | xsf `loggamma.h`, `zlog1.h`, `trig.h` (`sinpi`) | complex `sinpi` on the root package's `sinPiDD`/`cosPiDD`; port otherwise **exact**, so that equality with SciPy is limited by the elementary layer |
| `ZGamma`, `ZRGamma` | exp(±ZLogGamma); zeros of 1/Γ exact at non-positive integers; non-finite input → NaN (xsf guards libstdc++'s `std::polar` assertion) | xsf `loggamma.h` | — |

### The double-double complex type

`special/internal/ddc`: `type Complex struct{ Re, Im dd.DD }` with `Add`,
`Sub`, `Mul`, `MulC` (by a `complex128`), `Sqr`, `Div`, `Norm` (|z|² as `DD`),
`Conj`, `Ldexp`, `Hi() complex128`. Every operation is a composition of `dd`'s
exported double-word algorithms (Joldes, Muller, Popescu, ACM TOMS 44(2),
2017, already cited in accuracy.md); no new error-free transformation is
written. Bounds, tested by fuzzing against `math/big` as `dd` is: `Add`,
`Sub` componentwise 3u² (u = 2⁻⁵³); `Mul` normwise (2·5u² + 3u²) relative to
\|x\|·\|y\| (componentwise only when ac and bd do not cancel); `Div` normwise,
bound measured and stated in the package comment.

It is created with its first consumer, not before (the slice that measures a
gap a double kernel cannot close: candidates are the hyp2f1 series near
|z| = 1, the recurrence product of `ZLogGamma` between 0.1 and 7 if it shows
more than 2 ulp, and the Faddeeva continued fraction). It is promoted to an
exported `dd.Complex` only when a package outside this module needs it.

## Accuracy targets

Componentwise: the error of each part, in ulps of that part's exact value,
against the mpmath reference stored as double-double (the existing
`|y - (hi+lo)| / ulp(hi)` measure applied to each part). A target is the goal
the implementation is written for; the CI gate is the existing budget rule
(1.5 x the measured maximum, at least 1 ulp, per region).

| function | target, finite region | notes |
|---|---|---|
| `Abs`, `Phase`, `Conj`, `Proj`, `Rect` | as `math.Hypot` / `math.Atan2` (<= 1 ulp) | unchanged code |
| `Mul`, `Div` | <= 1 ulp per part (`Mul`); <= 2 ulp per part (`Div`) | NumPy's naive multiply has no componentwise bound: report it |
| `Exp`, `Sin`, `Cos`, `Sinh`, `Cosh` | <= 2 ulp per part, including Re z up to 1454.9 where a part is finite | Go: both parts overflow once e^x does (709.78) |
| `Tan`, `Tanh`, `Cot` | <= 3 ulp per part | Go: NaN for \|Im\| > 355 |
| `Log` | Re <= 1 ulp everywhere including \|z\| ≈ 1; Im <= 1 ulp | Go: Re loses all digits near \|z\| = 1 |
| `Log10`, `Log2` | <= 1.5 ulp per part | |
| `Sqrt` | <= 1.5 ulp per part | |
| inverse functions | <= 2 ulp per part, near 0, the branch points ±1, ±i, both sides of each cut, \|z\| up to DBL_MAX, subnormals | Go: wrong quadrant above 1e154, ~1e8 ulp near the branch points |
| `Log1p`, `Expm1` | <= 2 ulp per part | NumPy's are naive (all digits lost for \|z\| < 1e-8) |
| `Pow` | exact for integer exponents where the exact result is representable and no intermediate rounds (i², (1+i)⁴); otherwise normwise <= (2 + \|b·Log a\|) ulp | the `Exp(b Log a)` path inherits the absolute error of b·Log a; a double-double path is a later task, gated by measurement |
| `ZLogGamma` | normwise <= 2 ulp; Im exact multiple of 2π branch tracking (no 2π jumps off the principal branch) | xsf is the algorithm; SciPy measured on the same points |
| `ZGamma`, `ZRGamma` | normwise <= 4 ulp for \|Re z\| <= 171; relative error grows with \|ZLogGamma\| beyond (stated, as for the real `Gamma` above 10 before its kernels) | |

Special values: every entry of Annex G G.6 (G.6.1.1 through G.6.4.2, and the
functions G.6p7 derives) is an exact expectation, including the sign of each
zero; entries whose sign the standard leaves unspecified accept either sign.
A mismatch is `wrong`, not an ulp count.

## Validation

**Reference.** `testdata/gen/fam_complex.py` (one more family in the existing
generator): mpmath 1.4.1 evaluated twice, at 192 and at 320 bits, a point
whose two evaluations differ by more than 2⁻⁸⁰ relative in either part dropped
and counted in the header; each part stored as `hi lo`. The same rows carry
NumPy 2.5.4's and SciPy 1.18.1's results on the same float64 inputs, generated
on the CI's linux/amd64 runner (glibc; recorded in the header with
`np.show_config()`'s libm line), and a second NumPy column generated on
windows/amd64 (where `npy_config.h` forces the msun inverse functions): the
Windows column is the one whose inverse functions run the same algorithm as
this package.

**Regions** per function (about 1000 points each, fixed seed, log-uniform in
\|z\| and uniform in arg where the scale matters):

- near 0 (\|z\| in 2⁻¹⁰⁷⁴ … 2⁻²⁶), subnormal parts, one subnormal and one
  normal part;
- moderate (\|z\| in 1/8 … 8), all four quadrants;
- the band 0.71 <= \|z\| <= 1.73 and \|z\| = 1 ± 2⁻⁵² … 2⁻²⁰ (`Log`);
- around each branch point (±1, ±i, 0) at distances 2⁻⁵² … 2⁻¹⁰;
- **each branch cut on both sides**: Im = +0 and Im = -0, and Im = ±2⁻¹⁰⁷⁴,
  ±2⁻⁵⁰, for `Sqrt`, `Log*`, `Pow` on (-∞, 0]; `Asin`, `Acos`, `Atanh` on
  (-∞, -1] ∪ [1, ∞); `Asinh`, `Atan` on the imaginary axis beyond ±i;
  `Acosh` on (-∞, 1]; `ZLogGamma` on (-∞, 0] (its cut) — with the expected
  jump of the imaginary part asserted, not only the error;
- near overflow: Re in (709, 711) and (1454, 1460) for the exponential family,
  \|Im\| in (354, 357) for `Tan`/`Tanh`, \|z\| in 1e154 … DBL_MAX for the
  inverse functions;
- special values: the Cartesian grid {±0, ±2⁻¹⁰⁷⁴, ±1, ±2¹⁰²³, ±Inf, NaN}²
  against the G.6 tables (exact, no mpmath).

**Symmetries, bit-exact**: f(conj z) = conj f(z) for every function, odd/even
(`Sinh`, `Sin`, `Tanh`, `Tan`, `Asinh`, `Asin`, `Atanh`, `Atan` odd; `Cosh`,
`Cos` even), applied to every fixture point.

**Negative control**: the existing script, extended to perturb one constant
of the new code (`CCOSH_BIG`, a crossover of `catrig`, one Stirling
coefficient of `ZLogGamma`) and assert that the gate fails.

**Published**: a complex section in `docs/accuracy-table.md` with, per
region, this package's maximum error, NumPy (Linux), NumPy (Windows),
SciPy, and the bit-equality rate with each; regions where this package is
above NumPy or SciPy are open defects.

## Performance

- `nd/perf_test.go` gains the complex ufuncs: `np.exp`, `np.log`, `np.sqrt`,
  `np.sin`, `np.tanh`, `np.arcsin`, `np.arccosh`, `np.arctanh`, `np.power`
  (integer and complex exponent), `np.log1p`, and `scipy.special.loggamma`,
  `gamma`, `rgamma` on complex input; 10⁶ points generated identically on both
  sides, rounds interleaved, single-thread ratio as the headline, never run
  at a 1-minute load above 8 (deferred and said so otherwise).
- Target: at or below NumPy's single-thread time per element for every
  elementary function; a slower row is an open defect with an issue.
- Order of work where a row is slow: algorithm first (one `math.Sincos` per
  call instead of two calls; one `math.Exp` shared by `cosh` and `sinh`; the
  `dd` band of `Log` entered only when 0.71 <= \|z\| <= 1.73); then batch
  kernels for the `nd` layer from go-simd (vector exp, log, sincos), checking
  what go-simd has first; a missing kernel goes into go-simd, not here.
  Allocation count asserted 0 per call by `testing.AllocsPerRun`.

## Fuzzing and security

- No input is parsed; the attack surface is numeric. Every loop is bounded by
  a constant: `Pow`'s powering by \|b\| < 100 (checked before conversion),
  `zlog1`'s series by 17 terms, `ZLogGamma`'s recurrence by 7 steps and its
  reflection by one level of recursion; no function allocates.
- Fuzz targets (corpus committed under `testdata/fuzz/`): one per function,
  two float64 inputs (and a third for `Pow`, `Mul`, `Div`), asserting: no
  panic; bit-exact conj and odd/even symmetries; Annex G classification
  (finite in → finite or ±Inf out where the function is finite there, NaN
  only where G.6 says NaN); `Exp(Log z)` within a relative bound of z for
  finite nonzero z away from overflow; `Mul`/`Div` against `ddc` or
  `math/big` exact products.
- 32-bit platforms (386, arm): no int arithmetic on exponents beyond `logb`
  results (|e| < 2100), so no overflow; the CI matrix already runs both.
- `govulncheck` already runs in CI; this package adds no dependency.

## Module boundary

`special` requires go-ndarray (for `nd`). If go-ndarray's complex ufuncs
switch from `math/cmplx` to this package, go-ndarray would require `special`
and the two modules would require each other. Go accepts module cycles, but
each side then pins an older version of the other. The clean form is to make
`cmplx` and `dd` nested modules (`github.com/go-numerics/special/dd`,
`…/special/cmplx`, stdlib-only), released together with a `special` version
that no longer contains them (so no build sees the package in two modules).
That is the last task below and needs its own short design in go-ndarray;
nothing before it depends on it, and `cmplx` imports only the standard
library and `dd` from the start so that the carve-out is mechanical.

## File layout

```text
cmplx/
  doc.go            package comment: Annex G, accuracy contract, attribution list
  stdlib.go         Abs, Phase, Conj, Polar, Rect, Inf, NaN, IsInf, IsNaN
  arith.go          Mul, Div, Proj
  exp.go            Exp, scaled exponential, Expm1
  hyp.go            Cosh, Sinh, Tanh, Cos, Sin, Tan, Cot
  log.go            Log, Log10, Log2, Log1p
  sqrt.go           Sqrt
  pow.go            Pow
  catrig.go         Asinh, Asin, Acos, Acosh, Atanh, Atan (and their helpers)
  annexg_test.go    the G.6 special-value tables as data, the symmetry checks
  accuracy_test.go  per-region errors against testdata/fixtures/c_*.tsv.gz
  fuzz_test.go
  bench_test.go
internal/ddc/       the double-double complex type (with its first consumer)
loggamma_complex.go ZLogGamma, ZGamma, ZRGamma, zsinpi, zcospi, zlog1
testdata/gen/fam_complex.py
testdata/fixtures/c_*.tsv.gz
docs/design-complex.md   this page
```

## Tasks (one pull request each, in order)

1. **Complex accuracy harness**: `fam_complex.py` (mpmath two-precision
   reference, NumPy Linux + Windows and SciPy columns), the componentwise
   error measure, the G.6 special-value tables as test data, the symmetry
   checker; `cmplx` package with `doc.go` and `stdlib.go` only.
2. **`cmplx.Mul`, `Div`, `Proj`.**
3. **`cmplx.Exp`, `Cosh`, `Sinh`, `Cos`, `Sin`** (scaled exponential).
4. **`cmplx.Tanh`, `Tan`, `Cot`** (Kahan).
5. **`cmplx.Log`, `Log10`, `Log2`, `Sqrt`.**
6. **`cmplx.Log1p`, `Expm1`.**
7. **`cmplx.Asinh`, `Asin`, `Acos`, `Acosh`, `Atanh`, `Atan`** (catrig).
8. **`cmplx.Pow`.**
9. **Complex perf report and fuzz corpus**; README, landing and docs site
   pages for `cmplx`; release (minor).
10. **`ZLogGamma`, `ZGamma`, `ZRGamma`** (xsf port; complex `sinpi`/`cospi`);
    unblocks go-fft `fht` together with the real `Poch` (row 65).
11. **`nd` complex dtype**: the array forms of the functions above (C2).
12. **Carve `dd` and `cmplx` into nested modules**, then go-ndarray switches
    its complex ufuncs (design note in go-ndarray first).

The consumer slices that follow (not part of this foundation, each with its
own fixtures): Faddeeva (`ZWofz`, `ZErf`, `ZErfc`, `ZErfcx`, `ZErfi`,
`ZDawsn`, then `VoigtProfile`) — creates `internal/ddc` if its measurements
need it; `ZDigamma`; AMOS (`ZJv`, `ZYv`, `ZIv`, `ZKv`, scaled forms,
`ZHankel1/2`, `ZAiry`); `ZHyp2f1`, `ZHyp1f1`; `ZExp1`, `ZExpi`;
`ZLambertW`, `ZWrightOmega`; `ZSpence`; Kelvin; spherical Bessel of complex
argument; `SphHarmY`.
