# Roadmap: what other reference libraries do better

This document lists what the special-function libraries of Julia, R and GSL
do better than this package, or have and this package lacks. The SciPy gaps
are already listed in [coverage.md](coverage.md) and are not repeated here.
The aim is to bring each improvement into go-numerics, so that every binding
gets it: Ruby (rbgo), Julia, and R and C. R's binding needs the Rmath building
blocks, and Julia's needs SpecialFunctions.jl and LogExpFunctions.jl parity.

The inventory was taken on 2026-11-11 from the sources, and each reference
was opened at the path cited:

- **Julia (MIT):** SpecialFunctions, LogExpFunctions, HypergeometricFunctions,
  Elliptic and StatsFuns, from a Julia depot.
- **R 4.6.2 `src/nmath` (GPL-2+).**
- **GSL 2.8 `specfunc` (GPL-3+).**

Absence was checked with `git grep` on the exported names. Nothing was
implemented.

## Rules

**Licence.** R and GSL were read only to understand them. Implementations
must come from papers, from Cephes (BSD through SciPy), from Boost.Math
(BSL-1.0), from AMOS or SLATEC on netlib (public domain), or from MIT Julia
packages. They must never come from R's nmath or from GSL. StatsFuns'
`nchisq`, `nt`, `nbeta` and `srdist` wrap Rmath, so they are GPL and count as
papers only.

**Oracle.** As everywhere in this package, a function is measured in ulps
against mpmath on regions that cover its whole domain. The reference library
named in each row is measured on the same points.

## Ranked by gain / cost

Sizes: S < 300 lines, M 300–1500, L > 1500.

### Building blocks to export first

Most of these already exist privately here or in go-statistics. Exporting
them removes duplicates in the stack and gives R and Julia binders their
primitives.

| # | Item | Reference (licence), evidence | State here | Gain for the fleet | Source | Oracle | Size |
|---|---|---|---|---|---|---|---|
| 1 | Log-space set: `Log1mExp`, `Log2mExp`, `LogExpm1`, `LogAddExp`, `LogSubExp`, `Log1pmx`, `Logmxp1`, `XlogX`, `LogCosh`, `LogAbsSinh`, `Cloglog` | LogExpFunctions.jl (MIT) `src/basicfuns.jl:12,138,224,350,365,377,421,469,529,551,615`; R `nmath/pgamma.c:121,225,239`, `plogis.c:36` (read only) | Private: `log1pmxDD` (`beta.go:20`, and again in `incgamma.go`). go-statistics holds private copies (`log1mexpD`, `log1mexpDD`, `log1pmx`) | R `R_Log1_Exp` and `logspace_add`/`logspace_sub`; StatsFuns and LogExpFunctions parity; one copy for the stack | LogExpFunctions.jl (MIT); Mächler (2012), "Accurately computing log(1 − exp(−\|a\|))", Rmpfr vignette (paper) | mpmath `log1p(-exp(x))`, `log(exp(a)+exp(b))`, `log1p(x)-x`; compare LogExpFunctions | S |
| 2 | `Stirlerr` and `Bd0`, plus the `DpoisRaw`/`DbinomRaw` saddle-point kernels | R `nmath/stirlerr.c:53`, `bd0.c:48` (ebd0 at :241), `dpois.c:43`, `dbinom.c:73` (read only); StatsFuns (MIT) `src/misc.jl:71` | Private in go-statistics (`loader.go`: `stirlingErrDD`, `devianceTerm`, `poissonTerm`, `binomTerm`) | Core of every discrete log-density in R (`dpois`, `dbinom`, `dnbinom`, `dt`, `dgamma`); Julia `poislogpdf` and `binomlogpdf` | Loader (2000), "Fast and accurate computation of binomial probabilities" (paper); Mächler (2021) ebd0 note (paper) | mpmath `loggamma(x+1)-(x+.5)log x+x-log√(2π)` and `x log(x/M)+M-x` at 100+ bits; compare `scipy.stats.poisson.logpmf` and `binom.logpmf` | S |
| 3 | `LogGamma1p` (log Γ(1+a) accurate near 0) | R `nmath/pgamma.c:146` (read only); SpecialFunctions.jl `src/gamma_inc.jl:106` (`loggamma1p`) | Private `gamma1pm1` (`gamma.go:56`) | R `lgamma1p`, used in incomplete gamma | TOMS 708 `gamln1` (DiDonato & Morris, ACM TOMS 18, 1992); Boost.Math `lgamma` small-z | mpmath `loggamma(1+a)`; compare `scipy.special.gammaln(1+a)` | S |
| 4 | `SinPi`, `CosPi`, `TanPi` | R `nmath/cospi.c:26,49,93` (read only) | Private `sinPi`/`cosPi` (`poly.go:56,59`) | R `sinpi`/`cospi`/`tanpi`; Julia Base `sinpi` | this package's own kernels | mpmath `sinpi`/`cospi`; compare numpy `sin(pi*x)` | S |

### Better algorithms and APIs

| # | Item | Evidence | Gain | Source / oracle | Size |
|---|---|---|---|---|---|
| A1 | `log_p` and `lower_tail` inputs for the inverses (`GammaPInv`, `BetaIncInv` from log p or from the upper tail) | R `nmath/qbeta.c:57` (read only); SpecialFunctions.jl `src/gamma_inc.jl:738` (`gamma_inc_inv_psmall` takes log r); StatsFuns `*invlogcdf` | R `q*(…, log.p = TRUE)` and Julia `*invlogcdf` without losing extreme tails to `exp` underflow | Mächler (2019) qbeta notes (paper); DiDonato & Morris, ACM TOMS 12 (1986). Oracle: mpmath `findroot` on `log(betainc)`; compare `scipy.special.betaincinv(exp(lp))` | M |

### New functions

| # | Item | Reference (licence), evidence | Gain for the fleet | Source | Oracle | Size |
|---|---|---|---|---|---|---|
| 5 | Owen's T | StatsFuns (MIT) `src/owens_t.jl:29-235` (Patefield–Tandy T1–T6) | skew-normal, bivariate normal, noncentral t; coverage.md row for `owens_t` | Patefield & Tandy, J. Stat. Softw. 5 (2000); Boost `owens_t` (BSL) | mpmath `quad`; compare `scipy.special.owens_t` | S |
| 6 | Lambert W, branches 0 and −1 | GSL `specfunc/lambert.c:110,167` (read only) | coverage.md row for `lambertw`; Ruby and Julia (LambertW.jl); closed-form quantiles | Fukushima, J. Comput. Appl. Math. 244 (2013); Boost `lambert_w` (BSL) | mpmath `lambertw(x, 0/-1)`; compare `scipy.special.lambertw` | S |
| 7 | `Dawson`, `Erfi`, `LogErfc`, `LogErfcx`, `InvDigamma`, `Eta` (Dirichlet), `LogAbsBinomial` | SpecialFunctions.jl (MIT) `src/erf.jl:677,704`, `src/gamma.jl:377,491,942`; GSL `dawson.c:226` (read only) | SpecialFunctions.jl parity; the log-erfcx pieces already exist privately (`erf.go`) | Cephes `dawsn` (BSD); SpecialFunctions.jl (MIT); Rybicki (1989) for Dawson | mpmath `erfi`, `log(erfc)`, `altzeta`; compare `scipy.special.dawsn/erfi` | S each |
| 8 | Exponential and trigonometric integrals: E1, Ei, En, Si, Ci, Shi, Chi | SpecialFunctions.jl (MIT) `src/expint.jl:78,398,564`, `src/sincosint.jl:1,101`; GSL `expint`, `Shi`, `sinint` (read only) | coverage.md rows for `exp1`/`expi`/`expn` and `sici`/`shichi`; Julia `expint`/`expinti`/`sinint`/`cosint`; Ruby | SpecialFunctions.jl (MIT); Cephes `expn`/`sici`/`shichi` (BSD) | mpmath `e1`, `ei`, `expint`, `si`, `ci`, `shi`, `chi`; compare `scipy.special` | M |
| 9 | Carlson symmetric integrals RF, RD, RJ, RC and RG; incomplete F, E and Π; Jacobi sn, cn and dn | Elliptic.jl (MIT) `src/Elliptic.jl:14,69,92,115` (SLATEC DRF/DRD/DRJ); GSL `ellint.c:74,117,183,242`, `elljac.c:40` (read only) | coverage.md rows for `elliprf`…, `ellipkinc`, `ellipeinc`, `ellipj`; Elliptic.jl parity | Carlson, Numer. Algorithms 10 (1995); SLATEC netlib (public domain); Boost `ellint_rf` (BSL) | mpmath `elliprf/elliprd/elliprj/elliprg`, `ellipf`, `ellipe(φ,m)`, `ellipfun`; compare `scipy.special`. Elliptic.jl's `ellipj(u::Real, m::Real)` uses an undefined `phi` at `Elliptic.jl:123`, so it is not an oracle there | M |
| 10 | Noncentral kernels: noncentral beta and F (AS 226), noncentral t (AS 243), and noncentral χ² promoted from go-statistics (`ncx2.go`) | SpecialFunctions.jl (MIT) `src/betanc.jl:17,70,156,183`; R `nmath/pnt.c:53`, `pnbeta.c:112`, `pnchisq.c:46` (read only) | R `pnt`, `pnbeta`, `pnchisq`, `pnf`; Julia `ncbeta`/`ncF`; coverage.md rows for `chndtr`, `nctdtr`, `ncfdtr` | Lenth, AS 226 (1987) and AS 243 (1989), Appl. Stat. (papers); Ding, AS 275 (1992); Boost `non_central_*` (BSL) | mpmath `nsum` of Poisson-weighted `betainc`; compare `scipy.stats.ncf/nct/ncx2` | M |
| 11 | Studentized range (`ptukey`/`qtukey`), exact Wilcoxon and signed-rank distributions | R `nmath/ptukey.c:209` (read only); StatsFuns (MIT) `src/distrs/wilcox.jl:43`, `signrank.jl:25` (pure Julia) | R `ptukey`, `qtukey`, `pwilcox`, `psignrank`; may belong in go-statistics | Copenhaver & Holland, J. Stat. Comput. Simul. 30 (1988); StatsFuns (MIT) recursions | mpmath double `quad`; compare `scipy.stats.studentized_range` | M |
| 12 | Hypergeometric 1F1, 2F1, U, pFq | HypergeometricFunctions.jl (MIT) `src/gauss.jl:80`, `confluent.jl:26`, `drummond.jl:10-331`, `generalized.jl:61`; GSL `hyperg_2F1.c:643`, `hyperg_1F1.c:1844`, `hyperg_U.c:1767` (read only) | coverage.md rows for `hyp1f1`, `hyp2f1`, `hyperu`; Julia parity; closed forms for noncentral distributions | HypergeometricFunctions.jl (MIT; Drummond and Weniger transformations); Pearson, Olver & Porter, Numer. Algorithms 74 (2017) | mpmath `hyp1f1`, `hyp2f1`, `hyperu`, `hyper`; compare `scipy.special` | L |
| 13 | Airy functions, spherical Bessel functions, Hankel functions, and Bessel functions of complex argument | SpecialFunctions.jl (MIT) `src/bessel.jl:33,234-303,766` (AMOS via openspecfun) | coverage.md rows for `airy`, `spherical_jn`…, `hankel1`, complex `jv`; Julia `besselj`/`airyai` parity | AMOS, netlib (Amos, ACM TOMS 12 (1986), Alg. 644; public domain) | mpmath `airyai`, `airybi`, `besselj` (complex); compare `scipy.special` | L |
| 14 | Value plus error-estimate API, as in GSL's `gsl_sf_result` | GSL specfunc throughout (read only) | Lets a C binding ask for a bound; optional | design only | n/a | M |

## Where this package is already ahead

- Incomplete gamma and beta, against SpecialFunctions.jl. This package uses
  double-double Temme and TOMS 708 and reaches 0.6–9 ulp (README). Julia's
  `IND` flag trades accuracy for speed (`gamma_inc.jl:849`), and nothing here
  needs it.
- Complete elliptic integrals K and E are already present.

## Not read

These were not read in this inventory. They are unknowns, not absences.

- `docs/accuracy.md`, beyond the parts a grep found.
- R: the body of `qbeta.c`, `bessel_*.c`, `gamma_cody.c`, `choose.c` and
  `polygamma.c`.
- SpecialFunctions.jl: `logabsgamma/`, the complex zeta and polygamma paths,
  and the internals of `beta_inc.jl`.
- Gamma.jl.
- GSL: debye, dilog, clausen, fermi_dirac, laguerre, legendre, coulomb,
  coupling (3j/6j/9j), synchrotron, transport, mathieu, gegenbauer, hermite,
  `psi_1piy`, and the Bessel zeros.
- go-statistics, beyond the signatures in `loader.go`, `elementary.go` and
  `ncx2.go`.
