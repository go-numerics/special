# Coverage against SciPy, MATLAB and Mathematica

Inventory of what `go-numerics/special` (main) implements against
`scipy.special` v1.18.1, the MATLAB special-function set and Mathematica's
special-function list. Status: **done** (same function, real arguments,
accuracy measured), **partial** (what is missing is named), **missing**.
"Dep" is what a missing item needs beyond this package. A row groups the
SciPy names that share an algorithm family; its status applies to every name in
it unless the cell says otherwise.

Sources. SciPy: `scipy/special/__init__.py` at tag v1.18.1 (the rows follow
its sections in order; every name in its autosummary lists appears in some
row). MATLAB: the MathWorks function lists could not be read by the tool used
(the pages are rendered client-side), so the MATLAB column comes from
knowledge of the core and Symbolic Math Toolbox function names and must be
re-checked against the MathWorks reference lists before a slice is started.
Mathematica: from knowledge of the "Special Functions" guide page, likewise to
be re-checked. Go side: README, `go doc`, and the sources on main.

Conventions. The package is real-valued: `float64` in, `float64` out. SciPy
accepts complex arguments for `gamma`, `loggamma`, `psi`, `erf`, `jv`, `iv`,
`kv`, `hyp2f1`, `lambertw` and others; that is tracked once, in row C1, not
repeated per name. `scipy.special.geterr/seterr/errstate` and the
`cython_special` module are not counted: Go functions return NaN or Inf where
SciPy's default does, and there is no error-state machinery.

## Totals

Counted by `go run` over the tables below (rows, not names).

| | done | partial | missing | rows |
|---|---|---|---|---|
| SciPy v1.18.1 (sections 1-11) | 33 | 1 | 89 | 123 |
| Beyond SciPy (Mathematica / MATLAB, section 12) | 0 | 0 | 22 | 22 |

## 1. Airy, elliptic

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 1 | `airy`, `airye` | `airy` | | missing | Cephes airy; or Bessel of order 1/3 (have `Jv`, `Iv`, `Kv`) |
| 2 | `ai_zeros`, `bi_zeros` | | | missing | root finding on `airy` |
| 3 | `itairy` | | | missing | Zhang-Jin (specfun) port |
| 4 | `ellipk` | `ellipke` | `EllipK` | done | |
| 5 | `ellipkm1` | | `EllipKm1` | done | |
| 6 | `ellipe` | `ellipke` | `EllipE` | done | |
| 7 | `ellipkinc`, `ellipeinc` | `ellipticF`, `ellipticE` (Symbolic) | | missing | Carlson `RF`/`RD` (row 9) |
| 8 | `ellipj` | `ellipj` | | missing | Cephes ellipj (AGM descent) |
| 9 | `elliprc`, `elliprd`, `elliprf`, `elliprg`, `elliprj` | | | missing | Carlson duplication algorithms (DLMF 19.36, SciPy's C++ port) |

## 2. Bessel functions

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 10 | `jv` | `besselj` | `Jv`, `Jn` | done | real argument and order; complex in C1 |
| 11 | `jve`, `yve` | `besselj(...,1)`, `bessely(...,1)` | | missing | scaling of `Jv`/`Yv` (trivial for real x, matters for complex) |
| 12 | `yn`, `yv` | `bessely` | `Yn`, `Yv` | done | |
| 13 | `iv`, `ive` | `besseli` | `Iv`, `Ive`, `In` | done | |
| 14 | `kn`, `kv`, `kve` | `besselk` | `Kn`, `Kv`, `Kve` | done | |
| 15 | `hankel1`, `hankel1e`, `hankel2`, `hankel2e` | `besselh` | | missing | complex values: `Jv`+i`Yv` for real x; complex x needs AMOS port |
| 16 | `wright_bessel`, `log_wright_bessel` | | | missing | series/asymptotics from SciPy's C++ (Luchko) |
| 17 | `lmbda` | | | missing | Jahnke-Emden (specfun) |
| 18 | `jn_zeros` | | `JnZeros` | done | |
| 19 | `jnp_zeros`, `jnjnp_zeros`, `jnyn_zeros`, `yn_zeros`, `ynp_zeros` | | | missing | McMahon + Newton on the existing `Jv`/`Yv` |
| 20 | `y0_zeros`, `y1_zeros`, `y1p_zeros` | | | missing | same |
| 21 | `j0`, `j1`, `y0`, `y1` | | `J0`, `J1`, `Y0`, `Y1` | done | |
| 22 | `i0`, `i0e`, `i1`, `i1e`, `k0`, `k0e`, `k1`, `k1e` | | the same, capitalised | done | |
| 23 | `itj0y0`, `it2j0y0`, `iti0k0`, `it2i0k0` | | | missing | Zhang-Jin (specfun) |
| 24 | `besselpoly` | | | missing | Cephes besselpoly (series) |
| 25 | `jvp`, `yvp`, `ivp`, `kvp`, `h1vp`, `h2vp` | | | missing | recurrences on existing functions (`h1vp`/`h2vp` need row 15) |
| 26 | `spherical_jn`, `spherical_yn`, `spherical_in`, `spherical_kn` (+ derivative) | | | missing | closed forms / recurrences; `Jv` at half-integer order |
| 27 | `riccati_jn`, `riccati_yn` | | | missing | recurrences |
| 28 | `struve`, `modstruve` | | | missing | Struve H/L series + asymptotics (Cephes `struve`, SciPy 1.x C++) |
| 29 | `itstruve0`, `it2struve0`, `itmodstruve0` | | | missing | Zhang-Jin (specfun) |

## 3. Raw statistical functions (distribution kernels)

The Go distributions in `go-statistics/stats` compute these internally
through `BetaInc`, `GammaP` and friends, but the functions themselves are not
exported from `special`.

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 30 | `bdtr`, `bdtrc`, `bdtri` | `binocdf`, `binoinv` (Statistics) | | missing | thin wrappers on `BetaInc`, `BetaIncInv` |
| 31 | `bdtrik`, `bdtrin` | | | missing | CDFLIB inverses: 1-D root finder (optimize `Brentq`) |
| 32 | `btdtria`, `btdtrib` | | | missing | root finder on `BetaInc` |
| 33 | `fdtr`, `fdtrc`, `fdtri` | `fcdf`, `finv` | | missing | wrappers on `BetaInc` |
| 34 | `fdtridfd` | | | missing | root finder |
| 35 | `gdtr`, `gdtrc`, `gdtria`, `gdtrib`, `gdtrix` | `gamcdf` | | missing | wrappers on `GammaP`; `ria`/`rib` need a root finder |
| 36 | `nbdtr`, `nbdtrc`, `nbdtri`, `nbdtrik`, `nbdtrin` | `nbincdf` | | missing | wrappers on `BetaInc`; `ik`/`in` root finder |
| 37 | `ncfdtr`, `ncfdtri`, `ncfdtridfd`, `ncfdtridfn`, `ncfdtrinc` | `ncfcdf` | | missing | CDFLIB `cumfnc` (noncentral beta series) + root finder |
| 38 | `nctdtr`, `nctdtridf`, `nctdtrit`, `nctdtrinc` | `nctcdf` | | missing | non-central t series (Boost.Math `non_central_t`) + root finder |
| 39 | `nrdtrimn`, `nrdtrisd` | | | missing | closed forms on `Ndtri` |
| 40 | `ndtr`, `ndtri`, `log_ndtr` | `normcdf`, `norminv` | `Ndtr`, `Ndtri`, `LogNdtr` | done | |
| 41 | `ndtri_exp` | | | missing | inverse of `LogNdtr` (Newton on existing tails) |
| 42 | `pdtr`, `pdtrc`, `pdtri`, `pdtrik` | `poisscdf` | | missing | wrappers on `GammaQ`, `GammaQInv`; `pdtrik` root finder |
| 43 | `stdtr`, `stdtrit`, `stdtridf` | `tcdf`, `tinv` | | missing | wrappers on `BetaInc`; `stdtridf` root finder |
| 44 | `chdtr`, `chdtrc`, `chdtri`, `chdtriv` | `chi2cdf`, `chi2inv` | | missing | wrappers on `GammaP`; `chdtriv` root finder |
| 45 | `chndtr`, `chndtridf`, `chndtrinc`, `chndtrix` | `ncx2cdf` | | missing | CDFLIB `cumchn` series + root finder |
| 46 | `smirnov`, `smirnovi`, `kolmogorov`, `kolmogi` | | | missing | exists inside `go-statistics/stats` (`ksdist.go`); moving it down is a refactor, not new maths |
| 47 | `boxcox`, `boxcox1p`, `inv_boxcox`, `inv_boxcox1p` | `boxcox` (Statistics) | | missing | none (`expm1`, `log1p`) |
| 48 | `logit`, `expit`, `log_expit` | `logit` (Statistics) | `Logit`, `Expit`, `LogExpit` | done | |
| 49 | `tklmbda` | | | missing | Tukey lambda cdf (root finder on the quantile) |
| 50 | `owens_t` | | | missing | Patefield-Tandy / Boost `owens_t` |

## 4. Information theory

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 51 | `entr`, `rel_entr`, `kl_div` | `entropy` (Image) | | missing | none |
| 52 | `huber`, `pseudo_huber` | | | missing | none |

## 5. Gamma and related

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 53 | `gamma` | `gamma` | `Gamma` | done | |
| 54 | `gammaln` | `gammaln` | `LogGamma`, `LogGammaSign` | done | |
| 55 | `gammasgn` | | `GammaSign` | done | |
| 56 | `gammainc`, `gammaincc` | `gammainc` (`'lower'`/`'upper'`) | `GammaP`, `GammaQ` | done | |
| 57 | `gammaincinv`, `gammainccinv` | `gammaincinv` | `GammaPInv`, `GammaQInv` | done | |
| 58 | `beta`, `betaln` | `beta`, `betaln` | `Beta`, `LogBeta` | done | |
| 59 | `betainc`, `betaincc` | `betainc` (`'lower'`/`'upper'`) | `BetaInc`, `BetaIncC` | done | |
| 60 | `betaincinv`, `betainccinv` | `betaincinv` | `BetaIncInv`, `BetaIncCInv` | done | |
| 61 | `psi`, `digamma` | `psi` | `Digamma` | done | |
| 62 | `polygamma` | `psi(k,x)` | `Polygamma` | done | |
| 63 | `rgamma` | | | missing | none (`1/Gamma` with exact zeros at poles) |
| 64 | `multigammaln` | | | missing | none (sum of `LogGamma`) |
| 65 | `poch` | `pochhammer` (Symbolic) | | missing | none (ratio via `LogGamma`, exact for integers) |
| 66 | `loggamma` (complex principal branch) | | | missing | complex, see C1 |

## 6. Error function and Fresnel

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 67 | `erf`, `erfc` | `erf`, `erfc` | `Erf`, `Erfc` | done | |
| 68 | `erfcx` | `erfcx` | `Erfcx` | done | |
| 69 | `erfinv`, `erfcinv` | `erfinv`, `erfcinv` | `ErfInv`, `ErfcInv` | done | |
| 70 | `erfi` | `erfi` (Symbolic) | | missing | Faddeeva / Dawson (row 72) |
| 71 | `wofz` | | | missing | Faddeeva package (Johnson) port; complex, see C1 |
| 72 | `dawsn` | | | missing | Rybicki or Cephes `dawsn` |
| 73 | `fresnel` | `fresnels`, `fresnelc` (Symbolic) | | missing | Cephes `fresnl` |
| 74 | `modfresnelp`, `modfresnelm` | | | missing | Zhang-Jin (specfun) |
| 75 | `fresnel_zeros`, `fresnelc_zeros`, `fresnels_zeros`, `erf_zeros` | | | missing | complex Newton |
| 76 | `voigt_profile` | | | missing | `wofz` (row 71) |

## 7. Legendre, ellipsoidal, orthogonal polynomials

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 77 | `legendre_p`, `legendre_p_all` | `legendre` | | missing | three-term recurrence |
| 78 | `assoc_legendre_p`, `assoc_legendre_p_all` | `legendre` | | missing | recurrence, all norm/branch options |
| 79 | `sph_legendre_p`, `sph_legendre_p_all`, `sph_harm_y`, `sph_harm_y_all` | | | missing | recurrence; complex result for `sph_harm_y` (C1) |
| 80 | `lpmv`, `lqn`, `lqmn` | | | missing | Zhang-Jin (specfun) |
| 81 | `ellip_harm`, `ellip_harm_2`, `ellip_normal` | | | missing | Lame eigenproblem (eigenvalues: go-ndarray linalg) |
| 82 | `eval_legendre`, `eval_chebyt`, `eval_chebyu`, `eval_chebyc`, `eval_chebys` | `legendreP`, `chebyshevT`, `chebyshevU` (Symbolic) | | missing | recurrences, hypergeometric for non-integer degree |
| 83 | `eval_jacobi`, `eval_gegenbauer`, `eval_laguerre`, `eval_genlaguerre`, `assoc_laguerre`, `eval_hermite`, `eval_hermitenorm` | `jacobiP`, `gegenbauerC`, `laguerreL`, `hermiteH` (Symbolic) | | missing | recurrences; `hyp2f1` for non-integer degree (row 99) |
| 84 | `eval_sh_legendre`, `eval_sh_chebyt`, `eval_sh_chebyu`, `eval_sh_jacobi` | | | missing | affine map of rows 82-83 |
| 85 | `roots_legendre`, `roots_sh_legendre` | | `integrate.GaussLegendre` (other module) | missing | move or re-export; Golub-Welsch / Glaser-Liu-Rokhlin |
| 86 | `roots_chebyt`, `roots_chebyu`, `roots_chebyc`, `roots_chebys`, `roots_sh_chebyt`, `roots_sh_chebyu` | | | missing | closed-form nodes |
| 87 | `roots_jacobi`, `roots_sh_jacobi`, `roots_gegenbauer`, `roots_laguerre`, `roots_genlaguerre`, `roots_hermite`, `roots_hermitenorm` | | | missing | Golub-Welsch (symmetric tridiagonal eigenproblem) |
| 88 | `legendre`, `chebyt`, `chebyu`, `chebyc`, `chebys`, `jacobi`, `laguerre`, `genlaguerre`, `hermite`, `hermitenorm`, `gegenbauer`, `sh_*` (15 `orthopoly1d` constructors) | | | missing | polynomial object: go-numerics/polynomial if it exists, else own coefficient slice |

## 8. Hypergeometric, parabolic cylinder, Mathieu, spheroidal, Kelvin

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 89 | `hyp2f1` | `hypergeom` (Symbolic) | | missing | Gauss series + transformations (Pearson 2017; SciPy's C++); complex z (C1) |
| 90 | `hyp1f1` | `hypergeom`, `kummerU` | | missing | series, Kummer, asymptotics (Boost / Cephes) |
| 91 | `hyperu` | `kummerU` (Symbolic) | | missing | needs `hyp1f1` |
| 92 | `hyp0f1` | | | missing | Bessel `Iv`/`Jv` closed form |
| 93 | `pbdv`, `pbvv`, `pbwa`, `pbdv_seq`, `pbvv_seq`, `pbdn_seq` | | | missing | Zhang-Jin (specfun) |
| 94 | `mathieu_a`, `mathieu_b`, `mathieu_even_coef`, `mathieu_odd_coef` | | | missing | continued-fraction / tridiagonal eigenproblem |
| 95 | `mathieu_cem`, `mathieu_sem`, `mathieu_modcem1`, `mathieu_modcem2`, `mathieu_modsem1`, `mathieu_modsem2` | | | missing | needs row 94 |
| 96 | `pro_ang1`, `pro_rad1`, `pro_rad2`, `obl_ang1`, `obl_rad1`, `obl_rad2`, `pro_cv`, `obl_cv`, `pro_cv_seq`, `obl_cv_seq` and the six `*_cv` forms | | | missing | Zhang-Jin (specfun) spheroidal port |
| 97 | `kelvin`, `ber`, `bei`, `berp`, `beip`, `ker`, `kei`, `kerp`, `keip` | | | missing | series + asymptotics (specfun), or complex Bessel (C1) |
| 98 | `kelvin_zeros`, `ber_zeros`, `bei_zeros`, `berp_zeros`, `beip_zeros`, `ker_zeros`, `kei_zeros`, `kerp_zeros`, `keip_zeros` | | | missing | needs row 97 |

## 9. Combinatorics, Lambert W, other

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 99 | `comb`, `perm` | `nchoosek`, `perms` (permutations list) | `Comb`, `Perm` | done | |
| 100 | `binom` | `nchoosek` | `Binom` | done | |
| 101 | `factorial`, `factorial2` | `factorial` | `Factorial`, `Factorial2`, `LogFactorial` | done | |
| 102 | `factorialk` | | | missing | none |
| 103 | `stirling2` | | | missing | recurrence, exact via big integers |
| 104 | `lambertw`, `wrightomega` | `lambertw` (Symbolic) | | missing | Halley iteration; complex branches k != 0 need C1 |
| 105 | `agm` | | | missing | none |
| 106 | `bernoulli`, `euler` | `bernoulli` (Symbolic) | | missing | none (recurrence; exact via math/big for large n) |
| 107 | `diric` | `diric` (Signal) | | missing | none |
| 108 | `expn`, `exp1`, `expi` | `expint` | | missing | Cephes `expn`, `e1`, `ei`; complex `exp1` (C1) |
| 109 | `shichi`, `sici` | `sinint`, `cosint` (Symbolic) | | missing | Cephes `shichi`, `sici` |
| 110 | `spence` | `polylog(2,x)` (Symbolic) | | missing | Cephes `spence`; complex (C1) |
| 111 | `zeta` | `zeta` (Symbolic) | `Zeta`, `HurwitzZeta` | done | |
| 112 | `zetac` | | | missing | `Zeta - 1` without cancellation (Cephes `zetac`) |
| 113 | `softplus` | | `Log1pExp` | done | |
| 114 | `softmax`, `log_softmax`, `logsumexp` | `softmax` (Deep Learning) | `Softmax`, `LogSoftmax`, `LogSumExp` | done | axis/array forms live in `nd`; `logsumexp` options `b=` (weights) and `return_sign` not provided |
| 115 | `xlogy`, `xlog1py` | | `Xlogy`, `Xlog1py` | done | |

## 10. Convenience functions

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 116 | `log1p`, `expm1`, `cbrt`, `exp2` | `log1p`, `expm1`, `nthroot`, `pow2` | `math.Log1p`, `math.Expm1`, `math.Cbrt`, `math.Exp2` | done | standard library, not re-exported |
| 117 | `exp10`, `radian`, `round` | | | missing | none |
| 118 | `cosdg`, `sindg`, `tandg`, `cotdg` | `cosd`, `sind`, `tand`, `cotd` | | missing | none (exact at multiples of 90 degrees) |
| 119 | `cosm1`, `powm1` | | | missing | none |
| 120 | `exprel` | | | missing | none |
| 121 | `sinc` | `sinc` (Signal) | | missing | none |

## 11. Cross-cutting rows

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| C1 | complex arguments of `gamma`, `loggamma`, `psi`, `erf`, `erfc`, `wofz`, `jv`, `yv`, `iv`, `kv`, `hankel*`, `hyp2f1`, `expi`, `exp1`, `lambertw`, `spence`, `sph_harm_y` | all accept complex | | missing | `complex128` API and kernels (AMOS for Bessel, Lanczos/Stirling for `loggamma`, Faddeeva for `erf`); a design decision first: parallel `ZXxx` functions or generics |
| C2 | array (ufunc) layer for every scalar above | all accept arrays | `nd` subpackage (31 kernels) and the 20 `...To` slice kernels | partial | each new scalar function needs its `nd` form to keep the broadcasting contract; only about half of the 67 existing scalars have one |

## 12. Beyond SciPy: Mathematica and MATLAB (Symbolic Math Toolbox) only

Functions that are in the Mathematica "Special Functions" guide, or in MATLAB's
Symbolic Math Toolbox, and have no `scipy.special` counterpart. Re-check the
list against the Wolfram Language reference before a slice is started. All
missing; dep is what each needs beyond this package.

| # | Mathematica | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| M1 | `PolyLog[n, z]` | `polylog` (Symbolic, MuPAD) | | missing | series, Bernoulli expansion, inversion; complex (C1) |
| M2 | `LerchPhi` | | | missing | `HurwitzZeta` + series |
| M3 | `HypergeometricPFQ` (general pFq), `Hypergeometric0F1Regularized` | `hypergeom` | | missing | `hyp2f1`/`hyp1f1` first |
| M4 | `MeijerG`, `FoxH` | `meijerG` | | missing | pFq sums, Mellin-Barnes |
| M5 | `WhittakerM`, `WhittakerW` | `whittakerM`, `whittakerW` | | missing | `hyp1f1`, `hyperu` |
| M6 | `EllipticTheta`, `EllipticThetaPrime`, `JacobiTheta` | | | missing | q-series |
| M7 | `JacobiSN/CN/DN` (+ 9 more Glaisher), `JacobiZeta`, `JacobiAmplitude` | `ellipj` covers sn, cn, dn only | | missing | `ellipj` (row 8) |
| M8 | `EllipticPi`, `EllipticF`, `EllipticE[phi, m]` | `ellipticPi`, `ellipticF` | | missing | Carlson `RJ`/`RF` (row 9) |
| M9 | `WeierstrassP`, `WeierstrassSigma`, `WeierstrassZeta`, `WeierstrassHalfPeriods`, `WeierstrassInvariants` | `weierstrassP` | | missing | complex lattice, theta functions |
| M10 | `CoulombF`, `CoulombG`, `CoulombH` | | | missing | Steed's / Barnett algorithm |
| M11 | `ClebschGordan`, `ThreeJSymbol`, `SixJSymbol`, `NineJSymbol` | | | missing | Racah formulas; exact via integers |
| M12 | `MarcumQ`, `Clausen`/`ClausenC`/`ClausenS`, `Polylog` special cases | | | missing | series |
| M13 | `StruveH`, `StruveL` (complex), `AngerJ`, `WeberE`, `AiryAiPrime` family | | | missing | rows 1, 28 |
| M14 | `HarmonicNumber`, `Fibonacci`, `LucasL`, `StirlingS1`, `BellB`, `BernoulliB[n, x]`, `EulerE[n, x]`, `CatalanNumber`, `PartitionsP`, `PartitionsQ` | `bernoulli`, `euler` (Symbolic) | | missing | `math/big` for exact values |
| M15 | `ZetaZero`, `RiemannSiegelZ`, `RiemannSiegelTheta`, `DirichletL`, `DirichletEta`, `LogGamma` complex | | | missing | Riemann-Siegel, Euler-Maclaurin |
| M16 | `BarnesG`, `Gamma[a, z0, z1]` (generalized incomplete), `Glaisher`, `Stieltjes` | | | missing | `loggamma` complex |
| M17 | `InverseErf`/`InverseErfc` complex, `InverseGammaRegularized` complex, `InverseBetaRegularized` generalized | | | missing | C1 |
| M18 | `SphericalBesselJ/Y`, `SphericalHankelH1/H2` | | | missing | row 26 |
| M19 | `LegendreQ`, `LegendreP[n, m, type 2/3]`, `SphericalHarmonicY` | `legendre` | | missing | rows 77-80 |
| M20 | `MathieuC/S`, `SpheroidalPS/QS` ... | | | missing | rows 94-96 |
| M21 | `DedekindEta`, `ModularLambda`, `KleinInvariantJ`, `EllipticNomeQ`, `InverseEllipticNomeQ` | | | missing | theta functions |
| M22 | `Gudermannian`, `InverseGudermannian`, `LambertW[k, z]`, `ProductLog`, `Sinc` variants, `FresnelF`, `FresnelG`, `SinIntegral`, `CosIntegral`, `SinhIntegral`, `CoshIntegral`, `LogIntegral`, `ExpIntegralE`, `ExpIntegralEi`, `Erfi`, `DawsonF` | `sinint`, `cosint`, `expint`, `erfi` | | missing | rows 72, 73, 104, 108, 109 |

## Proposed slices

Ordered by cost per function gained; each is one PR with accuracy fixtures
against mpmath, as for the existing functions.

1. **Thin wrappers (no new maths).** Rows 30, 33, 35 (`gdtr`/`gdtrc`), 36,
   39, 42, 43, 44 (the `...tr`/`...tri` pairs over `BetaInc`, `GammaP`,
   `Ndtr`), 47, 51, 52, 63, 64, 65, 102, 105-107, 112, 117-121, plus rows 82-84
   integer-degree recurrences. About 60 names. Dep: none.
2. **Inverse-in-parameter CDFLIB family.** Rows 31, 32, 34, 35 (`ria`/`rib`/`rix`),
   42 (`pdtrik`), 43, 44, 45, 49. Dep: a 1-D bracketing root finder
   (`go-numerics/optimize` Brentq, or a private copy to avoid a module cycle).
3. **Non-central distributions.** Rows 37, 38, 45 (`ncf`, `nct`, `ncx2` cdf and
   inverses). Dep: slice 2.
4. **Exponential-integral and Fresnel group.** Rows 70, 72, 73, 108, 109, 110
   (`expn`, `exp1`, `expi`, `shichi`, `sici`, `spence`, `dawsn`, `fresnel`,
   `erfi`). Cephes ports; dep: none.
5. **Carlson and Jacobi elliptic.** Rows 7, 8, 9, M7, M8. Dep: none.
6. **Airy, Struve, Bessel extras.** Rows 1-3, 19, 20, 23-29. Dep: Zhang-Jin
   specfun ports, existing Bessel.
7. **Hypergeometric core.** Rows 89-92, 83 (non-integer degree), M3, M5.
   Dep: none; the hardest numerically (Pearson 2017 transformations).
8. **Orthogonal polynomials and Gauss rules.** Rows 77-88. Dep:
   symmetric tridiagonal eigensolver for Golub-Welsch (go-ndarray linalg) and
   a polynomial type.
9. **Complex layer (C1).** Design decision first; then `loggamma`, `erf`
   family via Faddeeva, AMOS Bessel, `hankel*`, Kelvin (rows 15, 66, 71, 97,
   104). Dep: `complex128` conventions agreed with go-ndarray.
10. **Specfun tail.** Rows 93-96, 98 (parabolic cylinder, Mathieu, spheroidal,
    Kelvin zeros). Dep: Zhang-Jin port; lowest demand.
11. **Mathematica-only (section 12).** After slices 5, 7, 9; exact-integer
    functions (M14) can go any time with `math/big`.
