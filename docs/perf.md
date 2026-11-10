# Speed

The comparison is with SciPy 1.18.1's vectorised ufuncs (C/C++, one thread),
measured by [refbench](https://github.com/go-scikit/refbench) v0.1.2 from
`bench/perf_test.go`. The generated table is [perf-table.md](perf-table.md)
(data: `perf.json`); the public functions without a row are in
[perf-exempt.md](perf-exempt.md).

## Method

- **One core on both sides.** refbench sets `GOMAXPROCS=1` around the run and
  starts the SciPy worker with `OMP/OPENBLAS/MKL/VECLIB_*_THREADS=1`. The `nd/`
  rows therefore run on one goroutine; the earlier multi-core column is gone.
- **Like with like: one call over an array on both sides.** Every row times one
  call over the same 10 000 points. The SciPy body is one ufunc call (result
  array allocated by the call, as the Go side allocates its own); the ufunc's
  per-call overhead (about 1 µs) is amortised over the points. Three Go bodies
  are timed against the same SciPy call: the scalar function in a loop over the
  points (`Gamma`), the public array kernel (`GammaTo`: the Go counterpart of
  a ufunc loop, no call per element), and the `nd` wrapper (`nd/Gamma`,
  kernel plus allocation and dtype handling). A scalar loop calls the function
  once per element through a function value; SciPy's ufunc loop does the same
  through a C function pointer, so that row is fair, and the kernel rows are
  the strict array-against-array comparison.
- **Same bytes.** The points are `lo + (hi-lo)·frac(i·0.618…)`; `go test -run
  TestDataParity -perf.parity` compares the SHA-256 of the Go arrays with
  NumPy's, row by row (all equal on 2026-11-07).
- **Integer orders, combinatorics, reductions** (`bench/perf_extra_test.go`).
  These rows carry their own Python setup, because their arguments are not all
  float64 arrays: `Jn`, `Yn`, `In`, `Kn` against `scipy.special.jn`, `yn`, `iv`,
  `kn` (orders `i mod 21` as an int64 array, x in (0.1, 50); the `nd/` rows pass
  the orders as float64, the values are the same); `Factorial` (non-integers,
  the gamma path, and a second row for integers, the table path), `Factorial2`,
  `Comb`, `Perm` against `factorial`, `factorial2`, `comb`, `perm` (Python
  wrappers: their call overhead is part of what SciPy costs); `JnZeros` against
  `jn_zeros(2, 20)`; `LogSumExp`, `Softmax`, `LogSoftmax` over 10 000 points
  against `logsumexp`, `softmax`, `log_softmax`. `go test -run TestRefSmoke
  -perf.smoke .` runs every SciPy expression once (all run on 2026-11-08) and
  `TestDataParity` hashes the integer arrays too. `LogFactorial` stays exempt:
  SciPy has no such function.
- **Interleaved rounds, load gate.** Rounds alternate Go-first and SciPy-first;
  refbench refuses to time at a 1-minute load of 8 or more, and the table says
  `deferred` when no window opened.

```
cd bench
go test -run TestPerf -cpu 1 .                 # CI: allocations, table, inventory
go test -run TestPerf -cpu 1 -perf.run .       # timing (writes perf.json and the table)
go test -run TestPerf -cpu 1 -perf.record .    # allocation counts only
```

## Status: the valid run of v0.6.0 (2026-11-10)

`perf.json` and [perf-table.md](perf-table.md) are a **valid** timing run: tag
v0.6.0 (commit f398d831c6451ca2cafac2a84c6e0bfc280c3916), darwin/arm64, 16
CPUs, Go 1.27.3, SciPy 1.18.1, refbench v0.1.2, 1-minute load over the rounds
min 4.79, median 6.43, max 7.87 (gate 8). Ratios are Go time over SciPy time;
above 1 the Go loop is slower. They are single-core ratios on 10 000-point
calls; the `nd/` rows run on one goroutine here and sit within a few percent
of their scalar rows. It replaces the run of v0.5.7 (2026-11-09), where `Gamma`
was 1.13 and `LogGamma` 1.36. Code of v0.6.1 (`Binom` fast path, #50) is not in
this run.

- **Faster than SciPy** (ratio below 1): `Gamma` 0.85 (8.11 against 9.59 ns
  per element), `nd.Gamma` 0.84, `LogGamma` 0.76, `GammaTo` 0.79,
  `LogGammaTo` 0.67, `Digamma` 0.80, `Beta` 0.83, `Erfcx` 1.00 (a tie;
  `nd.Erfcx` 0.92), `Log1pExp` 0.91, `Zeta` 0.64, `HurwitzZeta` 0.66,
  `Polygamma` 0.05, `J0`/`J1`/`Y0`/`Y1` 0.73–0.76, `I0` 0.91, `I0e` 0.30,
  `I1` 0.96, `I1e` 0.35, `K0` 0.98, `K0e` 0.43, `K1` 0.93, `K1e` 0.37, the
  complete elliptic integrals 0.37–0.44, `Comb` 0.85, `Factorial` of
  integers 0.08, `Factorial2` 0.03, `JnZeros` 0.96.
- **Still 1.1 to 1.5 times slower**, the elementwise families: `GammaSign`
  1.21, `LogBeta` 1.20, `Erf` 1.12, `Erfc` 1.23, `ErfInv` 1.11, `ErfcInv`
  1.25, `Ndtr` 1.15, `Ndtri` 1.45, `LogNdtr` 1.23, `Expit` 1.23, `Logit`
  1.39, `LogExpit` 1.08, `Xlogy` 1.09, `GammaP` 1.17, `GammaQ` 1.15, `Jv`
  1.27, `Yv` 1.22, `Perm` 1.26, `LogSumExp` 1.48, `Jn` 1.51,
  `GammaPInv`/`GammaQInv` 1.5.
- **Open gaps** (each a double-double evaluation whose result is correctly
  rounded; see #33 for the profile of each): `Yn` 33.5, `Iv` 7.9, `Binom`
  5.6, `In` 5.3, `Softmax` 5.1, `LogSoftmax` 4.1, `BetaIncInv`/`BetaIncCInv`
  3.5, `Kv`/`Kve` 2.3, `Xlog1py` 2.2, `BetaInc`/`BetaIncC` 2.1, `Factorial`
  of non-integers 1.87, `Kn` 1.88, `Ive` 1.70.

Allocations: `BetaIncInv` and `BetaIncCInv` allocate nothing
(`TestBetaIncInvAllocatesNothing`), `Softmax` since v0.5.6; the `nd/` rows
allocate 7 to 20 objects per array call.

## What was done (#33)

- `BetaIncInv`/`BetaIncCInv` allocate nothing (#36); `nd.Map2`/`Map3` run one
  direct loop for same-shape and scalar operands, bit-identical to the old
  path (#37).
- `Gamma` on 0 < x < 1 and x < 0 runs a float64 evaluation with an a-priori
  error bound and falls back to the double-double path only when the bound
  cannot decide the rounding, so the results are bit-identical (derivation and
  fallback rates: [gamma-fastpath.md](gamma-fastpath.md)); `sinPi`/`cosPi`
  reduce their argument with an exact `a - 2⌊a/2⌋` instead of `math.Mod`.
- v0.5.6: `I1`, `I1e`, `K1`, `K1e` from fitted octave interpolants (results
  change, more accurate); `Comb` by an exact integer recurrence, `Factorial2`
  from a table, `Factorial` (n >= 10) and `Xlog1py` through certified float64
  kernels, `GammaSign` without `math.Mod` (bit-identical).
- Not done: float64 fast paths for the remaining double-double functions. A
  fast path is acceptable only if a certified error bound decides when float64
  is already correctly rounded, so that the bits do not move; none has been
  derived for the open gaps above.

## Reading the table

A plain Go loop is faster than SciPy's C on `Gamma`, `LogGamma` and `Digamma`
since v0.6.0 (table kernels, #33), within 1.1–1.5 times on the other
elementwise families, faster on the Bessel functions of order 0 and 1 and on
the elliptic, zeta and polygamma functions, and several times slower where the
result needs a double-double continued fraction or series (integer-order
`Y`, `I`, `K`, the incomplete beta function and its inverses, the softmax
reductions). That gap is the price of results rounded correctly where SciPy's
errors run from 2 to 1e15 ulp (see accuracy.md).
