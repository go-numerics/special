# Speed

The comparison is with SciPy 1.18.1's vectorised ufuncs (C/C++, one thread),
measured by [refbench](https://github.com/go-scikit/refbench) v0.3.0 from
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

## Status: the valid run of v0.13.0 (2026-11-13)

`perf.json` and [perf-table.md](perf-table.md) are a **valid** timing run: tag
v0.13.0 (commit 6577eb4e807cd3c0bd835d9ed5c17d607870f868), darwin/arm64, 16
CPUs, Go 1.27.4, SciPy 1.18.1, refbench v0.3.0 (`-perf.commit` the exact HEAD
of the clean tag checkout; only `bench/go.mod` and `go.sum` differ, by the
refbench require), 1-minute load over the rounds min 6.48, median 7.48, max
7.99 (gate 8); 151 of 151 rows valid, 11 rounds retried after ending above the
gate. Ratios are Go time over SciPy time; above 1 the Go loop is slower. They
are single-core ratios on 10 000-point calls; the `nd/` rows run on one
goroutine here. Changes against the run of v0.8.0 (2026-11-10):
`Jn` 1.52 to 0.18 and `Kn` 1.88 to 0.28 (v0.12.0 and v0.13.0, #33), `Yn` 33.4
to 2.37, `LogBeta` 1.20 to 0.58, `ErfInv` 1.12 to 0.49, `ErfcInv` 1.24 to
0.62, `Logit` 1.37 to 0.90, `Factorial` of non-integers 1.87 to 1.22.

- **Faster than SciPy** (ratio below 1, verdict faster): `Gamma` 0.84 (8.48
  against 10.08 ns per element), `nd.Gamma` 0.85, `LogGamma` 0.76, `GammaTo`
  0.78, `LogGammaTo` 0.67, `Ndtri` 0.57, `LogNdtr` 0.59, `Ndtr` 0.84, `Erf` 0.89
  (`ErfTo` 0.83, `ErfcTo` 0.91), `Digamma` 0.80, `Beta` 0.84, `LogBeta` 0.58,
  `ErfInv` 0.49, `ErfcInv` 0.62, `Erfcx` 0.99, `Log1pExp` 0.92, `Zeta` 0.64,
  `HurwitzZeta` 0.66, `Polygamma` 0.05, `J0`/`J1`/`Y0`/`Y1` 0.71–0.75, `I0`
  0.91, `I0e` 0.30, `I1e` 0.35, `K0` 0.99, `K0e` 0.47, `K1` 0.93, `K1e` 0.37,
  the complete elliptic integrals 0.37–0.43, `Comb` 0.85, `Factorial` of
  integers 0.08, `Factorial2` 0.03, `JnZeros` 0.89, `Jn` 0.18 (130 against 736
  ns per element) and `Kn` 0.28 (102 against 363 ns).
- **Ties** (verdict tie, or within 3 percent): `Logit` 0.90, `I1` 0.97,
  `Erfc` 1.03, `nd.Erfc` 1.03, `nd.K0` 1.03, `nd.Comb` 1.02, `nd.Logit` 0.99.
  `nd.I1` measured 1.52 in this run with a wide interquartile range (1.78e5 to
  1.94e5 ns against 1.15e5 for the scalar loop of the same kernel; 1.02 in the
  v0.8.0 run): unexplained; rerun needed.
- **Still slower, 1.08 to 1.54 times**: `LogExpit` 1.08, `Xlogy` 1.08, `GammaQ`
  1.13, `GammaP` 1.18, `Expit` 1.20, `GammaSign` 1.22, `Perm` 1.22, `Factorial`
  of non-integers 1.22, `Yv` 1.22, `Jv` 1.28, `LogSumExp` 1.47, `GammaQInv`
  1.48, `GammaPInv` 1.54 (the `nd/` rows of `GammaSign` 1.67, `Expit` 1.36,
  `Perm` 1.55).
- **Open gaps** (each a double-double evaluation whose result is correctly
  rounded; see #33 for the profile of each): `Iv` 7.94, `In` 5.24, `Softmax`
  4.84, `LogSoftmax` 4.13, `BetaIncInv`/`BetaIncCInv` 3.45, `Binom` 3.03,
  `Yn` 2.37 (91 against 38 ns per element), `Kve` 2.37, `Kv` 2.32,
  `BetaInc`/`BetaIncC` 2.1, `Xlog1py` 1.91, `Ive` 1.69.

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

A plain Go loop is faster than SciPy's C on `Gamma`, `LogGamma`, `LogBeta`,
`Digamma`, `Erf`, `ErfInv`, `ErfcInv`, `Ndtr`, `Ndtri` and `LogNdtr` (table
kernels, #33), within 1.0–1.5 times on the other elementwise families, faster
on the Bessel functions of order 0 and 1, on `Jn` and `Kn` and on the
elliptic, zeta and polygamma functions, and several times slower where the
result needs a double-double continued fraction or series (`Yn`, `In`, `Iv`,
`Kv`, the incomplete beta function and its inverses, the softmax reductions).
That gap is the price of results rounded correctly where SciPy's errors run
from 2 to 1e15 ulp (see accuracy.md).
