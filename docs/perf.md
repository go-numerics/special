# Speed

The comparison is with SciPy 1.18.1's vectorised ufuncs (C/C++, one thread),
measured by [refbench](https://github.com/go-scikit/refbench) v0.4.0 from
`bench/perf_test.go`. The generated table is [perf-table.md](perf-table.md)
(data: `perf.json`); the public functions without a row are in
[perf-exempt.md](perf-exempt.md).

## Multicore

The tables below are single core. The `nd` functions also split one call across
goroutines, by a policy measured on `nd/parallel_bench_test.go`
(`BenchmarkPar`; Apple M4 Max, 12 P + 4 E cores, interleaved with abbench at
1-minute load at most 6, 6 to 8 rounds, GOMAXPROCS 1 against 16).

Per element, one core: `GammaSign` 3.3 ns, `Expit` 7, `Erf` 9, `Gamma` 12,
`BetaInc` 460, `BetaIncInv` 2100. One threshold in elements cannot serve costs
400 times apart, so each function has a cost class (`costCheap` 4 ns,
`costLight` 12 ns, `costMedium` 60 ns for scalar functions of unknown cost,
`costHeavy` 500 ns for the incomplete beta and gamma family and their
inverses), and a call is parallel when `n x cost` is at least 40 µs, with at
most `n x cost / 20 µs` workers. The workers claim blocks (at least 8 µs of
work, eight per worker) from an atomic counter instead of one equal chunk
each, so a slow core or an expensive run of elements delays only its own block.
Every element is computed by one call of the same function, so the result does
not depend on the split.

Crossover measured with the previous policy (one threshold of 8192 elements,
16 equal chunks), 16 cores against 1, time ratio:

| n | GammaSign | Expit | Erf | Gamma | BetaInc | BetaIncInv |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1.0 (serial) | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 4096 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 8192 | 1.12 slower | 0.75 | 0.66 | 0.55 | 0.13 | 0.12 |
| 16384 | 0.98 | 0.64 | 0.56 | 0.44 | 0.13 | 0.12 |
| 65536 | 0.36 | 0.36 | 0.37 | 0.25 | 0.13 | 0.12 |

With the cost-aware policy (same machine, 16 cores, time against the previous
policy): `BetaInc` at n = 1024 and 4096 takes 0.18 and 0.15 of the time
(467 to 86 µs, 1938 to 288 µs), `BetaIncInv` 0.16 and 0.13, `Gamma` at 4096
0.79, `Erf` at 4096 0.88, `GammaSign` at 8192 0.83 (it is now serial there,
27 µs of work), and no row measured slower.

The cheap kernels stop scaling near 3x on 16 cores (allocation and memory
bandwidth); the expensive ones reach 8x, which is what 12 P and 4 E cores give
on this code.

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

## Update 2026-11-23: table re-measured with the five new kernels

`perf.json` and [perf-table.md](perf-table.md) are now a new valid run, taken
on the branch that adds `GammaSignTo`, `Factorial2To`, `PermTo`, `BinomTo` and
`Xlog1pyTo` (commit 7c12baa4047fe0f8e4fd32bb31897f597cd286ea, darwin/arm64,
Go 1.27.4, SciPy 1.18.1, 1-minute load min 2.71, median 3.07, max 6.30; 156 of
156 rows valid, 0 retries). The ratios quoted in the next section are those of
the v0.20.0 run and are not refreshed; the table is the current data.

## Status: the valid run of v0.20.0 (2026-11-16)

`perf.json` and [perf-table.md](perf-table.md) are a **valid** timing run: tag
v0.20.0 (commit d415e46012b1f640eff1e03f2b12b475a1dc7229), darwin/arm64, 16
CPUs, Go 1.27.4, SciPy 1.18.1, refbench v0.4.0 (`-perf.commit` the exact HEAD
of the clean tag checkout; only `bench/go.mod` and `go.sum` differ, by the
refbench require; the root module is byte-identical to the tag), 1-minute load
over the rounds min 4.03, median 5.68, max 7.97 (gate 8); 151 of 151 rows
valid, 2 rounds retried after ending above the gate. **SciPy is the only
reference timed.** The Julia worker (SpecialFunctions.jl, Bessels.jl) and the
worst-ulp columns were not run: the bench of v0.20.0 has no wiring for them (no
`Refs`, no Julia project, no accuracy scorer), so the table has no
other-reference section and no per-side ulp; accuracy is in
[accuracy-table.md](accuracy-table.md). The `*Exact` softmax functions have no
row. Ratios are Go time over SciPy time; above 1 the Go loop is slower. They
are single-core ratios on 10 000-point calls; the `nd/` rows run on one
goroutine here. Changes against the run of v0.13.0 (2026-11-13): `Kn` 0.28 to
0.11, `Jv` 1.28 to 0.47, `Yv` 1.22 to 0.54, `Kv` 2.32 to 0.60, `Kve` 2.37 to
0.58, `Ive` 1.69 to 0.37, `In` 5.24 to 0.86, `Iv` 7.94 to 1.77, `Softmax` 4.84
to 1.05, `LogSoftmax` 4.13 to 1.09, `LogSumExp` 1.47 to 0.35, `Yn` 2.37 to
2.34.

- **Faster than SciPy** (ratio below 1, verdict faster): `Gamma` 0.83 (8.37
  against 10.03 ns per element), `nd.Gamma` 0.84, `LogGamma` 0.76, `GammaTo`
  0.78, `LogGammaTo` 0.67, `Ndtri` 0.57, `LogNdtr` 0.60, `Ndtr` 0.84, `Erf` 0.88
  (`ErfTo` 0.83, `ErfcTo` 0.91), `Digamma` 0.80, `Beta` 0.83, `LogBeta` 0.59,
  `ErfInv` 0.49, `ErfcInv` 0.62, `Logit` 0.89, `Log1pExp` 0.92, `Zeta` 0.65,
  `HurwitzZeta` 0.66, `Polygamma` 0.05, `J0`/`J1`/`Y0`/`Y1` 0.71–0.76, `I0`
  0.91, `I0e` 0.29, `I1` 0.97, `I1e` 0.36, `K0` 0.93, `K0e` 0.41, `K1` 0.92,
  `K1e` 0.37, the complete elliptic integrals 0.37–0.46, `Comb` 0.86,
  `Factorial` of integers 0.08, `Factorial2` 0.03, `JnZeros` 0.89, `Jv` 0.47,
  `Yv` 0.54, `Kv` 0.60, `Kve` 0.58, `Ive` 0.37, `In` 0.86, `Jn` 0.18 (128
  against 722 ns per element), `Kn` 0.11 (40 against 357 ns) and `LogSumExp`
  0.35.
- **Ties** (verdict tie): `Erfc` 1.00, `Erfcx` 0.98, `LogExpit` 1.08,
  `nd.K0` 0.98, `nd.Comb` 1.00.
- **Still slower, 1.01 to 1.54 times** (open perf defects): `nd.Erfc` 1.01,
  `nd.I1` 1.01, `Softmax` 1.05, `Xlogy` 1.07, `LogSoftmax` 1.09, `GammaQ` 1.13,
  `GammaP` 1.17, `Perm` 1.20, `ExpitTo` 1.20, `Expit` 1.23, `Factorial` of
  non-integers 1.23, `GammaSign` 1.24, `GammaQInv` 1.48, `GammaPInv` 1.54 (the
  `nd/` rows of `Softmax` 1.20, `LogSoftmax` 1.26, `Expit` 1.34, `Xlogy` 1.14,
  `LogExpit` 1.17, `Logit` 1.06, `Perm` 1.46, `Factorial` 1.28, `GammaSign`
  1.75).
- **Open gaps** (each a double-double evaluation whose result is correctly
  rounded; see #33 for the profile of each; open perf defects): `BetaIncInv`
  and `BetaIncCInv` 3.49 and 3.48, `Binom` 3.04, `Yn` 2.34 (88 against 38 ns
  per element), `BetaInc`/`BetaIncC` 2.12 and 2.14, `Xlog1py` 1.89 (`nd`
  2.11), `Iv` 1.77.

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
- `Softmax`, `LogSoftmax`, `LogSumExp` (#33, this change): the v0.13.0 profile of
  `Softmax` was `dd.ExpMant` 49 % cumulative, `math.Ldexp` 18 %, `dd.Add` 8 %,
  the certified quotient of #48 11 %. Now `floats.Exp`, an exact shift
  x - m = h + l, a compensated sum and one reciprocal: 144 us to about 31 us on
  10 000 points (in-repo benchmark, M4 Max, load above 8, so indicative; refbench
  not rerun). Relative error at most 8u, 5u, 4u (accuracy.md); the results are no
  longer correctly rounded. The remaining cost is `floats.Exp`, a scalar loop
  (13.6 of 31 us).
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
