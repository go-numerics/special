# Speed

The comparison is with SciPy 1.18.1's vectorised ufuncs (C/C++, one thread),
measured by [refbench](https://github.com/go-scikit/refbench) v0.1.1 from
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

## Status (2026-11-07)

The committed `perf.json` is **deferred**: allocation counts only (the host's
1-minute load stayed between 8 and 50 during the session). `BetaIncInv` and
`BetaIncCInv` allocated once per call (a closure in the iteration, 10 000 per
10 000-point row); the iteration's evaluator is now a value with a method and
the rows record 0 (`TestBetaIncInvAllocatesNothing`). The speed effect is
unmeasured (load above 8). The `nd/` rows allocate 8 to 22 objects per array
call.

2026-11-08: the 25 rows listed in the previous exemptions as follow-ups (integer
orders, combinatorics, reductions, scalar and `nd/`) have allocation counts in
`perf.json`; their timing is unmeasured for the same reason (load 60 to 80).

## What is done, and what is not (#33)

- Done, speed unmeasured (load above 8): `BetaIncInv`/`BetaIncCInv` allocate
  nothing (#36); `nd.Map2`/`Map3` run one direct loop for same-shape and scalar
  operands, bit-identical to the old path (#37).
- Not done: a float64 fast path for the double-double functions (`BetaInc`,
  `Jv`, `Kv`, ...). The profile puts `dd.Mul`/`FastTwoSum` on top, but a fast
  path is acceptable only if a certified error bound decides when float64 is
  already correctly rounded and the double-double path runs otherwise, so that
  the bits do not move. No such bound has been derived for any of these
  functions, so none was added.
- Done, speed unmeasured (load above 8): `Gamma` on 0 < x < 1 and x < 0
  runs a float64 evaluation with an a-priori error bound and falls back to
  the double-double path only when the bound cannot decide the rounding, so
  the results are bit-identical (derivation and fallback rates:
  [gamma-fastpath.md](gamma-fastpath.md)); `sinPi`/`cosPi` reduce their
  argument with an exact `a - 2⌊a/2⌋` instead of `math.Mod` (same values).
  The refbench `Gamma` row, (0.1, 170), was already float64 for 94 % of its
  points (x >= 10, since #7). The 45.5 ns/elem in the earlier table below
  predates #7, which moved `Gamma` onto the float64 kernels.
- Timing: the first `-perf.run` at a load below 8 replaces the deferred record.

## Earlier table (before refbench; not a verdict)

The table below is the pre-refbench measurement: SciPy against a Go scalar loop
(function call per element) and `nd` on 16 goroutines, on a loaded host
(load 7.7 at the start, 23 during). It is the record of the gap the campaign
closes (0.07x to 0.70x single-thread), not a current number.

1000000 points per call, median of 5 interleaved rounds; darwin/arm64, GOMAXPROCS=16; host load at start: 23:36  up 17 days, 10:34, 1 user, load averages: 7.67 16.16 32.74

| function | SciPy (ns/elem) | Go scalar loop (ns/elem) | Go nd parallel (ns/elem) | scalar speed-up | parallel speed-up |
|---|--:|--:|--:|--:|--:|
| Gamma | 13.4 | 45.5 | 6.4 | 0.30x | 2.10x |
| LogGamma | 7.0 | 26.1 | 3.9 | 0.27x | 1.78x |
| Digamma | 9.1 | 22.6 | 3.3 | 0.40x | 2.75x |
| Beta | 43.1 | 149.8 | 20.4 | 0.29x | 2.11x |
| LogBeta | 24.4 | 132.0 | 17.3 | 0.19x | 1.41x |
| Erf | 8.8 | 27.7 | 3.6 | 0.32x | 2.45x |
| Erfc | 7.9 | 35.1 | 4.4 | 0.22x | 1.78x |
| Erfcx | 5.5 | 19.8 | 2.7 | 0.28x | 2.09x |
| ErfInv | 10.1 | 45.9 | 5.9 | 0.22x | 1.71x |
| ErfcInv | 9.1 | 45.3 | 5.9 | 0.20x | 1.53x |
| Ndtr | 9.0 | 35.5 | 4.7 | 0.25x | 1.90x |
| Ndtri | 7.7 | 70.1 | 8.8 | 0.11x | 0.88x |
| LogNdtr | 9.2 | 71.2 | 9.1 | 0.13x | 1.01x |
| Expit | 3.8 | 27.4 | 3.9 | 0.14x | 0.98x |
| Logit | 5.1 | 39.6 | 5.3 | 0.13x | 0.95x |
| Log1pExp | 8.0 | 24.6 | 3.4 | 0.33x | 2.36x |
| Xlogy | 3.8 | 14.8 | 3.3 | 0.26x | 1.13x |
| Zeta | 15.1 | 21.5 | 3.0 | 0.70x | 5.09x |
| HurwitzZeta | 130.8 | 988.7 | 113.1 | 0.13x | 1.16x |
| Polygamma1 | 136.0 | 221.3 | 28.1 | 0.61x | 4.85x |
| GammaP | 136.5 | 301.5 | 36.5 | 0.45x | 3.74x |
| GammaPInv | 469.0 | 1498.7 | 131.4 | 0.31x | 3.57x |
| BetaInc | 234.8 | 3241.7 | 305.6 | 0.07x | 0.77x |
| J0 | 12.6 | 130.4 | 16.8 | 0.10x | 0.75x |
| Y0 | 12.9 | 143.3 | 17.1 | 0.09x | 0.76x |
| I0 | 15.8 | 199.9 | 25.6 | 0.08x | 0.62x |
| K0 | 15.6 | 72.8 | 9.3 | 0.21x | 1.67x |
| Jv | 1347.6 | 7427.2 | 717.0 | 0.18x | 1.88x |
| Kv | 494.5 | 6923.1 | 651.3 | 0.07x | 0.76x |
| EllipK | 9.5 | 97.4 | 12.4 | 0.10x | 0.77x |
| EllipE | 8.8 | 97.0 | 12.4 | 0.09x | 0.71x |

## Reading the table

- **Single thread, the price of accuracy.** A plain Go loop is 1.4x to 15x
  slower than SciPy's C: about 3x for the gamma and error-function families,
  5–9x for the inverse normal and the logistic helpers, 10–15x for the
  Bessel functions and `BetaInc`. Almost all of it is double-double
  arithmetic, which is what turns SciPy's errors of 2–1e15 ulp into about
  half an ulp (see accuracy.md). Zeta is the closest (1.4x).
- **Across cores, faster.** The `nd` functions split large arrays across
  goroutines; on this 16-core host (lightly loaded at the start of the run)
  they beat SciPy for 21 of the 31 functions, typically 1.5–3x. They do
  not for `Expit`, `Logit` and `Ndtri` (0.88–0.98x, near parity) nor for
  `J0`, `Y0`, `I0`, `Kv`, `BetaInc`, `EllipK`, `EllipE` (0.62–0.77x), where
  the single-thread gap is larger than what the cores recover on a shared
  host.
- **The slow outliers.** `Jv`, `Yv`, `Kv` (7 µs per call) run Steed's
  continued fractions in double-double; `HurwitzZeta` (1 µs) sums up to
  ~0.7(s+30) Euler–Maclaurin terms with a double-double power each;
  `BetaInc` (3 µs) carries TOMS 708 with double-double prefixes. These are
  the first candidates for float64 fast paths where an error analysis shows
  the double-double is not needed.
