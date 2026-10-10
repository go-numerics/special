# Speed

The comparison is with SciPy 1.18.1's vectorised ufuncs (C/C++, one thread)
on the same 10^6 points per function.

## Method

- `nd/perf_test.go` (`go test -run TestPerfReport -perf.report=...` in
  `nd/`) starts `bench/scipy_server.py` once through `uv` and drives it
  over a pipe. For every function and every round it times, back to back:
  the SciPy call on its 10^6-point array, a plain Go loop calling the scalar
  function on the same points, and the `nd` array function (which splits
  the work across `GOMAXPROCS` goroutines).
- The points are generated identically on both sides
  (`lo + (hi-lo)·frac(0.618…·i)`, the same IEEE operations), over a
  representative range per function (listed in both files).
- Rounds are interleaved function by function, so a change in the load of
  the host hits both sides alike; the table reports the median of the
  rounds. SciPy's time includes allocating its result array, as the Go side
  does.
- The host is shared and was heavily loaded (load average in the table
  header, on 16 cores). Absolute times are therefore inflated for both
  sides and the parallel column in particular understates what an idle
  machine gives; the single-thread ratio is the more robust number.

## Results

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
