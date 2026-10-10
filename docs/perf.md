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

PERF_TABLE_PLACEHOLDER

## Reading the table

PERF_READING_PLACEHOLDER
