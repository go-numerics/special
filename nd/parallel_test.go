package nd

import (
	"math"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/go-numerics/special"
)

// TestParallelOnCoversEveryIndexOnce checks the chunking at the boundary
// sizes of every cost class: n = 0 and 1, just under, at and just over the
// serial threshold and the threshold of each worker count, and sizes that no
// block size divides. Every index must be visited exactly once (the race
// detector sees any overlap of two blocks), every range must lie in [0, n),
// and below the threshold the body must run once on the whole range.
func TestParallelOnCoversEveryIndexOnce(t *testing.T) {
	for _, c := range append([]cost{costUser}, costClasses...) {
		serial := parallelWork / int(c)
		perWorker := max(1, workerWork/int(c))
		sizes := []int{0, 1, 2, 7, serial - 1, serial, serial + 1, 2 * serial, 2*perWorker - 1, 2 * perWorker,
			3*perWorker + 1, 5*perWorker + 3, 16 * perWorker, 40*perWorker + 11, 100003}
		for _, w := range []int{1, 2, 3, 5, 16, 64} {
			for _, n := range sizes {
				if n < 0 {
					continue
				}
				seen := make([]int, n)
				var mu sync.Mutex
				calls := 0
				parallelOn(w, n, c, func(lo, hi int) {
					mu.Lock()
					calls++
					mu.Unlock()
					if lo < 0 || hi > n || lo >= hi && n > 0 {
						t.Errorf("cost %d w %d n %d: range [%d, %d)", c, w, n, lo, hi)
						return
					}
					for i := lo; i < hi; i++ {
						seen[i]++
					}
				})
				for i, k := range seen {
					if k != 1 {
						t.Fatalf("cost %d w %d n %d: index %d visited %d times", c, w, n, i, k)
					}
				}
				if (w == 1 || n < serial) && calls != 1 {
					t.Errorf("cost %d w %d n %d: %d calls below the threshold, want 1", c, w, n, calls)
				}
				if w >= 2 && n >= 2*perWorker && n >= serial && n >= 4*max(1, blockWork/int(c)) && calls < 2 {
					t.Errorf("cost %d w %d n %d: %d call, want a split", c, w, n, calls)
				}
			}
		}
	}
}

// TestPolicyInvariant: a call over the serial threshold is worth at least two
// workers, for every cost class (parallelOn relies on it).
func TestPolicyInvariant(t *testing.T) {
	for _, c := range append([]cost{costUser}, costClasses...) {
		if parallelWork/int(c)/max(1, workerWork/int(c)) < 2 {
			t.Errorf("cost %d: threshold %d is under two workers of %d", c, parallelWork/int(c), workerWork/int(c))
		}
	}
}

// TestParallelOnWorkerBound: the number of goroutines never exceeds w, nor
// the work divided by workerWork.
func TestParallelOnWorkerBound(t *testing.T) {
	for _, tc := range []struct {
		w, n int
		c    cost
		max  int
	}{
		// one element short of 9 workers' worth: the cap is 8, while the
		// blocks (at least blockWork each) number 9, so only the cap holds
		// the count at 8
		{16, 9*(workerWork/int(costHeavy)) - 1, costHeavy, 8},
		{2, 1000000, costLight, 2},
		{16, 1000000, costCheap, 16},
	} {
		var mu sync.Mutex
		running, peak := 0, 0
		parallelOn(tc.w, tc.n, tc.c, func(lo, hi int) {
			mu.Lock()
			running++
			peak = max(peak, running)
			mu.Unlock()
			time.Sleep(time.Millisecond) // let the other workers start
			mu.Lock()
			running--
			mu.Unlock()
		})
		if tc.n < parallelWork/int(tc.c) {
			t.Fatalf("w %d n %d cost %d: below the serial threshold %d, the case tests nothing", tc.w, tc.n, tc.c, parallelWork/int(tc.c))
		}
		if peak < 2 {
			t.Errorf("w %d n %d cost %d: %d bodies at once, want a parallel run", tc.w, tc.n, tc.c, peak)
		}
		if peak > tc.max {
			t.Errorf("w %d n %d cost %d: %d bodies at once, bound %d", tc.w, tc.n, tc.c, peak, tc.max)
		}
	}
}

// TestParallelOnUnevenBlocks: a block that takes far longer than the others
// does not hold the rest back (the others claim the remaining blocks) and the
// result is still complete.
func TestParallelOnUnevenBlocks(t *testing.T) {
	n := 64 * blockWork / int(costHeavy) * 4
	var mu sync.Mutex
	done := 0
	parallelOn(4, n, costHeavy, func(lo, hi int) {
		if lo == 0 {
			time.Sleep(20 * time.Millisecond)
		}
		mu.Lock()
		done += hi - lo
		mu.Unlock()
	})
	if done != n {
		t.Errorf("covered %d of %d", done, n)
	}
}

// TestBoundarySizesBitIdentical runs real functions of every cost class at
// sizes around their thresholds with several GOMAXPROCS, and compares each
// element with the scalar function.
func TestBoundarySizesBitIdentical(t *testing.T) {
	old := runtime.GOMAXPROCS(0)
	defer runtime.GOMAXPROCS(old)
	for _, procs := range []int{1, 2, 3, 7} {
		runtime.GOMAXPROCS(procs)
		for _, n := range []int{0, 1, 2, 79, 80, 81, 163, 6665, 6666, 6667, 13332, 13333, 13334, 332, 333, 334, 999, 1000, 1001, 3332, 3333, 3334, 7999, 8000, 8001, 9999, 10000, 10001, 20011} {
			x := make([]float64, n)
			p := make([]float64, n)
			for i := range x {
				x[i] = 0.5 + 9*float64(i)/float64(n+1)
				p[i] = (float64(i) + 0.5) / float64(n+1)
			}
			ax, ap := arr(t, x, n), arr(t, p, n)
			check := func(name string, got []float64, want func(i int) float64) {
				t.Helper()
				for i, g := range got {
					if w := want(i); math.Float64bits(g) != math.Float64bits(w) && !(math.IsNaN(g) && math.IsNaN(w)) {
						t.Fatalf("%s procs %d n %d [%d] = %v, want %v", name, procs, n, i, g, w)
					}
				}
			}
			r, err := GammaSign(ax) // costCheap, apply1
			if err != nil {
				t.Fatal(err)
			}
			check("GammaSign", data(t, r), func(i int) float64 { return special.GammaSign(x[i]) })
			r, err = Erf(ax) // costLight, apply1
			if err != nil {
				t.Fatal(err)
			}
			check("Erf", data(t, r), func(i int) float64 { return special.Erf(x[i]) })
			r, err = BetaInc(ax, ax, ap) // costHeavy, map3
			if err != nil {
				t.Fatal(err)
			}
			check("BetaInc", data(t, r), func(i int) float64 { return special.BetaInc(x[i], x[i], p[i]) })
			r, err = GammaP(ax, ap) // costHeavy, map2
			if err != nil {
				t.Fatal(err)
			}
			check("GammaP", data(t, r), func(i int) float64 { return special.GammaP(x[i], p[i]) })
			r, err = Map1(special.J0, ax) // costUser
			if err != nil {
				t.Fatal(err)
			}
			check("J0", data(t, r), func(i int) float64 { return special.J0(x[i]) })
		}
	}
}
