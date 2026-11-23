package nd

import (
	"encoding/json"
	"math"
	"os"
	"regexp"
	"runtime"
	"slices"
	"sort"
	"sync/atomic"
	"testing"
	"time"
)

// measuredNs is the median serial time of one element of each function of
// docs/perf.json that is timed over 10000 points, by function name.
func measuredNs(t *testing.T) map[string]float64 {
	t.Helper()
	b, err := os.ReadFile("../docs/perf.json")
	if err != nil {
		t.Fatal(err)
	}
	var doc struct {
		Rows []struct {
			Name   string `json:"name"`
			Func   string `json:"func"`
			Ref    string `json:"reference"`
			Rounds []struct {
				GoNs float64 `json:"go_ns"`
			} `json:"rounds"`
		} `json:"rows"`
	}
	if err := json.Unmarshal(b, &doc); err != nil {
		t.Fatal(err)
	}
	out := map[string]float64{}
	for _, r := range doc.Rows {
		if r.Name != r.Func || len(r.Rounds) == 0 || !regexp.MustCompile(`over 10000 points`).MatchString(r.Ref) {
			continue
		}
		v := make([]float64, len(r.Rounds))
		for i, x := range r.Rounds {
			v[i] = x.GoNs
		}
		sort.Float64s(v)
		out[r.Func] = v[len(v)/2] / 10000
	}
	return out
}

// nearest is the index in costClasses of the class closest to ns on a
// logarithmic scale.
func nearest(ns float64) int {
	best, bd := 0, math.Inf(1)
	for i, c := range costClasses {
		if d := math.Abs(math.Log(ns / float64(c))); d < bd {
			best, bd = i, d
		}
	}
	return best
}

// TestCostsMatchMeasurement: the class of every function of the costs table is
// at most one class away from its measured cost in docs/perf.json, so the
// classes cannot drift from the measurements, and every function is measured
// (apart from the ones listed).
func TestCostsMatchMeasurement(t *testing.T) {
	ns := measuredNs(t)
	unmeasured := map[string]bool{"LogFactorial": true}
	for name, c := range costs {
		i := slices.Index(costClasses, c)
		if i < 0 {
			t.Errorf("%s: %d is not a class", name, c)
			continue
		}
		m, ok := ns[name]
		if !ok {
			if !unmeasured[name] {
				t.Errorf("%s has no measured row in docs/perf.json", name)
			}
			continue
		}
		if d := i - nearest(m); d > 1 || d < -1 {
			t.Errorf("%s: class %d ns, measured %.1f ns (class %d ns): more than one class apart",
				name, c, m, costClasses[nearest(m)])
		}
	}
}

// TestCostsKeysMatchWrappers: every costs["X"] of the wrappers names a
// function of the table, and every function of the table is used.
func TestCostsKeysMatchWrappers(t *testing.T) {
	src, err := os.ReadFile("wrappers.go")
	if err != nil {
		t.Fatal(err)
	}
	used := map[string]bool{}
	for _, m := range regexp.MustCompile(`costs\["(\w+)"\]`).FindAllSubmatch(src, -1) {
		used[string(m[1])] = true
		if _, ok := costs[string(m[1])]; !ok {
			t.Errorf("wrappers.go uses costs[%q], which is not in the table", m[1])
		}
	}
	for name := range costs {
		if !used[name] {
			t.Errorf("costs[%q] is not used by wrappers.go", name)
		}
	}
}

// TestThresholds pins the thresholds that a wrong class used to get wrong
// (Binom ran serially up to 9999 elements, the scaled Bessel functions were
// parallel from 667).
func TestThresholds(t *testing.T) {
	for _, tc := range []struct {
		name string
		want int
	}{{"Binom", 333}, {"GammaSign", 13333}, {"EllipK", 6666}, {"J0", 3333}, {"HurwitzZeta", 333}, {"BetaInc", 80}} {
		if got := parallelWork / int(costs[tc.name]); got != tc.want {
			t.Errorf("%s: parallel from %d elements, want %d", tc.name, got, tc.want)
		}
	}
	if got := parallelWork / int(costUser); got < 8000 || got > 8192 {
		t.Errorf("user function: parallel from %d elements, want main's 8192", got)
	}
}

// TestUserFunctionUsesEveryProcessor: from the threshold on, a user function
// gets all w workers (its cost is unknown, main used GOMAXPROCS), not the
// n/4000 of its nominal class. Rendezvous of w bodies proves it.
func TestUserFunctionUsesEveryProcessor(t *testing.T) {
	const w = 8
	n := 8192 // 2 workers under the nominal class, w on main
	var arrived atomic.Int32
	met := make(chan struct{})
	var failed atomic.Bool
	parallelOn(w, n, costUser, func(lo, hi int) {
		if arrived.Add(1) == w {
			close(met)
		}
		select {
		case <-met:
		case <-time.After(2 * time.Second):
			failed.Store(true)
		}
	})
	if failed.Load() {
		t.Errorf("fewer than %d bodies ran at once at n = %d", w, n)
	}
}

// rendezvous makes a body wait until a second one is running at the same
// time: the proof that two goroutines really execute bodies concurrently.
type rendezvous struct {
	arrived atomic.Int32
	met     chan struct{}
}

func newRendezvous() *rendezvous { return &rendezvous{met: make(chan struct{})} }

// wait reports whether another caller arrived before the timeout.
func (r *rendezvous) wait() bool {
	if r.arrived.Add(1) == 2 {
		close(r.met)
	}
	select {
	case <-r.met:
		return true
	case <-time.After(2 * time.Second):
		return false
	}
}

// TestParallelOnRunsConcurrently: above the threshold at least two bodies run
// at the same time (a loop that spawned no worker would time out here).
func TestParallelOnRunsConcurrently(t *testing.T) {
	r := newRendezvous()
	var failed atomic.Bool
	parallelOn(2, 10*parallelWork/int(costHeavy), costHeavy, func(lo, hi int) {
		if !r.wait() {
			failed.Store(true)
		}
	})
	if failed.Load() {
		t.Error("no second body ran while the first was waiting")
	}
}

// TestMap1RunsConcurrently is the same through a public function.
func TestMap1RunsConcurrently(t *testing.T) {
	old := runtime.GOMAXPROCS(4)
	defer runtime.GOMAXPROCS(old)
	r := newRendezvous()
	var failed atomic.Bool
	var calls atomic.Int32
	x := make([]float64, 100000)
	if _, err := Map1(func(v float64) float64 {
		if calls.Add(1) <= 2 && !r.wait() {
			failed.Store(true)
		}
		return v
	}, arr(t, x, len(x))); err != nil {
		t.Fatal(err)
	}
	if failed.Load() {
		t.Error("no second call of f ran while the first was waiting")
	}
}

// recovered runs f and returns what it panicked with, nil if it did not.
func recovered(f func()) (v any) {
	defer func() { v = recover() }()
	f()
	return nil
}

// TestMap1PanicReachesCaller: a panic of the user function is raised in the
// calling goroutine at every size, as on a serial loop, with GOMAXPROCS 4.
func TestMap1PanicReachesCaller(t *testing.T) {
	old := runtime.GOMAXPROCS(4)
	defer runtime.GOMAXPROCS(old)
	for _, n := range []int{1, 1000, 8192, 100000} {
		x := make([]float64, n)
		v := recovered(func() {
			_, _ = Map1(func(v float64) float64 { panic("boom") }, arr(t, x, n))
		})
		if v != "boom" {
			t.Errorf("n %d: recovered %v, want boom", n, v)
		}
		// only one element panics, in whatever worker meets it
		v = recovered(func() {
			for k := range x {
				x[k] = float64(k)
			}
			_, _ = Map1(func(v float64) float64 {
				if int(v) == n-1 {
					panic(n)
				}
				return v
			}, arr(t, x, n))
		})
		if v != n {
			t.Errorf("n %d: recovered %v, want %d", n, v, n)
		}
	}
}

// TestParallelOnPanics: when several workers panic, the first value reaches
// the caller and the others are dropped; every worker has returned by then.
func TestParallelOnPanics(t *testing.T) {
	r := newRendezvous()
	var running atomic.Int32
	n := 10 * parallelWork / int(costHeavy)
	v := recovered(func() {
		parallelOn(2, n, costHeavy, func(lo, hi int) {
			running.Add(1)
			defer running.Add(-1)
			r.wait()
			panic("both")
		})
	})
	if v != "both" {
		t.Errorf("recovered %v, want both", v)
	}
	if k := running.Load(); k != 0 {
		t.Errorf("%d bodies still running after the panic reached the caller", k)
	}
}
