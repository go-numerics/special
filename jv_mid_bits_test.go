package special

import (
	"encoding/binary"
	"math"
	"math/rand"
	"os"
	"testing"
)

// The bit-identity check of Jv and Yv against a baseline build (a clean clone of
// the previous release, special main at b4e5a09 = v0.18.0's content): the same
// file is run there to dump the outputs of jvMidBitCount inputs, then here to
// compare.
//
//	(baseline clone) JVMID_BITS_DUMP=/path/jv.bits go test -run TestJvMidBits .
//	(this clone)     JVMID_BITS_COMPARE=/path/jv.bits go test -run TestJvMidBits -v .
//
// Every input outside the range of jv_mid_fast.go (1 <= x <= 20, non-integer |v|
// <= 40.5), and every input in it that the path declines, must give the
// baseline's bits; where it answers, the answer must be within the oracle's
// jvMidBitsBound (checked on the inputs where the two differ by more than 2
// ulp, for the baseline's error is up to 40000 ulp near the zeros). The file
// refers to no symbol of the new path (the baseline has none): the range is
// spelled with its constants, and jvMidProbe, set by jv_mid_fast_test.go, tells
// whether the path answers.

const (
	jvMidBitCount  = 120000
	jvMidBitsBound = 0.5625
)

var jvMidProbe func(v, x float64, wantY bool) bool

func TestJvMidBits(t *testing.T) {
	dump, cmp := os.Getenv("JVMID_BITS_DUMP"), os.Getenv("JVMID_BITS_COMPARE")
	if dump == "" && cmp == "" {
		t.Skip("needs JVMID_BITS_DUMP or JVMID_BITS_COMPARE")
	}
	r := rand.New(rand.NewSource(63))
	vs, xs := make([]float64, jvMidBitCount), make([]float64, jvMidBitCount)
	for i := range vs {
		vs[i], xs[i] = jvMidBitInput(r, i)
	}
	got := make([]float64, 2*jvMidBitCount)
	kvParallel(jvMidBitCount, func(i int) {
		got[2*i], got[2*i+1] = Jv(vs[i], xs[i]), Yv(vs[i], xs[i])
	})
	if dump != "" {
		var b []byte
		for _, g := range got {
			b = binary.LittleEndian.AppendUint64(b, math.Float64bits(g))
		}
		if err := os.WriteFile(dump, b, 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	b, err := os.ReadFile(cmp)
	if err != nil {
		t.Fatal(err)
	}
	if len(b) != 16*jvMidBitCount {
		t.Fatalf("dump has %d bytes, want %d", len(b), 16*jvMidBitCount)
	}
	var identical, outside, declinedIn, answered, judged int
	var worst, worstBase, worstNew float64
	for i := range vs {
		for k, y := range []bool{false, true} {
			want := math.Float64frombits(binary.LittleEndian.Uint64(b[8*(2*i+k):]))
			g := got[2*i+k]
			inRange := xs[i] >= 1 && xs[i] <= 20 && math.Abs(vs[i]) <= 40.5 && vs[i] != math.Trunc(vs[i])
			ok := inRange && jvMidProbe(vs[i], xs[i], y)
			switch {
			case sameBits(g, want):
				identical++
				if !inRange {
					outside++
				} else if !ok {
					declinedIn++
				}
			case !ok:
				t.Errorf("Jv/Yv(%g, %g) Y=%v: %v, baseline %v, but the fast path declines or is not in range", vs[i], xs[i], y, g, want)
			default:
				answered++
				d := math.Abs(g-want) / ulp(want)
				worst = math.Max(worst, d)
				if d > 2 {
					o := bigJv(vs[i], xs[i])
					if y {
						o = bigYv(vs[i], xs[i])
					}
					judged++
					e := jvUlps(g, o)
					worstNew = math.Max(worstNew, e)
					worstBase = math.Max(worstBase, jvUlps(want, o))
					if e > jvMidBitsBound {
						t.Errorf("Jv/Yv(%g, %g) Y=%v: %v is %.4f ulp from the oracle (baseline %v, %.2f ulp)", vs[i], xs[i], y, g, e, want, jvUlps(want, o))
					}
				}
			}
		}
	}
	t.Logf("%d inputs, J and Y (%d values): %d identical (%d outside the range, %d declined inside), %d answered with other bits (up to %.2f ulp from the baseline); %d of those more than 2 ulp away judged by the oracle: the baseline is up to %.2f ulp off there, the new value within %.4f",
		jvMidBitCount, 2*jvMidBitCount, identical, outside, declinedIn, answered, worst, judged, worstBase, worstNew)
}

// jvMidBitInput returns the i-th input: orders in seven classes (uniform,
// integers, half-integers, tiny, near an integer, 38 to 44, small), x in [0.5,
// 25] with shares at both edges.
func jvMidBitInput(r *rand.Rand, i int) (v, x float64) {
	switch i % 7 {
	case 0:
		v = float64(float64(r.Float64()*82) - 41)
	case 1:
		v = float64(r.Intn(83) - 41)
	case 2:
		v = float64(r.Intn(82)-41) + 0.5
	case 3:
		v = float64(math.Exp(float64(r.Float64()*(math.Log(1e-3)-math.Log(1e-12)))) * 1e-12)
		if r.Intn(2) == 0 {
			v = -v
		}
	case 4:
		v = float64(r.Intn(41)) + float64((r.Float64()-0.5)*2e-6)
	case 5:
		v = 38 + float64(r.Float64()*6)
		if r.Intn(2) == 0 {
			v = -v
		}
	case 6:
		v = float64(float64(r.Float64()*20) - 10)
	}
	x = 0.5 + float64(r.Float64()*24.5)
	switch i % 11 {
	case 0:
		x = 1 + float64((r.Float64()-0.5)*1e-6)
	case 1:
		x = 20 + float64((r.Float64()-0.5)*1e-6)
	}
	return v, x
}
