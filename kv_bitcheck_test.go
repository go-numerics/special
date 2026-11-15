package special

import (
	"encoding/binary"
	"flag"
	"math"
	"math/rand"
	"os"
	"testing"
)

// The bit-identity check of the Kv and Kve fast paths against a baseline build
// (a clean clone of the previous release tag): the same file is run in the
// baseline to dump the outputs of kvBitCount inputs, then here to compare.
//
//	(baseline clone) go test -run TestKvBits -kvbits.dump=/path/kv.bits .
//	(this clone)     go test -run TestKvBits -kvbits.compare=/path/kv.bits -v .
//
// Every input must give the baseline's bits unless it is in the region of
// kv_temme_fast.go (0 < |v| < 32, 2^-1000 <= x < 1.5) and the fast path answers
// there; and every input the path declines, in the region or out of it, is the
// baseline's, bit for bit. The file refers to no symbol of the new path (the
// baseline has none), so the region is spelled with its constants.

var (
	kvBitsDump    = flag.String("kvbits.dump", "", "TestKvBits: write Kv and Kve of every input to this file")
	kvBitsCompare = flag.String("kvbits.compare", "", "TestKvBits: compare with a dump of the baseline")
)

const (
	kvBitCount = 240000
	kvBitMinX  = 0x1p-1000
	kvBitMaxX  = 1.5
	kvBitMaxV  = 32.0
)

// kvBitInput returns the i-th input: orders in seven classes (uniform below 40,
// integers, half-integers, tiny, near an integer, the Debye range and beyond,
// negative), x log-uniform from 1e-310 to 1e7 with shares at the edges of the
// regions.
func kvBitInput(r *rand.Rand, i int) (v, x float64) {
	switch i % 7 {
	case 0:
		v = float64(r.Float64() * 40)
	case 1:
		v = float64(r.Intn(41))
	case 2:
		v = float64(r.Intn(40)) + 0.5
	case 3:
		v = math.Exp(float64(r.Float64() * (math.Log(1e-3) - math.Log(1e-300))))
		v = float64(v * 1e-300)
	case 4:
		v = float64(r.Intn(40)) + float64((r.Float64()-0.5)*2e-6)
	case 5:
		v = 20 + float64(r.Float64()*1500)
	case 6:
		v = -float64(r.Float64() * 40)
	}
	x = math.Exp(math.Log(1e-310) + float64(r.Float64()*(math.Log(1e7)-math.Log(1e-310))))
	switch i % 11 {
	case 0:
		x = math.Exp(float64(r.Float64()*math.Log(1.5/1e-300)) + math.Log(1e-300)) // the Temme region
	case 1:
		x = float64(r.Float64() * 1.6)
	case 2:
		x = 1.5 + float64((r.Float64()-0.5)*1e-9)
	case 3:
		x = 24 + float64((r.Float64()-0.5)*1e-6)
	}
	return v, x
}

func TestKvBits(t *testing.T) {
	if *kvBitsDump == "" && *kvBitsCompare == "" {
		t.Skip("needs -kvbits.dump or -kvbits.compare")
	}
	r := rand.New(rand.NewSource(31))
	vs, xs := make([]float64, kvBitCount), make([]float64, kvBitCount)
	for i := range vs {
		vs[i], xs[i] = kvBitInput(r, i)
	}
	got := make([]float64, 2*kvBitCount)
	kvParallel(kvBitCount, func(i int) {
		got[2*i], got[2*i+1] = Kv(vs[i], xs[i]), Kve(vs[i], xs[i])
	})
	if *kvBitsDump != "" {
		var b []byte
		for _, g := range got {
			b = binary.LittleEndian.AppendUint64(b, math.Float64bits(g))
		}
		if err := os.WriteFile(*kvBitsDump, b, 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	b, err := os.ReadFile(*kvBitsCompare)
	if err != nil {
		t.Fatal(err)
	}
	if len(b) != 16*kvBitCount {
		t.Fatalf("dump has %d bytes, want %d", len(b), 16*kvBitCount)
	}
	var same, answered, declinedInRegion, outside, judged int
	var worst, worstBase float64
	for i := range vs {
		v, x := math.Abs(vs[i]), xs[i]
		for k, scaled := range []bool{false, true} {
			want := math.Float64frombits(binary.LittleEndian.Uint64(b[8*(2*i+k):]))
			g := got[2*i+k]
			region := v > 0 && v < kvBitMaxV && x >= kvBitMinX && x < kvBitMaxX
			if sameBits(g, want) {
				same++
				if region {
					if _, ok := kvFast(v, x, scaled); !ok {
						declinedInRegion++
					}
				} else {
					outside++
				}
				continue
			}
			if !region {
				t.Errorf("Kv/Kve(%g, %g) scaled=%v: %v, baseline %v (outside the region)", vs[i], x, scaled, g, want)
				continue
			}
			if _, ok := kvFast(v, x, scaled); !ok {
				t.Errorf("Kv/Kve(%g, %g) scaled=%v: %v, baseline %v, but the fast path declines", vs[i], x, scaled, g, want)
				continue
			}
			answered++
			d := math.Abs(g-want) / ulp(want)
			if d > worst {
				worst = d
			}
			// where the new answer is more than 2 ulp from the baseline's, the
			// oracle decides which of the two is wrong (an order whose u is not
			// below the oracle's 2^-100, not an integer)
			if d > 2 && v != math.Trunc(v) && math.Abs(v-math.Round(v)) >= 1e-25 {
				ke := bigKve(v, x)
				if !scaled {
					ke = bnew().Mul(ke, bigExp(bf(-x)))
				}
				if e := ivOracleUlps(g, ke); e > kvFastBound {
					t.Errorf("Kv/Kve(%g, %g) scaled=%v: %v is %.3f ulp from the oracle (baseline %v, %.3f ulp)", vs[i], x, scaled, g, e, want, ivOracleUlps(want, ke))
				}
				judged++
				if e := ivOracleUlps(want, ke); e > worstBase {
					worstBase = e
				}
			}
		}
	}
	t.Logf("%d inputs, Kv and Kve (%d values): %d identical outside the region, %d identical inside (%d of them where the path declines, the rest answered with the baseline's bits), %d answered with other bits (up to %.2f ulp from the baseline); %d of those more than 2 ulp away judged by the oracle: the baseline is up to %.2f ulp off there, the new value within %.2f",
		kvBitCount, 2*kvBitCount, outside, same-outside, declinedInRegion, answered, worst, judged, worstBase, kvFastBound)
}
