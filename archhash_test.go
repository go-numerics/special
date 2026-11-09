package special

import (
	"bufio"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"flag"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
	"testing"
)

// The cross-architecture golden hash: every public function evaluated on a
// fixed grid, the output bits hashed per function, and the hashes compared
// with testdata/archhash.golden. The CI lanes run it on amd64, 386, arm64,
// arm, ppc64le, s390x, riscv64 and loong64; a product left open to fusion
// (see tools/fmalint) gives different bits on the architectures where Go
// fuses multiply-adds, and this test names the function.
//
//	go test -run TestArchHash -archhash.update .   # rewrite the golden file

var archHashUpdate = flag.Bool("archhash.update", false, "TestArchHash: rewrite testdata/archhash.golden")

const archHashGolden = "testdata/archhash.golden"

// grid returns n points spread over [lo, hi] by the golden-ratio sequence
// (no point lands on a simple fraction by construction).
func grid(n int, lo, hi float64) []float64 {
	x := make([]float64, n)
	for i := range x {
		f := float64(i) * 0.6180339887498949
		f -= math.Floor(f)
		x[i] = lo + float64((hi-lo)*f)
	}
	return x
}

// spread returns grid points over several ranges at once.
func spread(n int, r ...[2]float64) []float64 {
	var x []float64
	for _, v := range r {
		x = append(x, grid(n, v[0], v[1])...)
	}
	return x
}

type hashCase struct {
	name string
	eval func(emit func(float64))
}

func f1(name string, f func(float64) float64, xs []float64) hashCase {
	return hashCase{name, func(emit func(float64)) {
		for _, x := range xs {
			emit(f(x))
		}
	}}
}

func f2(name string, f func(a, b float64) float64, as, bs []float64) hashCase {
	return hashCase{name, func(emit func(float64)) {
		for _, a := range as {
			for _, b := range bs {
				emit(f(a, b))
			}
		}
	}}
}

func fn(name string, f func(n int, x float64) float64, ns []int, xs []float64) hashCase {
	return hashCase{name, func(emit func(float64)) {
		for _, n := range ns {
			for _, x := range xs {
				emit(f(n, x))
			}
		}
	}}
}

func archHashCases() []hashCase {
	r := func(lo, hi float64) [2]float64 { return [2]float64{lo, hi} }
	wide := spread(400, r(-180, -0.01), r(0.01, 1), r(1, 10), r(10, 180))
	pos := spread(400, r(1e-6, 1), r(1, 30), r(30, 700))
	signed := spread(400, r(-40, -1), r(-1, 1), r(1, 40))
	unit := spread(400, r(1e-12, 1e-3), r(1e-3, 0.999), r(0.999, 1-1e-12))
	m := spread(300, r(-5, 0), r(0, 0.999999))
	par := spread(12, r(0.05, 1), r(1, 20), r(20, 300))
	px := spread(12, r(0.01, 1), r(1, 30), r(30, 400))
	pu := spread(12, r(1e-9, 0.01), r(0.01, 0.99))
	orders := []int{0, 1, 2, 5, 13, 40}
	vs := spread(6, r(-7.5, -0.1), r(0, 3), r(3, 60))
	bx := spread(40, r(0.01, 2), r(2, 30), r(30, 500))
	three := func(name string, f func(a, b, x float64) float64, xs []float64) hashCase {
		ab := spread(6, r(0.1, 1), r(1, 15), r(15, 200))
		return hashCase{name, func(emit func(float64)) {
			for _, a := range ab {
				for _, b := range ab {
					for _, x := range xs {
						emit(f(a, b, x))
					}
				}
			}
		}}
	}
	return []hashCase{
		f1("Gamma", Gamma, wide),
		f1("LogGamma", LogGamma, wide),
		f1("GammaSign", GammaSign, wide),
		f1("Digamma", Digamma, wide),
		fn("Polygamma", Polygamma, []int{0, 1, 2, 3, 6}, spread(100, r(-30, -0.1), r(0.01, 100))),
		f2("Beta", Beta, par, par),
		f2("LogBeta", LogBeta, par, par),
		f1("Erf", Erf, signed),
		f1("Erfc", Erfc, spread(400, r(-6, 0), r(0, 27))),
		f1("Erfcx", Erfcx, spread(400, r(-26, 0), r(0, 1e5))),
		f1("ErfInv", ErfInv, spread(400, r(-1, 1))),
		f1("ErfcInv", ErfcInv, spread(400, r(1e-300, 1e-10), r(1e-10, 2))),
		f1("Ndtr", Ndtr, spread(400, r(-38, 0), r(0, 9))),
		f1("Ndtri", Ndtri, unit),
		f1("LogNdtr", LogNdtr, spread(400, r(-1e5, -10), r(-10, 10), r(10, 40))),
		f1("Expit", Expit, signed),
		f1("LogExpit", LogExpit, signed),
		f1("Logit", Logit, unit),
		f1("Log1pExp", Log1pExp, spread(400, r(-745, -1), r(-1, 40))),
		f2("Xlogy", Xlogy, signed, pos[:200]),
		f2("Xlog1py", Xlog1py, signed[:200], spread(60, r(-0.999, 1), r(1, 1e4))),
		f1("Zeta", Zeta, spread(400, r(-60, 0.99), r(1.01, 80))),
		f2("HurwitzZeta", HurwitzZeta, spread(10, r(1.1, 3), r(3, 50)), px),
		f2("GammaP", GammaP, par, px),
		f2("GammaQ", GammaQ, par, px),
		f2("GammaPInv", GammaPInv, par, pu),
		f2("GammaQInv", GammaQInv, par, pu),
		three("BetaInc", BetaInc, spread(8, r(0.001, 0.5), r(0.5, 0.999))),
		three("BetaIncC", BetaIncC, spread(8, r(0.001, 0.5), r(0.5, 0.999))),
		three("BetaIncInv", BetaIncInv, spread(4, r(1e-6, 0.5), r(0.5, 0.999))),
		three("BetaIncCInv", BetaIncCInv, spread(4, r(1e-6, 0.5), r(0.5, 0.999))),
		f1("J0", J0, bx), f1("J1", J1, bx), f1("Y0", Y0, bx), f1("Y1", Y1, bx),
		f1("I0", I0, bx), f1("I1", I1, bx), f1("I0e", I0e, bx), f1("I1e", I1e, bx),
		f1("K0", K0, bx), f1("K1", K1, bx), f1("K0e", K0e, bx), f1("K1e", K1e, bx),
		fn("Jn", Jn, orders, bx), fn("Yn", Yn, orders, bx), fn("In", In, orders, bx), fn("Kn", Kn, orders, bx),
		f2("Jv", Jv, vs, bx[:60]), f2("Yv", Yv, vs, bx[:60]), f2("Iv", Iv, vs, bx[:60]),
		f2("Ive", Ive, vs, bx[:60]), f2("Kv", Kv, vs, bx[:60]), f2("Kve", Kve, vs, bx[:60]),
		f1("EllipK", EllipK, m), f1("EllipE", EllipE, m), f1("EllipKm1", EllipKm1, spread(300, r(1e-300, 1e-5), r(1e-5, 1))),
		f2("Binom", Binom, spread(10, r(-5, 0), r(0, 80)), spread(10, r(-3, 0), r(0, 40))),
		f1("Factorial", Factorial, spread(200, r(0, 171))),
		f1("LogFactorial", LogFactorial, spread(200, r(0, 1e6))),
		{"Comb,Perm,Factorial2", func(emit func(float64)) {
			for n := int64(0); n < 70; n += 3 {
				for k := int64(0); k <= n; k += 2 {
					emit(Comb(n, k))
					emit(Perm(n, k))
				}
				emit(Factorial2(int(n)))
			}
		}},
		{"LogSumExp,Softmax,LogSoftmax", func(emit func(float64)) {
			x := spread(300, r(-50, 50))
			emit(LogSumExp(x))
			for _, v := range Softmax(nil, x) {
				emit(v)
			}
			for _, v := range LogSoftmax(nil, x) {
				emit(v)
			}
		}},
		{"JnZeros", func(emit func(float64)) {
			for _, n := range []int{0, 1, 4} {
				for _, v := range JnZeros(n, 30) {
					emit(v)
				}
			}
		}},
	}
}

func archHashes() map[string]string {
	out := map[string]string{}
	var buf [8]byte
	for _, c := range archHashCases() {
		h := sha256.New()
		c.eval(func(v float64) {
			if v != v {
				v = math.NaN() // one NaN pattern: payloads differ by architecture
			}
			binary.LittleEndian.PutUint64(buf[:], math.Float64bits(v))
			h.Write(buf[:])
		})
		out[c.name] = hex.EncodeToString(h.Sum(nil))
	}
	return out
}

var (
	archHashDump    = flag.String("archhash.dump", "", "TestArchHashDiff: write every output of the hash grid to this file")
	archHashCompare = flag.String("archhash.compare", "", "TestArchHashDiff: report, per function, how many outputs differ from this dump and by how many ulps at most")
)

// TestArchHashDiff measures what a change of bits amounts to: dump the grid's
// outputs from one build, compare from another.
func TestArchHashDiff(t *testing.T) {
	if *archHashDump == "" && *archHashCompare == "" {
		t.Skip("needs -archhash.dump or -archhash.compare")
	}
	type rec struct {
		name string
		v    []float64
	}
	var all []rec
	for _, c := range archHashCases() {
		r := rec{name: c.name}
		c.eval(func(v float64) { r.v = append(r.v, v) })
		all = append(all, r)
	}
	if *archHashDump != "" {
		var b []byte
		for _, r := range all {
			b = binary.LittleEndian.AppendUint32(b, uint32(len(r.v)))
			for _, v := range r.v {
				b = binary.LittleEndian.AppendUint64(b, math.Float64bits(v))
			}
		}
		if err := os.WriteFile(*archHashDump, b, 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	b, err := os.ReadFile(*archHashCompare)
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range all {
		n := int(binary.LittleEndian.Uint32(b))
		b = b[4:]
		if n != len(r.v) {
			t.Fatalf("%s: dump has %d values, grid %d", r.name, n, len(r.v))
		}
		diff, maxULP := 0, 0.0
		for _, v := range r.v {
			w := math.Float64frombits(binary.LittleEndian.Uint64(b))
			b = b[8:]
			if sameBits(v, w) {
				continue
			}
			diff++
			if u := math.Abs(v-w) / ulp(w); u > maxULP || math.IsNaN(u) {
				maxULP = u
			}
		}
		if diff > 0 {
			t.Logf("%-28s %6d of %6d outputs differ, at most %.3g ulp", r.name, diff, len(r.v), maxULP)
		}
	}
}

func TestArchHash(t *testing.T) {
	got := archHashes()
	if *archHashUpdate {
		names := make([]string, 0, len(got))
		for n := range got {
			names = append(names, n)
		}
		sort.Strings(names)
		var b strings.Builder
		b.WriteString("# function sha256(output bits); go test -run TestArchHash -archhash.update .\n")
		for _, n := range names {
			fmt.Fprintf(&b, "%s %s\n", n, got[n])
		}
		if err := os.WriteFile(archHashGolden, []byte(b.String()), 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	f, err := os.Open(archHashGolden)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	want := map[string]string{}
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := sc.Text()
		if strings.HasPrefix(line, "#") || line == "" {
			continue
		}
		name, sum, ok := strings.Cut(line, " ")
		if !ok {
			t.Fatalf("%s: malformed line %q", archHashGolden, line)
		}
		want[name] = sum
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	var bad []string
	for n, sum := range got {
		w, ok := want[n]
		if !ok {
			t.Errorf("%s: no golden hash (run with -archhash.update)", n)
			continue
		}
		if w != sum {
			bad = append(bad, n)
		}
	}
	sort.Strings(bad)
	if len(bad) > 0 {
		t.Errorf("output bits differ from %s on this architecture for %d function(s): %s",
			archHashGolden, len(bad), strings.Join(bad, ", "))
	}
	if len(want) != len(got) {
		t.Errorf("%s has %d entries, the test computes %d", archHashGolden, len(want), len(got))
	}
}
