package cmplx

import (
	"bufio"
	"compress/gzip"
	"flag"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"testing"
)

// The accuracy harness (docs/design-complex.md, "Validation"). Fixtures are
// testdata/c_<name>.tsv.gz, written by tools/gencomplex from mpmath: per
// row, a region, the float64 arguments, and each part of the exact value
// as a double-double. The error of each part is measured in ulps of that
// part's exact value; the maximum per region is gated by a budget.

var tableOut = flag.String("cmplx.table", "", "replace the per-region ULP table (markdown) in this page, after its marker line")

type crow struct {
	region string
	args   []float64
	re, im [2]float64 // hi, lo
}

// accCase is one fixture: the function under test (nil before it exists),
// the same operation in math/cmplx or the builtin operators (the comparison
// column), and the maximum error in ulps per part allowed per region ("*"
// is the default).
type accCase struct {
	fixture string
	fn      func(a []float64) complex128
	std     func(a []float64) complex128
	budget  map[string]float64
}

var accCases []accCase

func register(c accCase) { accCases = append(accCases, c) }

type impl struct {
	fn     func(a []float64) complex128
	budget map[string]float64
}

// impls holds the functions under test by fixture name (filled by the init
// functions of the per-family test files, in any order).
var impls = map[string]impl{}

func implement(fixture string, fn func(a []float64) complex128, budget map[string]float64) {
	impls[fixture] = impl{fn, budget}
}

func loadC(tb testing.TB, name string) []crow {
	tb.Helper()
	f, err := os.Open("testdata/c_" + name + ".tsv.gz")
	if err != nil {
		tb.Fatal(err)
	}
	defer f.Close()
	gz, err := gzip.NewReader(f)
	if err != nil {
		tb.Fatal(err)
	}
	var rows []crow
	sc := bufio.NewScanner(gz)
	for sc.Scan() {
		line := sc.Text()
		if line == "" || line[0] == '#' {
			continue
		}
		cols := strings.Split(line, "\t")
		if len(cols) < 7 {
			tb.Fatalf("c_%s: short row %q", name, line)
		}
		nums := make([]float64, len(cols)-1)
		for i, c := range cols[1:] {
			v, err := strconv.ParseFloat(c, 64)
			if err != nil {
				tb.Fatalf("c_%s: %q: %v", name, line, err)
			}
			nums[i] = v
		}
		n := len(nums)
		rows = append(rows, crow{cols[0], nums[:n-4], [2]float64{nums[n-4], nums[n-3]}, [2]float64{nums[n-2], nums[n-1]}})
	}
	if err := sc.Err(); err != nil {
		tb.Fatal(err)
	}
	if len(rows) == 0 {
		tb.Fatalf("c_%s: no rows", name)
	}
	return rows
}

// ulp is the spacing of float64 at |x|; the smallest subnormal below the
// normal range.
func ulp(x float64) float64 {
	x = math.Abs(x)
	if x < 0x1p-1022 {
		return 0x1p-1074
	}
	if math.IsInf(x, 0) {
		return math.Inf(1)
	}
	_, e := math.Frexp(x)
	return math.Ldexp(1, e-53)
}

// ulpErr measures |got - (hi+lo)| in ulps of hi. A NaN or infinite expected
// value must be matched exactly; anything else is +Inf.
func ulpErr(got, hi, lo float64) float64 {
	switch {
	case math.IsNaN(hi):
		if math.IsNaN(got) {
			return 0
		}
		return math.Inf(1)
	case math.IsNaN(got):
		return math.Inf(1)
	case math.IsInf(hi, 0):
		if got == hi {
			return 0
		}
		return math.Inf(1)
	case math.IsInf(got, 0):
		return math.Inf(1)
	}
	return math.Abs((got-hi)-lo) / ulp(hi)
}

// stat is the worst error of each part in one region, with its argument.
type stat struct {
	n          int
	re, im     float64
	reAt, imAt []float64
}

func measure(rows []crow, f func([]float64) complex128) (map[string]*stat, []string) {
	out := map[string]*stat{}
	var order []string
	for _, r := range rows {
		s := out[r.region]
		if s == nil {
			s = &stat{}
			out[r.region] = s
			order = append(order, r.region)
		}
		s.n++
		got := f(r.args)
		if e := ulpErr(real(got), r.re[0], r.re[1]); e > s.re || s.reAt == nil {
			s.re, s.reAt = e, r.args
		}
		if e := ulpErr(imag(got), r.im[0], r.im[1]); e > s.im || s.imAt == nil {
			s.im, s.imAt = e, r.args
		}
	}
	return out, order
}

func fmtULP(x float64) string {
	switch {
	case math.IsInf(x, 0):
		return "wrong"
	case x >= 1e6:
		return fmt.Sprintf("%.1e", x)
	}
	return fmt.Sprintf("%.2f", x)
}

// TestAccuracy gates every registered function and prints, next to it, the
// error of the math/cmplx (or builtin) counterpart on the same points.
func TestAccuracy(t *testing.T) {
	var tbl strings.Builder
	tbl.WriteString("| function | region | points | Re (ulp) | Im (ulp) | math/cmplx Re | math/cmplx Im |\n|---|---|---:|---:|---:|---:|---:|\n")
	for _, c := range accCases {
		rows := loadC(t, c.fixture)
		std, order := measure(rows, c.std)
		var ours map[string]*stat
		if im, ok := impls[c.fixture]; ok {
			c.fn, c.budget = im.fn, im.budget
		}
		if c.fn != nil {
			ours, _ = measure(rows, c.fn)
		}
		for _, reg := range order {
			sre, sim := "-", "-"
			if ours != nil {
				s := ours[reg]
				sre, sim = fmtULP(s.re), fmtULP(s.im)
				b, ok := c.budget[reg]
				if !ok {
					b = c.budget["*"]
				}
				if s.re > b {
					t.Errorf("%s/%s: Re error %s ulp > budget %g at %v", c.fixture, reg, fmtULP(s.re), b, s.reAt)
				}
				if s.im > b {
					t.Errorf("%s/%s: Im error %s ulp > budget %g at %v", c.fixture, reg, fmtULP(s.im), b, s.imAt)
				}
			}
			fmt.Fprintf(&tbl, "| %s | %s | %d | %s | %s | %s | %s |\n", c.fixture, reg, std[reg].n, sre, sim, fmtULP(std[reg].re), fmtULP(std[reg].im))
		}
	}
	t.Log("\n" + tbl.String())
	if *tableOut != "" {
		// The table replaces everything after the marker line of the page.
		old, err := os.ReadFile(*tableOut)
		if err != nil {
			t.Fatal(err)
		}
		const marker = "<!-- table: go test -run TestAccuracy ./cmplx -args -cmplx.table=../docs/accuracy-complex.md -->\n"
		head, _, ok := strings.Cut(string(old), marker)
		if !ok {
			t.Fatalf("%s: no marker line %q", *tableOut, marker)
		}
		if err := os.WriteFile(*tableOut, []byte(head+marker+"\n"+tbl.String()), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

// same reports whether a and b are the same float64, NaNs alike whatever
// their payload, zeros by sign.
func same(a, b float64) bool {
	if math.IsNaN(a) || math.IsNaN(b) {
		return math.IsNaN(a) && math.IsNaN(b)
	}
	return math.Float64bits(a) == math.Float64bits(b)
}

func sameC(a, b complex128) bool { return same(real(a), real(b)) && same(imag(a), imag(b)) }

// Parity of a function under z -> -z (G.6p7 and the per-function G.6
// sections).
const (
	noParity = iota
	odd
	even
)

// checkSymmetry asserts, bit for bit, f(conj z) = conj f(z) and the parity,
// at every point given.
func checkSymmetry(t *testing.T, name string, f func(complex128) complex128, parity int, pts []complex128) {
	t.Helper()
	bad := 0
	for _, z := range pts {
		w := f(z)
		if got := f(Conj(z)); !sameC(got, Conj(w)) {
			bad++
			if bad <= 5 {
				t.Errorf("%s(conj %v) = %v, conj %s(z) = %v", name, z, got, name, Conj(w))
			}
		}
		var want complex128
		switch parity {
		case odd:
			want = -w
		case even:
			want = w
		default:
			continue
		}
		if got := f(-z); !sameC(got, want) {
			bad++
			if bad <= 5 {
				t.Errorf("%s(-(%v)) = %v, want %v", name, z, got, want)
			}
		}
	}
}

// fixturePoints returns the first two arguments of every row as points.
func fixturePoints(t *testing.T, name string) []complex128 {
	var pts []complex128
	for _, r := range loadC(t, name) {
		pts = append(pts, complex(r.args[0], r.args[1]))
	}
	return pts
}

// specialGrid is {±0, ±2^-1074, ±1, ±2^1023, ±Inf, NaN}², the grid on
// which the Annex G tables and the symmetries are checked.
func specialGrid() []complex128 {
	var v []float64
	for _, x := range []float64{0, 0x1p-1074, 1, 0x1p1023, math.Inf(1)} {
		v = append(v, x, -x)
	}
	v = append(v, math.NaN())
	var pts []complex128
	for _, x := range v {
		for _, y := range v {
			pts = append(pts, complex(x, y))
		}
	}
	return pts
}

// gcase is one Annex G expectation. anyRe/anyIm: the standard leaves the
// sign of that part unspecified.
type gcase struct {
	in, want     complex128
	anyRe, anyIm bool
}

func matchPart(got, want float64, anySign bool) bool {
	if anySign {
		got, want = math.Abs(got), math.Abs(want)
	}
	return same(got, want)
}

// checkAnnexG asserts every expectation, and its image under conj (G.6:
// f(conj z) = conj f(z) holds for every function of the family).
func checkAnnexG(t *testing.T, name string, f func(complex128) complex128, cases []gcase) {
	t.Helper()
	for _, c := range cases {
		for _, k := range []gcase{c, {Conj(c.in), Conj(c.want), c.anyRe, c.anyIm}} {
			got := f(k.in)
			if !matchPart(real(got), real(k.want), k.anyRe) || !matchPart(imag(got), imag(k.want), k.anyIm) {
				t.Errorf("%s(%v) = %v, want %v (Annex G)", name, k.in, got, k.want)
			}
		}
	}
}

var (
	inf   = math.Inf(1)
	nan   = math.NaN()
	nzero = math.Copysign(0, -1)
)

// TestHarness checks the measure itself, and that the harness sees the
// defects of math/cmplx it exists to catch (a harness that cannot fail
// measures nothing).
func TestHarness(t *testing.T) {
	if ulp(1) != 0x1p-52 || ulp(0x1p-1030) != 0x1p-1074 || !math.IsInf(ulp(-inf), 1) {
		t.Error("ulp")
	}
	for _, c := range []struct{ got, hi, lo, want float64 }{
		{1, 1, 0, 0}, {math.Nextafter(1, 2), 1, 0, 1}, {1, 1, 0x1p-54, 0.25},
		{nan, nan, 0, 0}, {1, nan, 0, inf}, {nan, 1, 0, inf},
		{inf, inf, 0, 0}, {math.MaxFloat64, inf, 0, inf}, {inf, math.MaxFloat64, 0, inf},
	} {
		if got := ulpErr(c.got, c.hi, c.lo); got != c.want {
			t.Errorf("ulpErr(%g, %g, %g) = %g, want %g", c.got, c.hi, c.lo, got, c.want)
		}
	}
	if fmtULP(inf) != "wrong" || fmtULP(2e7) != "2.0e+07" || fmtULP(0.5) != "0.50" {
		t.Error("fmtULP")
	}
	if !same(nan, -nan) || same(0, nzero) || !sameC(complex(nan, 1), complex(-nan, 1)) {
		t.Error("same")
	}
	// The harness must see the defects documented in the design.
	rows := loadC(t, "tanh")
	st, _ := measure(rows, func(a []float64) complex128 { return stdTanh(complex(a[0], a[1])) })
	if !math.IsInf(st["huge"].re, 1) {
		t.Errorf("math/cmplx Tanh in region huge measured %g ulp; the harness misses its NaN", st["huge"].re)
	}
	rows = loadC(t, "exp")
	st, _ = measure(rows, func(a []float64) complex128 { return stdExp(complex(a[0], a[1])) })
	if !math.IsInf(st["overflow"].im, 1) && !math.IsInf(st["overflow"].re, 1) {
		t.Error("math/cmplx Exp near overflow measured finite; the harness misses its spurious Inf")
	}
	if len(specialGrid()) != 121 {
		t.Error("specialGrid")
	}
}
