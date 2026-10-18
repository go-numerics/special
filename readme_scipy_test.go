// Copyright (c) the go-numerics authors
//
// SPDX-License-Identifier: BSD-3-Clause

package special

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"
)

// This package's whole premise is scipy parity: every function is measured in
// ULPs against mpmath and cross-checked against scipy, and the README says so.
// What the README did NOT say was WHICH scipy function each one is. A reader
// arriving from scipy with `gammaln` in hand had no way to find `LogGamma`
// short of reading bench/scipy_server.py, and a consumer binding this package
// into another language had to derive the whole mapping from the harness --
// which one did, and got three of the four unpaired names wrong in the
// process, because a plausible name sitting in an otherwise-measured table
// inherits the table's authority.
//
// So the README now carries the scipy name beside the Go one, and this test is
// what keeps the two together. It does not invent a mapping: it reads the
// harness, which is the only place the pairing is MEASURED, and requires the
// README to agree with it.
//
// Where the harness pairs nothing, the README has to say so explicitly rather
// than guess -- that is the `unpaired` table below, and every entry in it
// carries its reason.

// scipyRef finds the scipy function an entry is measured against: the first
// `sc.NAME` inside it. The generators write the reference as the fourth element
// of a tuple, sometimes directly (`sc.gamma`) and sometimes wrapped in a lambda
// that casts its arguments (`lambda n, k: sc.comb(int(n), int(k))`), so the
// first occurrence is the right one in both shapes.
var scipyRef = regexp.MustCompile(`\bsc\.([a-z_0-9]+)`)

var entryStart = regexp.MustCompile(`(?m)^[[:space:]]+"([A-Za-z_0-9]+)": (?:lambda:|\()`)

// fixtureGo maps a harness fixture name to the Go function it exercises, for
// the names that differ by more than case and underscores. Anything not here
// is matched by folding case and underscores away.
var fixtureGo = map[string]string{
	"loggamma":     "LogGamma",
	"logbeta":      "LogBeta",
	"log1pexp":     "Log1pExp",
	"zeta_hurwitz": "HurwitzZeta",
	"jn_zeros":     "JnZeros",
	"softmax4":     "Softmax",
	"log_softmax4": "LogSoftmax",
	"logsumexp4":   "LogSumExp",
	"polygamma1":   "Polygamma",
}

// expression refines a measured pairing where the scipy FUNCTION is right but
// the call is not a one-for-one substitution, so the README may say the call
// instead of the bare name. Each entry must still agree with the harness about
// which function it is.
var expression = map[string]struct{ fn, spelling string }{
	// fam_misc.py measures LogFactorial against `lambda n: sc.gammaln(n + 1)`,
	// and its own description says "log n! (SciPy: gammaln(n+1))". The function
	// is gammaln; the argument is shifted.
	"LogFactorial": {"gammaln", "gammaln(n+1)"},
}

// unpaired names the functions the harness measures against no scipy function,
// with what the README may say instead. A name here is NOT evidence about
// scipy; it is a statement that this package has not measured one.
var unpaired = map[string]string{
	"GammaSign":    "gammasgn", // scipy has it; nothing here pairs them
	"Kn":           "kn",       // scipy has it; nothing here pairs them
	"In":           "—",        // scipy has no `in`: it is a Python keyword
	"LogGammaSign": "—",        // scipy splits it across gammaln and gammasgn
}

// harnessPairs reads every Go -> scipy pairing the harness records, by two
// rules, because the harness states them two ways.
//
//  1. POSITIONALLY. A generator entry is
//     "fixture": lambda: (nargs, cases, mpmath_ref, SCIPY_REF, "description"),
//     and bench/scipy_server.py's CASES is "GoName": (SCIPY_REF, ranges). The
//     first `sc.NAME` in the entry is the reference. Entries are NOT one line
//     each -- yn and jn_zeros wrap -- so the window runs to the next entry.
//
//  2. BY NAME, as a fallback. logsumexp4's reference is a named helper (lse4)
//     whose body calls sc.logsumexp, and following one level of indirection in
//     Python is not worth a parser. Instead: if the harness mentions an
//     `sc.NAME` anywhere whose folded spelling equals the function's, that is
//     the pairing. It cannot invent one -- the name has to appear.
//
// Rule 1 catches the pairings where the scipy name DIFFERS (LogGamma ->
// gammaln, Log1pExp -> softplus, GammaP -> gammainc); rule 2 catches the ones
// where it agrees but the reference is indirect. Neither can fabricate a name
// that is not written in the harness.
func harnessPairs(t *testing.T) map[string]string {
	t.Helper()
	exported := exportedFuncs(t)
	pairs := map[string]string{}

	files, err := filepath.Glob("testdata/gen/fam_*.py")
	if err != nil {
		t.Fatalf("globbing the generators: %v", err)
	}
	if len(files) == 0 {
		t.Fatal("read no generators under testdata/gen -- this test would then check the " +
			"README against nothing, which is not the same as agreement")
	}
	files = append(files, "bench/scipy_server.py")

	var all strings.Builder
	for _, f := range files {
		b, err := os.ReadFile(f)
		if err != nil {
			if f == "bench/scipy_server.py" {
				continue // optional
			}
			t.Fatalf("%s: %v", f, err)
		}
		all.Write(b)
		all.WriteString("\n")
		src := string(b)

		// Rule 1: each entry, windowed to the next one.
		locs := entryStart.FindAllStringSubmatchIndex(src, -1)
		for i, loc := range locs {
			name := src[loc[2]:loc[3]]
			end := len(src)
			if i+1 < len(locs) {
				end = locs[i+1][0]
			}
			if r := scipyRef.FindStringSubmatch(src[loc[1]:end]); r != nil {
				if g := goNameOf(exported, name); g != "" {
					pairs[g] = r[1]
				}
			}
		}
	}

	// Rule 2: an sc.NAME anywhere that spells an exported function.
	mentioned := map[string]bool{}
	for _, m := range scipyRef.FindAllStringSubmatch(all.String(), -1) {
		mentioned[m[1]] = true
	}
	for _, g := range exported {
		if _, ok := pairs[g]; ok {
			continue
		}
		for s := range mentioned {
			if fold(s) == fold(g) {
				pairs[g] = s
				break
			}
		}
	}

	if len(pairs) < 40 {
		t.Fatalf("read only %d pairings out of the harness; the entry patterns no longer "+
			"match how it is written, so this test proves nothing", len(pairs))
	}
	return pairs
}

// goNameOf resolves a harness name to the exported Go function it stands for.
func goNameOf(exported []string, name string) string {
	if g, ok := fixtureGo[name]; ok {
		return g
	}
	want := fold(name)
	for _, g := range exported {
		if fold(g) == want {
			return g
		}
	}
	return ""
}

func fold(s string) string { return strings.ToLower(strings.ReplaceAll(s, "_", "")) }

// readmeRow matches a Functions-table row's two cells: the family and the
// functions, each function written as `Go` (`scipy`) or `Go` (—).
var readmeRow = regexp.MustCompile("(?m)^\\| ([^|-][^|]*) \\| (.+) \\|$")

// readmeEntry matches one `Go` (`scipy`) pair inside a row.
var readmeEntry = regexp.MustCompile("`([A-Za-z0-9]+)`" + ` \((?:` + "`" + `([a-z_0-9()+]+)` + "`" + `|(—))\)`)

// TestReadmeNamesEveryScipyEquivalent is the gate: the README's Functions table
// must name, for every exported function, the scipy function the harness
// measures it against -- or, where the harness measures none, exactly what the
// unpaired table permits.
func TestReadmeNamesEveryScipyEquivalent(t *testing.T) {
	exported := exportedFuncs(t)
	pairs := harnessPairs(t)

	b, err := os.ReadFile("README.md")
	if err != nil {
		t.Fatal(err)
	}
	got := map[string]string{}
	for _, row := range readmeRow.FindAllStringSubmatch(string(b), -1) {
		for _, e := range readmeEntry.FindAllStringSubmatch(row[2], -1) {
			if e[3] != "" {
				got[e[1]] = "—"
				continue
			}
			got[e[1]] = e[2]
		}
	}
	if len(got) == 0 {
		t.Fatal("read no `Go` (`scipy`) entries from the README's Functions table -- " +
			"the table's shape changed and this test is checking nothing")
	}

	// Every exported function must appear, with the right scipy name.
	for _, g := range exported {
		want, measured := pairs[g]
		if !measured {
			w, ok := unpaired[g]
			if !ok {
				t.Errorf("%s: the harness pairs no scipy function with it and `unpaired` does not "+
					"say what the README may claim -- decide, do not guess", g)
				continue
			}
			want = w
		}
		switch g2, ok := got[g]; {
		case !ok:
			t.Errorf("%s is exported but the README's Functions table does not list it with a scipy name", g)
		case expression[g].spelling != "" && g2 == expression[g].spelling && want == expression[g].fn:
			// a refinement of the measured pairing, declared above
		case g2 != want:
			src := "the harness"
			if !measured {
				src = "the unpaired table"
			}
			t.Errorf("%s: the README says scipy %q, %s says %q", g, g2, src, want)
		}
	}
	// And nothing in the README that is not exported.
	for g := range got {
		if !contains(exported, g) {
			t.Errorf("the README's Functions table lists %s, which this package does not export", g)
		}
	}

	if testing.Verbose() {
		var names []string
		for _, g := range exported {
			names = append(names, g)
		}
		sort.Strings(names)
		for _, g := range names {
			if s, ok := pairs[g]; ok {
				t.Logf("%-14s %-16s measured", g, s)
			} else {
				t.Logf("%-14s %-16s UNPAIRED", g, unpaired[g])
			}
		}
		t.Logf("%d exported, %d measured against scipy, %d unpaired",
			len(exported), len(pairs), len(exported)-len(pairs))
	}
}

func contains(h []string, s string) bool {
	for _, x := range h {
		if x == s {
			return true
		}
	}
	return false
}

var _ = fmt.Sprint

// exportedFuncs lists this package's exported functions by parsing its own
// source. A hard-coded list would be a third place to keep in step; the source
// is the only one that cannot be wrong.
func exportedFuncs(t *testing.T) []string {
	t.Helper()
	fset := token.NewFileSet()
	pkgs, err := parser.ParseDir(fset, ".", func(fi os.FileInfo) bool {
		return !strings.HasSuffix(fi.Name(), "_test.go")
	}, 0)
	if err != nil {
		t.Fatalf("parsing this package: %v", err)
	}
	p, ok := pkgs["special"]
	if !ok {
		t.Fatalf("no package %q in this directory -- found %v", "special", keys(pkgs))
	}
	var out []string
	for _, f := range p.Files {
		for _, d := range f.Decls {
			fd, ok := d.(*ast.FuncDecl)
			if !ok || fd.Recv != nil || !fd.Name.IsExported() {
				continue
			}
			out = append(out, fd.Name.Name)
		}
	}
	if len(out) < 40 {
		t.Fatalf("found only %d exported functions; the parse read almost nothing", len(out))
	}
	sort.Strings(out)
	return out
}

func keys[V any](m map[string]V) []string {
	var out []string
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// TestHarnessPairingsAreReadable asserts the extraction itself, separately from
// the README: the harness must yield a pairing for the large majority of the
// exported functions, and the ones it does not must be exactly the set
// `unpaired` names. Without this, a pattern that silently stopped matching
// would make every function look unpaired and the README check would start
// demanding the unpaired spelling for all of them.
//
// Run with -v to print the whole mapping, which is what the README's column is
// built from.
func TestHarnessPairingsAreReadable(t *testing.T) {
	exported := exportedFuncs(t)
	pairs := harnessPairs(t)

	var missing []string
	for _, g := range exported {
		if _, ok := pairs[g]; !ok {
			missing = append(missing, g)
		}
		if testing.Verbose() {
			if s, ok := pairs[g]; ok {
				t.Logf("%-14s %-16s measured", g, s)
			} else {
				t.Logf("%-14s %-16s UNPAIRED", g, unpaired[g])
			}
		}
	}
	sort.Strings(missing)
	var want []string
	for g := range unpaired {
		want = append(want, g)
	}
	sort.Strings(want)
	if strings.Join(missing, " ") != strings.Join(want, " ") {
		t.Errorf("the harness pairs nothing for [%s]; `unpaired` names [%s].\n"+
			"If a function gained a scipy reference, drop it from `unpaired`; if one lost its "+
			"reference, that is the harness regressing and not a documentation question.",
			strings.Join(missing, " "), strings.Join(want, " "))
	}
	t.Logf("%d exported, %d measured against scipy, %d unpaired",
		len(exported), len(pairs), len(missing))
}
