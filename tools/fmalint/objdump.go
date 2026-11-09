package main

import (
	"bufio"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// The binary witness: `go tool objdump` of an arm64 (or ppc64le, s390x)
// build lists every fused multiply-add the compiler emitted, with the
// source line it came from. A fused instruction whose line does not call
// math.FMA is an implicit fusion the source analysis missed.

var (
	objText = regexp.MustCompile(`^TEXT (\S+)\(SB\) (\S+)$`)
	objLine = regexp.MustCompile(`^\s+(\S+\.go):(\d+)\s+0x[0-9a-f]+\s+[0-9a-f]+\s+(\S+)`)
	fmaOps  = map[string]bool{
		"FMADDD": true, "FMSUBD": true, "FNMADDD": true, "FNMSUBD": true, // arm64
		"FMADD": true, "FMSUB": true, "FNMADD": true, "FNMSUB": true, // ppc64le
		"FMADDS": true, "FMSUBS": true, "FNMADDS": true, "FNMSUBS": true,
		"WFMADB": true, "WFMSDB": true, "FMADD.D": true, // s390x, loong64/riscv64
	}
)

// ImplicitFusions reads objdump output and returns "file:line" for every
// fused multiply-add emitted in a function of a package under modulePath
// whose source line (looked up under srcRoot by base name) has no "FMA(".
func ImplicitFusions(r io.Reader, modulePath, srcRoot string) ([]string, error) {
	byBase := map[string][]string{}
	err := filepath.WalkDir(srcRoot, func(p string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() && (d.Name() == "testdata" || d.Name() == "tools" || strings.HasPrefix(d.Name(), ".")) && p != srcRoot {
			return filepath.SkipDir
		}
		if strings.HasSuffix(p, ".go") {
			byBase[filepath.Base(p)] = append(byBase[filepath.Base(p)], p)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	lines := map[string][]string{}
	srcLine := func(file string, n int) []string {
		var out []string
		for _, p := range byBase[file] {
			ls, ok := lines[p]
			if !ok {
				b, err := os.ReadFile(p)
				if err != nil {
					continue
				}
				ls = strings.Split(string(b), "\n")
				lines[p] = ls
			}
			if n-1 < len(ls) {
				out = append(out, ls[n-1])
			}
		}
		return out
	}
	seen := map[string]bool{}
	inModule := false
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		t := sc.Text()
		if m := objText.FindStringSubmatch(t); m != nil {
			inModule = strings.HasPrefix(m[1], modulePath) && !strings.Contains(m[1], "_test") && !strings.HasPrefix(m[1], modulePath+"/tools")
			continue
		}
		if !inModule {
			continue
		}
		m := objLine.FindStringSubmatch(t)
		if m == nil || !fmaOps[m[3]] || strings.HasSuffix(m[1], "_test.go") {
			// Test files are left out: their inputs come from math/rand,
			// whose inlined Float64/NormFloat64 fuse with the caller's
			// arithmetic, and their comparison columns measure Go's own
			// complex arithmetic on purpose.
			continue
		}
		n, _ := strconv.Atoi(m[2])
		explicit := false
		cands := srcLine(m[1], n)
		for _, l := range cands {
			if strings.Contains(l, "FMA(") {
				explicit = true
			}
		}
		if !explicit && len(cands) > 0 {
			seen[m[1]+":"+m[2]] = true
		}
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	out := make([]string, 0, len(seen))
	for k := range seen {
		out = append(out, k)
	}
	sort.Strings(out)
	return out, nil
}
