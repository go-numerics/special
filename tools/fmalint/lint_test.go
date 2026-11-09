package main

import (
	"bytes"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// wantLines returns the 1-based lines of file holding a "// want" comment.
func wantLines(t *testing.T, file string) map[int]bool {
	t.Helper()
	b, err := os.ReadFile(file)
	if err != nil {
		t.Fatal(err)
	}
	out := map[int]bool{}
	for i, l := range strings.Split(string(b), "\n") {
		if strings.Contains(l, "// want") {
			out[i+1] = true
		}
	}
	return out
}

func TestLintCases(t *testing.T) {
	file := filepath.Join("testdata", "p", "p.go")
	fs, err := Lint(filepath.Join("testdata", "p"), ".")
	if err != nil {
		t.Fatal(err)
	}
	want := wantLines(t, file)
	got := map[int]bool{}
	for _, f := range fs {
		got[f.Pos.Line] = true
		if !want[f.Pos.Line] {
			t.Errorf("line %d reported, not expected", f.Pos.Line)
		}
	}
	for l := range want {
		if !got[l] {
			t.Errorf("line %d not reported", l)
		}
	}
}

func TestFixRoundsAndIsIdempotent(t *testing.T) {
	dir := t.TempDir()
	src, err := os.ReadFile(filepath.Join("testdata", "p", "p.go"))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module q\n\ngo 1.27\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "p.go"), src, 0o644); err != nil {
		t.Fatal(err)
	}
	fs, err := Lint(dir, ".")
	if err != nil {
		t.Fatal(err)
	}
	if n, err := Fix(fs, ""); err != nil || n != len(fs) || n == 0 {
		t.Fatalf("Fix = %d, %v (findings %d)", n, err, len(fs))
	}
	fixed, _ := os.ReadFile(filepath.Join(dir, "p.go"))
	if !bytes.Contains(fixed, []byte("float64(a*b) + c")) || !bytes.Contains(fixed, []byte("-(float64(a * b))")) {
		t.Errorf("fixed source:\n%s", fixed)
	}
	if fs, err := Lint(dir, "."); err != nil || len(fs) != 0 {
		t.Errorf("after Fix: %d findings, %v", len(fs), err)
	}
}

func TestFixMovesMathToLibm(t *testing.T) {
	for _, c := range []struct{ src, want string }{
		{ // math still used: libm added to the block
			"package q\n\nimport (\n\t\"math\"\n)\n\nfunc F(x float64) float64 { return math.Exp(x) + math.Abs(x) }\n",
			"import (\n\t\"math\"\n\t\"q/libm\"\n)\n\nfunc F(x float64) float64 { return libm.Exp(x) + math.Abs(x) }\n",
		},
		{ // math no longer used: its import becomes libm's
			"package q\n\nimport \"math\"\n\nfunc F(x float64) float64 { return math.Log(x) }\n",
			"import \"q/libm\"\n\nfunc F(x float64) float64 { return libm.Log(x) }\n",
		},
		{ // lone import, math still used: becomes a block
			"package q\n\nimport \"math\"\n\nfunc F(x float64) float64 { return math.Sin(x) + math.Sqrt(x) }\n",
			"import (\n\t\"math\"\n\t\"q/libm\"\n)\n\nfunc F(x float64) float64 { return libm.Sin(x) + math.Sqrt(x) }\n",
		},
	} {
		dir := t.TempDir()
		os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module q\n\ngo 1.27\n"), 0o644)
		p := filepath.Join(dir, "q.go")
		os.WriteFile(p, []byte(c.src), 0o644)
		fs, err := Lint(dir, ".")
		if err != nil {
			t.Fatal(err)
		}
		if _, err := Fix(fs, "q/libm"); err != nil {
			t.Fatal(err)
		}
		got, _ := os.ReadFile(p)
		if !strings.HasSuffix(string(got), c.want) {
			t.Errorf("got:\n%s\nwant suffix:\n%s", got, c.want)
		}
	}
}

func TestCommand(t *testing.T) {
	var out, errb bytes.Buffer
	cmd := RootCmd()
	cmd.SetOut(&out)
	cmd.SetErr(&errb)
	cmd.SetArgs([]string{"-C", filepath.Join("testdata", "p"), "."})
	if err := cmd.Execute(); err == nil {
		t.Error("expected an error for the findings")
	}
	if strings.Count(out.String(), "open to fusion") != len(wantLines(t, filepath.Join("testdata", "p", "p.go"))) {
		t.Errorf("output:\n%s", out.String())
	}
	exited := -1
	osExit = func(c int) { exited = c }
	defer func() { osExit = os.Exit }()
	os.Args = []string{"fmalint", "-C", filepath.Join("testdata", "p"), "."}
	main()
	if exited != 1 {
		t.Errorf("main exit = %d, want 1", exited)
	}
}

// TestLibmUpToDate regenerates internal/libm from this toolchain's
// src/math into a scratch module and requires the committed files to be
// the same bytes.
func TestLibmUpToDate(t *testing.T) {
	committed := filepath.Join("..", "..", "internal", "libm")
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module libmcheck\n\ngo 1.27\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	glue, err := os.ReadFile(filepath.Join(committed, "libm.go"))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "libm.go"), glue, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := GenLibm(runtime.GOROOT(), dir); err != nil {
		t.Fatal(err)
	}
	for _, name := range libmSources {
		want, err := os.ReadFile(filepath.Join(committed, name))
		if err != nil {
			t.Fatal(err)
		}
		got, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, want) {
			t.Errorf("internal/libm/%s is stale: run go -C tools run ./fmalint --genlibm ../internal/libm", name)
		}
	}
}

// TestSpecialIsClean: the gate itself, on the module this tool lives in.
func TestSpecialIsClean(t *testing.T) {
	fs, err := Lint(filepath.Join("..", ".."), ".", "./dd", "./cmplx", "./nd", "./internal/libm")
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range fs {
		t.Errorf("%s: product open to fusion", f.Pos)
	}
}
