package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

const stream = `{"Action":"run","Package":"p","Test":"TestA"}
not json: build output
{"Action":"pass","Package":"p","Test":"TestA","Elapsed":2.5}
{"Action":"pass","Package":"p","Test":"TestA/sub","Elapsed":9}
{"Action":"pass","Package":"p","Test":"TestB","Elapsed":4}
{"Action":"pass","Package":"other","Elapsed":99}
{"Action":"pass","Package":"p","Elapsed":7}
`

func TestRead(t *testing.T) {
	r, err := Read(strings.NewReader(stream), "p", 1)
	if err != nil {
		t.Fatal(err)
	}
	if r.Elapsed != 7*time.Second || len(r.Slowest) != 1 || r.Slowest[0].Name != "TestB" {
		t.Fatalf("got %+v", r)
	}
	if _, err := Read(strings.NewReader(`{"Action":"pass","Package":"p","Test":"TestA","Elapsed":1}`), "p", 3); err == nil {
		t.Fatal("a stream without the package's final event must be an error")
	}
}

func run(t *testing.T, budget string) (string, error) {
	t.Helper()
	f := filepath.Join(t.TempDir(), "t.json")
	if err := os.WriteFile(f, []byte(stream), 0o644); err != nil {
		t.Fatal(err)
	}
	cmd := RootCmd()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetErr(&out)
	cmd.SetArgs([]string{"--package", "p", "--budget", budget, f})
	err := cmd.Execute()
	return out.String(), err
}

func TestBudget(t *testing.T) {
	if out, err := run(t, "8s"); err != nil || !strings.Contains(out, "TestB") {
		t.Fatalf("under budget: %v %q", err, out)
	}
	if _, err := run(t, "6s"); err == nil {
		t.Fatal("7s against a 6s budget must fail")
	}
	cmd := RootCmd()
	cmd.SetArgs([]string{"x.json"})
	cmd.SetOut(&bytes.Buffer{})
	cmd.SetErr(&bytes.Buffer{})
	if cmd.Execute() == nil {
		t.Fatal("missing flags must fail")
	}
}

func TestMainExit(t *testing.T) {
	var code int
	osExit = func(c int) { code = c }
	defer func() { osExit = os.Exit }()
	os.Args = []string{"testbudget", "--package", "p", "--budget", "1s", filepath.Join(t.TempDir(), "absent.json")}
	main()
	if code != 1 {
		t.Fatalf("exit %d, want 1", code)
	}
}

func readStr(t *testing.T, s, pkg string) Result {
	t.Helper()
	r, err := Read(strings.NewReader(s), pkg, 3)
	if err != nil {
		t.Fatal(err)
	}
	return r
}

func TestFailedOutputEveryPackage(t *testing.T) {
	s := `{"Action":"output","Package":"p","Test":"TestA","Output":"p-boom\n"}
{"Action":"fail","Package":"p","Test":"TestA","Elapsed":1}
{"Action":"output","Package":"q","Test":"TestQ","Output":"q-boom\n"}
{"Action":"fail","Package":"q","Test":"TestQ","Elapsed":1}
{"Action":"output","Package":"q","Test":"TestOK","Output":"fine\n"}
{"Action":"pass","Package":"q","Test":"TestOK","Elapsed":1}
{"Action":"output","Package":"r","Output":"panic: timeout\n"}
{"Action":"fail","Package":"r","Elapsed":3}
{"Action":"fail","Package":"p","Elapsed":2}
`
	r := readStr(t, s, "p")
	for _, want := range []string{"p-boom", "q-boom", "panic: timeout"} {
		if !strings.Contains(r.Failed, want) {
			t.Errorf("Failed lacks %q: %q", want, r.Failed)
		}
	}
	if strings.Contains(r.Failed, "fine") {
		t.Errorf("a passing test's output leaked: %q", r.Failed)
	}
	if r.Elapsed != 2*time.Second {
		t.Errorf("elapsed %v: other packages must not count", r.Elapsed)
	}
}

func TestBuildFailure(t *testing.T) {
	s := `{"ImportPath":"p [p.test]","Action":"build-output","Output":"./x.go:3:1: syntax error\n"}
{"ImportPath":"p [p.test]","Action":"build-fail"}
{"Action":"start","Package":"p"}
{"Action":"fail","Package":"p","Elapsed":0}
`
	r := readStr(t, s, "p")
	if !r.BuildFailed || !strings.Contains(r.Build, "syntax error") {
		t.Fatalf("got %+v", r)
	}
	f := filepath.Join(t.TempDir(), "b.json")
	if err := os.WriteFile(f, []byte(s), 0o644); err != nil {
		t.Fatal(err)
	}
	cmd := RootCmd()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetErr(&out)
	cmd.SetArgs([]string{"--package", "p", "--budget", "1h", f})
	if cmd.Execute() == nil {
		t.Fatal("a build failure must exit non-zero")
	}
	if !strings.Contains(out.String(), "syntax error") {
		t.Fatalf("compiler output not printed: %q", out.String())
	}
	// build-fail with no package event at all is still a failure, not a
	// "no final event" read error that hides the compiler output.
	only := `{"ImportPath":"p [p.test]","Action":"build-output","Output":"oops\n"}
{"ImportPath":"p [p.test]","Action":"build-fail"}
`
	if r := readStr(t, only, "p"); !r.BuildFailed || r.Build != "oops\n" {
		t.Fatalf("got %+v", r)
	}
}

func TestBudgetBoundary(t *testing.T) {
	// the stream's package time is 7s: exactly the budget passes.
	if _, err := run(t, "7s"); err != nil {
		t.Fatalf("7s against a 7s budget must pass: %v", err)
	}
	if _, err := run(t, "6999ms"); err == nil {
		t.Fatal("7s against 6.999s must fail")
	}
}

func TestSeveralFinalEvents(t *testing.T) {
	s := `{"Action":"pass","Package":"p","Elapsed":3}
{"Action":"pass","Package":"p","Elapsed":4}
`
	if r := readStr(t, s, "p"); r.Elapsed != 7*time.Second {
		t.Fatalf("elapsed %v, want the sum 7s", r.Elapsed)
	}
}

func TestCachedRun(t *testing.T) {
	s := `{"Action":"output","Package":"p","Output":"ok  \tp\t(cached)\n"}
{"Action":"pass","Package":"p","Elapsed":0}
`
	r := readStr(t, s, "p")
	if r.Elapsed != 0 || r.Failed != "" || r.BuildFailed {
		t.Fatalf("got %+v", r)
	}
}
