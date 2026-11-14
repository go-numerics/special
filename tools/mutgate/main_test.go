package main

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// Shapes of the mutsweep v0.8.0 report (README of go-fleettools/mutate).
const clean = `mutsweep: 5 comparisons, 3m0s each at most, through "go test -json -count=1 ." (the suite itself takes 7.571s)
[1/5] caught       kn_fast.go:84  k < n  [< to <=]  8s

5 mutations: 5 caught, 0 caught by PANIC, 0 not mutants, 0 survived, 0 hung
`

const dirty = `[5/5] caught by PANIC kn_fast.go:102  x < knFastMaxX  [< to <=]  1s  no test failed by an assertion

5 mutations: 3 caught, 1 caught by PANIC, 0 not mutants, 1 survived, 0 hung

what nothing holds:
  SURVIVED         kn_fast.go:89  s <= math.MaxFloat64  [<= to <]
  caught by PANIC  kn_fast.go:102  x < knFastMaxX  [< to <=]
  HUNG             jn_yn_fast.go:628  a || b
`

func TestParse(t *testing.T) {
	fs, err := Parse(strings.NewReader(clean), "boundary")
	if err != nil || len(fs) != 0 {
		t.Fatalf("clean: %v, %v", fs, err)
	}
	fs, err = Parse(strings.NewReader(dirty), "boundary")
	if err != nil || len(fs) != 3 {
		t.Fatalf("dirty: %v, %v", fs, err)
	}
	want := []Finding{
		{File: "kn_fast.go", Mode: "boundary", Verdict: "SURVIVED", Text: "s <= math.MaxFloat64  [<= to <]", Line: 89},
		{File: "kn_fast.go", Mode: "boundary", Verdict: "caught by PANIC", Text: "x < knFastMaxX  [< to <=]", Line: 102},
		{File: "jn_yn_fast.go", Mode: "boundary", Verdict: "HUNG", Text: "a || b", Line: 628},
	}
	for i := range want {
		if fs[i] != want[i] {
			t.Errorf("finding %d = %+v, want %+v", i, fs[i], want[i])
		}
	}
}

// A sweep that died before its summary is not a clean one.
func TestParseUnfinished(t *testing.T) {
	if _, err := Parse(strings.NewReader("[1/5] caught kn_fast.go:84 k < n\n"), "delete"); err == nil {
		t.Error("an output without a summary must be an error")
	}
	if _, err := Parse(strings.NewReader(""), "delete"); err == nil {
		t.Error("an empty output must be an error")
	}
	bad := "5 mutations: 0 caught, 0 caught by PANIC, 0 not mutants, 1 survived, 0 hung\nwhat nothing holds:\n  SURVIVED  no position here\n"
	if _, err := Parse(strings.NewReader(bad), "delete"); err == nil {
		t.Error("an unreadable finding line must be an error")
	}
}

func TestLoadAllowlist(t *testing.T) {
	ok := `[{"file":"a.go","func":"F","mode":"delete","verdict":"SURVIVED","text":"x","reason":"equivalent"}]`
	if l, err := LoadAllowlist(strings.NewReader(ok)); err != nil || len(l) != 1 {
		t.Fatalf("%v %v", l, err)
	}
	for name, in := range map[string]string{
		"no reason":     `[{"file":"a.go","func":"F","mode":"delete","verdict":"SURVIVED","text":"x"}]`,
		"blank reason":  `[{"file":"a.go","func":"F","mode":"delete","verdict":"SURVIVED","text":"x","reason":" "}]`,
		"no text":       `[{"file":"a.go","func":"F","mode":"delete","verdict":"SURVIVED","reason":"r"}]`,
		"no func":       `[{"file":"a.go","mode":"delete","verdict":"SURVIVED","text":"x","reason":"r"}]`,
		"unknown field": `[{"file":"a.go","func":"F","mode":"delete","verdict":"SURVIVED","text":"x","reason":"r","why":1}]`,
		"not json":      `nope`,
	} {
		if _, err := LoadAllowlist(strings.NewReader(in)); err == nil {
			t.Errorf("%s: must be refused", name)
		}
	}
}

func TestJudge(t *testing.T) {
	a := Finding{File: "a.go", Func: "F", Mode: "delete", Verdict: "SURVIVED", Text: "x", Reason: "r"}
	b := Finding{File: "a.go", Func: "F", Mode: "boundary", Verdict: "SURVIVED", Text: "x"}
	// The function takes part in the key: a generic text (`a < b`) listed for
	// one function must not hide the same text in another.
	other := a
	other.Func = "G"
	if un, st := Judge([]Finding{other}, []Finding{a}); len(un) != 1 || len(st) != 1 {
		t.Errorf("other function: unlisted %v stale %v", un, st)
	}
	// The line does not take part in the key; the mode and the verdict do.
	un, st := Judge([]Finding{{File: "a.go", Func: "F", Mode: "delete", Verdict: "SURVIVED", Text: "x", Line: 9}}, []Finding{a})
	if len(un) != 0 || len(st) != 0 {
		t.Errorf("listed finding: unlisted %v stale %v", un, st)
	}
	un, st = Judge([]Finding{b}, []Finding{a})
	if len(un) != 1 || len(st) != 1 {
		t.Errorf("other mode: unlisted %v stale %v", un, st)
	}
	pan := a
	pan.Verdict = "caught by PANIC"
	un, _ = Judge([]Finding{pan}, []Finding{a})
	if len(un) != 1 {
		t.Error("a PANIC is not covered by a SURVIVED entry")
	}
	// One entry covers one finding.
	un, _ = Judge([]Finding{a, a}, []Finding{a})
	if len(un) != 1 {
		t.Errorf("two findings, one entry: unlisted %v", un)
	}
	_, st = Judge(nil, []Finding{a})
	if len(st) != 1 {
		t.Error("an entry nothing matches is stale")
	}
}

func fakeRunner(outs map[string]string, exits map[string]int) Runner {
	return func(_ context.Context, _ string, args []string) ([]byte, int, error) {
		for i, a := range args {
			if a == "-mutate" {
				return []byte(outs[args[i+1]]), exits[args[i+1]], nil
			}
		}
		return nil, 0, nil
	}
}

func fixture(t *testing.T, allow string, files ...string) Options {
	t.Helper()
	d := t.TempDir()
	for _, f := range files {
		// one function, F, spanning every line a report names
		if err := os.WriteFile(filepath.Join(d, f), []byte("package p\n\nfunc F() {\n"+strings.Repeat("\n", 700)+"}\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	al := filepath.Join(d, "allow.json")
	if err := os.WriteFile(al, []byte(allow), 0o644); err != nil {
		t.Fatal(err)
	}
	return Options{Dir: d, Allowlist: al, Timeout: time.Minute, Log: &bytes.Buffer{}}
}

func TestGate(t *testing.T) {
	listed := `[
 {"file":"kn_fast.go","func":"F","mode":"boundary","verdict":"SURVIVED","text":"s <= math.MaxFloat64  [<= to <]","reason":"r1"},
 {"file":"kn_fast.go","func":"F","mode":"boundary","verdict":"caught by PANIC","text":"x < knFastMaxX  [< to <=]","reason":"r2"},
 {"file":"jn_yn_fast.go","func":"F","mode":"boundary","verdict":"HUNG","text":"a || b","reason":"r3"}]`
	outs := map[string]string{"boundary": dirty, "delete": clean}
	exits := map[string]int{"boundary": 1}
	ctx := context.Background()

	ok, err := Gate(ctx, fakeRunner(outs, exits), fixture(t, listed, "kn_fast.go", "jn_yn_fast.go"))
	if err != nil || !ok {
		t.Fatalf("all listed: ok %v err %v", ok, err)
	}

	o := fixture(t, `[]`, "kn_fast.go", "jn_yn_fast.go")
	ok, err = Gate(ctx, fakeRunner(outs, exits), o)
	if err != nil || ok {
		t.Fatalf("nothing listed: ok %v err %v", ok, err)
	}
	if !strings.Contains(o.Log.(*bytes.Buffer).String(), "UNLISTED: SURVIVED kn_fast.go:89") {
		t.Errorf("log lacks the unlisted finding:\n%s", o.Log)
	}

	// kn_fast.go swept alone: its two entries are stale, the entry of
	// jn_yn_fast.go (not swept) is not reported.
	o = fixture(t, listed, "kn_fast.go", "jn_yn_fast.go")
	o.Files = []string{"kn_fast.go"}
	ok, err = Gate(ctx, fakeRunner(map[string]string{"boundary": clean, "delete": clean}, nil), o)
	log := o.Log.(*bytes.Buffer).String()
	if err != nil || ok || strings.Count(log, "STALE") != 2 || strings.Contains(log, "STALE allowlist entry (nothing matches it any more, remove it): jn_yn_fast.go") {
		t.Fatalf("stale entries of a partial sweep: ok %v err %v\n%s", ok, err, log)
	}
	// ... and sweeping only the other file with its finding listed is green.
	o = fixture(t, listed, "kn_fast.go", "jn_yn_fast.go")
	o.Files = []string{"jn_yn_fast.go"}
	ok, err = Gate(ctx, fakeRunner(map[string]string{"boundary": "5 mutations: 4 caught, 0 caught by PANIC, 0 not mutants, 0 survived, 1 hung\n\nwhat nothing holds:\n  HUNG             jn_yn_fast.go:628  a || b\n", "delete": clean}, exits), o)
	if err != nil || !ok {
		t.Fatalf("partial sweep, entry of the other file: ok %v err %v\n%s", ok, err, o.Log)
	}

	o = fixture(t, listed, "kn_fast.go")
	ok, err = Gate(ctx, fakeRunner(map[string]string{"boundary": clean, "delete": clean}, nil), o)
	if err != nil || ok || !strings.Contains(o.Log.(*bytes.Buffer).String(), "STALE") {
		t.Fatalf("stale entries: ok %v err %v\n%s", ok, err, o.Log)
	}
}

func TestFuncAt(t *testing.T) {
	d := t.TempDir()
	src := "package p\n\nfunc A() {\n\t_ = 1\n}\n\ntype T struct{}\n\nfunc (t *T) M() {\n\t_ = 2\n}\n\nfunc (t T) V() {\n\t_ = 3\n}\n\ntype G[X any] struct{}\n\nfunc (g G[X]) N() {\n\t_ = 4\n}\n"
	if err := os.WriteFile(filepath.Join(d, "f.go"), []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	for line, want := range map[int]string{4: "A", 10: "T.M", 14: "T.V", 20: "G.N"} {
		if got, err := FuncAt(d, "f.go", line); err != nil || got != want {
			t.Errorf("line %d: %q, %v, want %q", line, got, err, want)
		}
	}
	if _, err := FuncAt(d, "f.go", 7); err == nil {
		t.Error("a line outside every function must be an error")
	}
	if _, err := FuncAt(d, "missing.go", 1); err == nil {
		t.Error("a missing file must be an error")
	}
	// A finding whose line is in no function stops the gate rather than
	// matching an entry by an empty name.
	o := fixture(t, `[]`, "kn_fast.go")
	if _, err := Gate(context.Background(), fakeRunner(map[string]string{"boundary": "1 mutations: 0 caught, 0 caught by PANIC, 0 not mutants, 1 survived, 0 hung\n\nwhat nothing holds:\n  SURVIVED  kn_fast.go:1  x\n", "delete": clean}, map[string]int{"boundary": 1}), o); err == nil {
		t.Error("a finding outside every function must be an error")
	}
}

func TestChangedFast(t *testing.T) {
	run := func(out string, exit int) Runner {
		return func(_ context.Context, _ string, args []string) ([]byte, int, error) {
			if strings.Join(args, " ") != "diff --name-only --diff-filter=d --relative main...HEAD" {
				t.Errorf("git args %v", args)
			}
			return []byte(out), exit, nil
		}
	}
	got, err := ChangedFast(context.Background(), run("README.md\nkv_fast.go\nsub/x_fast.go\nkv_fast_test.go\nbinom_fast.go\n", 0), ".", "main")
	if err != nil || strings.Join(got, ",") != "binom_fast.go,kv_fast.go" {
		t.Errorf("got %v, %v", got, err)
	}
	if got, err := ChangedFast(context.Background(), run("", 0), ".", "main"); err != nil || len(got) != 0 {
		t.Errorf("none changed: %v, %v", got, err)
	}
	if _, err := ChangedFast(context.Background(), run("fatal: bad revision", 128), ".", "main"); err == nil {
		t.Error("a failing git must be an error, not an empty list")
	}
}

func TestRootCmdChangedSince(t *testing.T) {
	var out, errb bytes.Buffer
	cmd := RootCmd()
	cmd.SetOut(&out)
	cmd.SetErr(&errb)
	cmd.SetArgs([]string{"-C", t.TempDir(), "--changed-since", "main", "--files", "a_fast.go"})
	if err := cmd.Execute(); err == nil {
		t.Error("--changed-since with --files must be refused")
	}
	// a git that fails is an error
	cmd = RootCmd()
	cmd.SetOut(&out)
	cmd.SetErr(&errb)
	cmd.SetArgs([]string{"-C", t.TempDir(), "--changed-since", "main", "--git", "false"})
	if err := cmd.Execute(); err == nil {
		t.Error("a failing git must be an error")
	}
	// a git that lists nothing: nothing to sweep, said out loud, success
	cmd = RootCmd()
	out.Reset()
	cmd.SetOut(&out)
	cmd.SetErr(&errb)
	cmd.SetArgs([]string{"-C", t.TempDir(), "--changed-since", "main", "--git", "true"})
	if err := cmd.Execute(); err != nil || !strings.Contains(out.String(), "nothing to sweep") {
		t.Errorf("nothing changed: %v %q", err, out.String())
	}
}

func TestGateRefusals(t *testing.T) {
	ctx := context.Background()
	if _, err := Gate(ctx, fakeRunner(nil, nil), fixture(t, `[]`)); err == nil {
		t.Error("no *_fast.go file must be an error")
	}
	// refused by mutsweep
	if _, err := Gate(ctx, fakeRunner(map[string]string{"boundary": "mutsweep: the command FAILS"}, map[string]int{"boundary": 2}), fixture(t, `[]`, "kn_fast.go")); err == nil {
		t.Error("exit 2 must be an error")
	}
	// unfinished sweep
	if _, err := Gate(ctx, fakeRunner(map[string]string{"boundary": "[1/2] caught kn_fast.go:3 x\n"}, nil), fixture(t, `[]`, "kn_fast.go")); err == nil {
		t.Error("an unfinished sweep must be an error")
	}
	// non-zero exit that lists nothing
	if _, err := Gate(ctx, fakeRunner(map[string]string{"boundary": clean, "delete": clean}, map[string]int{"boundary": 1}), fixture(t, `[]`, "kn_fast.go")); err == nil {
		t.Error("exit 1 with no finding must be an error")
	}
	// missing allowlist
	o := fixture(t, `[]`, "kn_fast.go")
	o.Allowlist = filepath.Join(o.Dir, "missing.json")
	if _, err := Gate(ctx, fakeRunner(nil, nil), o); err == nil {
		t.Error("a missing allowlist must be an error")
	}
}

func TestRootCmd(t *testing.T) {
	var out, errb bytes.Buffer
	cmd := RootCmd()
	cmd.SetOut(&out)
	cmd.SetErr(&errb)
	cmd.SetArgs([]string{"-C", t.TempDir(), "--allowlist", "nope.json"})
	if err := cmd.Execute(); err == nil {
		t.Error("a directory with no *_fast.go must fail")
	}
}

func TestMain_(t *testing.T) {
	code := -1
	old := osExit
	osExit = func(c int) { code = c }
	defer func() { osExit = old }()
	oldArgs := os.Args
	os.Args = []string{"mutgate", "-C", t.TempDir()}
	defer func() { os.Args = oldArgs }()
	main()
	if code != 1 {
		t.Errorf("exit code %d, want 1", code)
	}
}
