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
