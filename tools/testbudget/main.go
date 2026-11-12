// Command testbudget fails when a package's test time, read from the event
// stream of `go test -json`, exceeds a stated budget, and lists the slowest
// tests, so that the creep of a suite is visible long before the go test
// timeout kills the run (the Windows -race lane of go-numerics/special sat a
// minute below its 10-minute default three times).
//
//	go test -race -timeout 30m -json ./... > test.json
//	go -C tools run ./testbudget --package github.com/go-numerics/special --budget 13m ../test.json
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sort"
	"time"

	"github.com/spf13/cobra"
)

var osExit = os.Exit

type event struct {
	Action  string  `json:"Action"`
	Package string  `json:"Package"`
	Test    string  `json:"Test"`
	Elapsed float64 `json:"Elapsed"`
	Output  string  `json:"Output"`
}

// TestTime is one test's elapsed time.
type TestTime struct {
	Name    string
	Elapsed time.Duration
}

// Result is the package's elapsed time and its slowest tests (top-level tests
// only, slowest first).
type Result struct {
	Elapsed time.Duration
	Slowest []TestTime
	// Failed holds the output of every failed test and of the package itself
	// when it failed (a panic, a timeout), so that a -json run is as readable
	// as a plain one where it matters.
	Failed string
}

// Read returns the elapsed time of pkg and its n slowest top-level tests from
// a `go test -json` stream. It is an error if the stream holds no final event
// of the package (a run that never finished is not under budget).
func Read(r io.Reader, pkg string, n int) (Result, error) {
	var res Result
	found := false
	var tests []TestTime
	out := map[string]string{}
	sc :=bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1<<20), 1<<26)
	for sc.Scan() {
		var e event
		if err := json.Unmarshal(sc.Bytes(), &e); err != nil {
			continue // build output and other non-JSON lines
		}
		if e.Package != pkg {
			continue
		}
		if e.Action == "output" {
			out[e.Test] += e.Output
			continue
		}
		if e.Action != "pass" && e.Action != "fail" {
			continue
		}
		if e.Action == "fail" {
			res.Failed += out[e.Test]
		}
		d := time.Duration(e.Elapsed * float64(time.Second))
		if e.Test == "" {
			res.Elapsed, found = d, true
		} else if !containsSlash(e.Test) {
			tests = append(tests, TestTime{e.Test, d})
		}
	}
	if err := sc.Err(); err != nil {
		return res, err
	}
	if !found {
		return res, fmt.Errorf("no final pass/fail event for package %s in the stream", pkg)
	}
	sort.Slice(tests, func(i, j int) bool { return tests[i].Elapsed > tests[j].Elapsed })
	if len(tests) > n {
		tests = tests[:n]
	}
	res.Slowest = tests
	return res, nil
}

func containsSlash(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] == '/' {
			return true
		}
	}
	return false
}

// RootCmd returns the testbudget command.
func RootCmd() *cobra.Command {
	var pkg string
	var budget time.Duration
	var top int
	cmd := &cobra.Command{
		Use:          "testbudget --package P --budget D FILE",
		Short:        "Fail when a package's test time in a go test -json file exceeds a budget",
		Args:         cobra.ExactArgs(1),
		SilenceUsage: true,
		RunE: func(cmd *cobra.Command, args []string) error {
			if pkg == "" || budget <= 0 {
				return fmt.Errorf("-package and a positive -budget are required")
			}
			f, err := os.Open(args[0])
			if err != nil {
				return err
			}
			defer f.Close()
			res, err := Read(f, pkg, top)
			if err != nil {
				return err
			}
			out := cmd.OutOrStdout()
			fmt.Fprint(out, res.Failed)
			fmt.Fprintf(out, "%s: %v of a %v budget\n", pkg, res.Elapsed.Round(time.Second), budget)
			for _, t := range res.Slowest {
				fmt.Fprintf(out, "  %8v  %s\n", t.Elapsed.Round(10*time.Millisecond), t.Name)
			}
			if res.Elapsed > budget {
				return fmt.Errorf("test time %v exceeds the budget %v: shrink a sweep (big ones go behind an environment size) or raise the budget on purpose", res.Elapsed.Round(time.Second), budget)
			}
			return nil
		},
	}
	cmd.Flags().StringVar(&pkg, "package", "", "import path of the package to measure")
	cmd.Flags().DurationVar(&budget, "budget", 0, "largest accepted test time of the package")
	cmd.Flags().IntVar(&top, "top", 8, "number of slowest tests to list")
	return cmd
}

func main() {
	if err := RootCmd().Execute(); err != nil {
		fmt.Fprintln(os.Stderr, "Error:", err)
		osExit(1)
	}
}
