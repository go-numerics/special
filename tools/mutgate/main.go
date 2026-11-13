// Command mutgate is the CI gate over mutsweep (go-fleettools/mutate v0.8.0):
// it sweeps every *_fast.go of the package twice, with `-mutate boundary`
// (< to <=, > to >=, ...) and with the default deletion of each refusal, and
// fails on any finding that the committed allowlist does not list with its
// reason.
//
// A finding is what mutsweep prints under "what nothing holds:": SURVIVED
// (the suite stayed green), HUNG (no answer in time), caught by PANIC (red only
// through a runtime panic, which is luck, not an assertion: special#71, #73)
// and NOT RUN. A mutation that is not a mutant, or is caught by an assertion,
// is not a finding.
//
//	go -C tools run ./mutgate -C .. -allowlist mutation-allowlist.json
//
// Three ways to read a clean result wrongly are refused. A sweep whose output
// holds no "N mutations:" summary did not finish: that is an error, not a
// pass. A directory with no *_fast.go file is an error, since a gate over
// nothing reads as green. An allowlist entry that no finding matches is stale
// and fails too, so the list can only shrink when a guard gets held.
//
// Entries are keyed by file, mode, verdict and the mutation text (the line
// number is left out on purpose: it moves with every edit above it).
package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/spf13/cobra"
)

var osExit = os.Exit

// Finding is one mutation the suite did not hold by an assertion.
type Finding struct {
	File    string `json:"file"`
	Mode    string `json:"mode"`    // "boundary" or "delete"
	Verdict string `json:"verdict"` // SURVIVED, HUNG, caught by PANIC, NOT RUN
	Text    string `json:"text"`    // the mutation as mutsweep prints it, without the line
	Reason  string `json:"reason,omitempty"`
	Line    int    `json:"-"`
}

func (f Finding) key() string {
	return f.File + "\x00" + f.Mode + "\x00" + f.Verdict + "\x00" + f.Text
}

func (f Finding) String() string {
	return fmt.Sprintf("%s %s:%d %s [%s]", f.Verdict, f.File, f.Line, f.Text, f.Mode)
}

var (
	verdicts    = []string{"SURVIVED", "caught by PANIC", "HUNG", "NOT RUN"}
	mutationsRe = regexp.MustCompile(`^\d+ mutations: \d+ caught, \d+ caught by PANIC, \d+ not mutants, \d+ survived, \d+ hung`)
	fileLineRe  = regexp.MustCompile(`^(\S+\.go):(\d+)\s+(.*)$`)
)

// Parse reads the output of one mutsweep run. It returns the findings listed
// under "what nothing holds:" and an error when the run printed no summary of
// mutations (it never finished).
func Parse(r io.Reader, mode string) ([]Finding, error) {
	var out []Finding
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1<<20), 1<<26)
	summary, inList := false, false
	for sc.Scan() {
		line := sc.Text()
		if mutationsRe.MatchString(line) {
			summary = true
			continue
		}
		if strings.HasPrefix(line, "what nothing holds:") {
			inList = true
			continue
		}
		if !inList || !strings.HasPrefix(line, "  ") {
			continue
		}
		rest := strings.TrimSpace(line)
		for _, v := range verdicts {
			if !strings.HasPrefix(rest, v) {
				continue
			}
			m := fileLineRe.FindStringSubmatch(strings.TrimSpace(strings.TrimPrefix(rest, v)))
			if m == nil {
				return nil, fmt.Errorf("unreadable finding line %q", line)
			}
			var n int
			fmt.Sscanf(m[2], "%d", &n)
			out = append(out, Finding{File: m[1], Mode: mode, Verdict: v, Text: strings.TrimSpace(m[3]), Line: n})
			break
		}
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	if !summary {
		return nil, errors.New("mutsweep printed no \"N mutations:\" summary: the sweep did not finish")
	}
	return out, nil
}

// LoadAllowlist reads the committed allowlist; every entry needs a reason.
func LoadAllowlist(r io.Reader) ([]Finding, error) {
	var l []Finding
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&l); err != nil {
		return nil, fmt.Errorf("allowlist: %w", err)
	}
	for i, e := range l {
		if e.File == "" || e.Mode == "" || e.Verdict == "" || e.Text == "" {
			return nil, fmt.Errorf("allowlist entry %d: file, mode, verdict and text are required", i)
		}
		if strings.TrimSpace(e.Reason) == "" {
			return nil, fmt.Errorf("allowlist entry %d (%s %s): a reason is required", i, e.File, e.Text)
		}
	}
	return l, nil
}

// Judge splits findings into those the allowlist covers and those it does
// not, and lists the allowlist entries nothing matched. Each entry covers one
// finding.
func Judge(found, allow []Finding) (unlisted []Finding, stale []Finding) {
	avail := map[string][]int{}
	for i, a := range allow {
		avail[a.key()] = append(avail[a.key()], i)
	}
	used := make([]bool, len(allow))
	for _, f := range found {
		if idx := avail[f.key()]; len(idx) > 0 {
			used[idx[0]] = true
			avail[f.key()] = idx[1:]
			continue
		}
		unlisted = append(unlisted, f)
	}
	for i, a := range allow {
		if !used[i] {
			stale = append(stale, a)
		}
	}
	return unlisted, stale
}

// Runner runs one mutsweep and returns its stdout+stderr and exit status.
type Runner func(ctx context.Context, dir string, args []string) (out []byte, exit int, err error)

func execRunner(bin string) Runner {
	return func(ctx context.Context, dir string, args []string) ([]byte, int, error) {
		cmd := exec.CommandContext(ctx, bin, args...)
		cmd.Dir = dir
		cmd.Stdin = nil
		var buf bytes.Buffer
		cmd.Stdout, cmd.Stderr = &buf, &buf
		err := cmd.Run()
		var ee *exec.ExitError
		if errors.As(err, &ee) {
			return buf.Bytes(), ee.ExitCode(), nil
		}
		return buf.Bytes(), 0, err
	}
}

// Options configure Gate.
type Options struct {
	Dir       string
	Allowlist string
	Timeout   time.Duration // per mutant, passed to mutsweep -timeout
	Files     []string      // default: every *_fast.go of Dir
	Modes     []string      // default: boundary, delete
	Command   []string      // default: go test -json -count=1 .
	Log       io.Writer
}

// Gate runs the sweeps and returns the verdict: ok is false when a finding is
// unlisted or an entry is stale.
func Gate(ctx context.Context, run Runner, o Options) (ok bool, err error) {
	files := o.Files
	if len(files) == 0 {
		m, gerr := filepath.Glob(filepath.Join(o.Dir, "*_fast.go"))
		if gerr != nil {
			return false, gerr
		}
		for _, f := range m {
			files = append(files, filepath.Base(f))
		}
		sort.Strings(files)
	}
	if len(files) == 0 {
		return false, fmt.Errorf("no *_fast.go file in %s: a gate over nothing reads as green", o.Dir)
	}
	modes := o.Modes
	if len(modes) == 0 {
		modes = []string{"boundary", "delete"}
	}
	cmd := o.Command
	if len(cmd) == 0 {
		cmd = []string{"go", "test", "-json", "-count=1", "."}
	}
	af, aerr := os.Open(o.Allowlist)
	if aerr != nil {
		return false, aerr
	}
	allow, aerr := LoadAllowlist(af)
	af.Close()
	if aerr != nil {
		return false, aerr
	}
	var found []Finding
	for _, mode := range modes {
		args := []string{"-dir", ".", "-mutate", mode, "-files", strings.Join(files, ","), "-timeout", o.Timeout.String(), "--"}
		args = append(args, cmd...)
		start := time.Now()
		out, exit, rerr := run(ctx, o.Dir, args)
		fmt.Fprintf(o.Log, "mutgate: -mutate %s over %s: exit %d in %s\n", mode, strings.Join(files, ","), exit, time.Since(start).Round(time.Second))
		if rerr != nil {
			return false, rerr
		}
		fmt.Fprintf(o.Log, "%s\n", out)
		if exit == 2 {
			return false, fmt.Errorf("mutsweep -mutate %s refused (exit 2); its output is above", mode)
		}
		fs, perr := Parse(bytes.NewReader(out), mode)
		if perr != nil {
			return false, fmt.Errorf("mutsweep -mutate %s: %w", mode, perr)
		}
		// A non-zero exit with nothing listed means a verdict this gate does
		// not read: refuse it rather than pass it.
		if exit != 0 && len(fs) == 0 {
			return false, fmt.Errorf("mutsweep -mutate %s exited %d but listed no finding", mode, exit)
		}
		found = append(found, fs...)
	}
	for i := range found {
		if found[i].File == "" {
			return false, errors.New("finding without a file")
		}
	}
	unlisted, stale := Judge(found, allow)
	for _, f := range found {
		fmt.Fprintf(o.Log, "finding: %s\n", f)
	}
	for _, f := range unlisted {
		fmt.Fprintf(o.Log, "UNLISTED: %s\n", f)
	}
	for _, s := range stale {
		fmt.Fprintf(o.Log, "STALE allowlist entry (nothing matches it any more, remove it): %s %s [%s] %s\n", s.File, s.Text, s.Mode, s.Verdict)
	}
	return len(unlisted) == 0 && len(stale) == 0, nil
}

// RootCmd is the command.
func RootCmd() *cobra.Command {
	o := Options{}
	var bin string
	var files, modes []string
	cmd := &cobra.Command{
		Use:           "mutgate",
		Short:         "fail on any mutsweep finding the allowlist does not list",
		SilenceUsage:  true,
		SilenceErrors: true,
		Args:          cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			o.Files, o.Modes, o.Log = files, modes, cmd.OutOrStdout()
			ok, err := Gate(cmd.Context(), execRunner(bin), o)
			if err != nil {
				return err
			}
			if !ok {
				return errors.New("mutation gate failed: unlisted findings or stale allowlist entries above")
			}
			fmt.Fprintln(cmd.OutOrStdout(), "mutgate: every finding is on the allowlist, every entry is used")
			return nil
		},
	}
	cmd.Flags().StringVarP(&o.Dir, "dir", "C", ".", "package directory to sweep")
	cmd.Flags().StringVar(&o.Allowlist, "allowlist", "mutation-allowlist.json", "committed allowlist (JSON), relative to the current directory")
	cmd.Flags().DurationVar(&o.Timeout, "timeout", 5*time.Minute, "mutsweep -timeout per mutant; mutsweep refuses under 3x the baseline suite")
	cmd.Flags().StringVar(&bin, "mutsweep", "mutsweep", "the mutsweep binary (v0.8.0 or later)")
	cmd.Flags().StringSliceVar(&files, "files", nil, "files to sweep (default: every *_fast.go of --dir)")
	cmd.Flags().StringSliceVar(&modes, "modes", nil, "mutation modes (default: boundary,delete)")
	cmd.Flags().StringSliceVar(&o.Command, "command", nil, "test command (default: go test -json -count=1 .)")
	return cmd
}

func main() {
	cmd := RootCmd()
	if err := cmd.Execute(); err != nil {
		fmt.Fprintln(cmd.ErrOrStderr(), "mutgate:", err)
		osExit(1)
	}
}
