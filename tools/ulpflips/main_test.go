package main

import (
	"math"
	"math/big"
	"strings"
	"testing"
)

func TestUlpErr(t *testing.T) {
	want := new(big.Float).SetPrec(200).SetFloat64(1)
	if got := ulpErr(math.Nextafter(1, 2), want); got != 1 {
		t.Errorf("one ulp above 1: %v", got)
	}
	if got := ulpErr(math.Nextafter(1, 0), want); got != 1 { // the binade below 1 is finer: still one spacing
		t.Errorf("one ulp below 1: %v", got)
	}
	if got := ulpErr(1, want); got != 0 {
		t.Errorf("exact: %v", got)
	}
}

func TestParseAndJudge(t *testing.T) {
	ps, err := parse(strings.NewReader("1 1 1.0000000000000002\n2 2.0000000000000004 2\n"))
	if err != nil || len(ps) != 2 {
		t.Fatalf("%v %v", ps, err)
	}
	truth := []*big.Float{big.NewFloat(1), big.NewFloat(2)}
	wa, wb, ac, bc, tie := judge(ps, truth)
	if wa != 1 || wb != 1 || ac != 1 || bc != 1 || tie != 0 {
		t.Errorf("%v %v %d %d %d", wa, wb, ac, bc, tie)
	}
	two, err := parse(strings.NewReader("3 4 5 6\n"))
	if err != nil || two[0] != (point{3, 4, 5, 6}) {
		t.Errorf("four fields: %v %v", two, err)
	}
	if _, err := parse(strings.NewReader("1 2\n")); err == nil {
		t.Error("two fields accepted")
	}
	if _, err := parse(strings.NewReader("1 2 x\n")); err == nil {
		t.Error("bad float accepted")
	}
}

func TestRootCmdMissingFile(t *testing.T) {
	cmd := RootCmd()
	cmd.SetArgs([]string{"--expr", "x", "/nonexistent/flips.txt"})
	cmd.SetErr(new(strings.Builder))
	if err := cmd.Execute(); err == nil {
		t.Error("missing file accepted")
	}
}
