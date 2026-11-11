// Command ulpflips judges the points where two implementations of a function
// differ against mpmath: for each line "x a b" (three float64 in Go's
// round-trip format; "x y a b" for two arguments) of the file it evaluates --expr (a Python expression in
// x) at 200 bits and reports, in ulps of the true value, the worst error of
// a and of b, and how many points each one has closer.
//
//	go -C tools run ./ulpflips --expr 'mp.erfinv(x)' flips.txt
//
// The root package's sweep tests write the file (FLIPS_FILE=<path>).
package main

import (
	"bufio"
	"fmt"
	"io"
	"math"
	"math/big"
	"os"
	"strconv"
	"strings"

	"github.com/go-numerics/special/tools/internal/mporacle"
	"github.com/spf13/cobra"
)

var osExit = os.Exit

// ulpErr returns |got - want| in ulps of want (the float64 spacing at its
// binade), want exact.
func ulpErr(got float64, want *big.Float) float64 {
	w, _ := want.Float64()
	ulp := math.Abs(math.Nextafter(w, math.Inf(1)) - w)
	if w != 0 && math.Float64bits(w)&(1<<52-1) == 0 && math.Abs(got) < math.Abs(w) {
		ulp /= 2 // the binade below a power of two is finer
	}
	d := new(big.Float).SetPrec(200).Sub(new(big.Float).SetPrec(200).SetFloat64(got), want)
	r, _ := new(big.Float).SetPrec(200).Quo(d, new(big.Float).SetPrec(200).SetFloat64(ulp)).Float64()
	return math.Abs(r)
}

type point struct{ x, y, a, b float64 } // y: the second argument, 0 for one-argument lines

func parse(r io.Reader) ([]point, error) {
	var ps []point
	sc := bufio.NewScanner(r)
	for sc.Scan() {
		f := strings.Fields(sc.Text())
		if len(f) != 3 && len(f) != 4 {
			return nil, fmt.Errorf("line %q: want 3 or 4 fields", sc.Text())
		}
		var v [4]float64
		for i := range f {
			x, err := strconv.ParseFloat(f[i], 64)
			if err != nil {
				return nil, err
			}
			v[i] = x
		}
		if len(f) == 3 {
			v[1], v[2], v[3] = 0, v[1], v[2]
		}
		ps = append(ps, point{v[0], v[1], v[2], v[3]})
	}
	return ps, sc.Err()
}

// judge returns the worst error of a and of b and the counts of points where
// a, b or neither (equal error) is closer to the true values.
func judge(ps []point, truth []*big.Float) (wa, wb float64, aCloser, bCloser, tie int) {
	for i, p := range ps {
		ea, eb := ulpErr(p.a, truth[i]), ulpErr(p.b, truth[i])
		wa, wb = max(wa, ea), max(wb, eb)
		switch {
		case ea < eb:
			aCloser++
		case eb < ea:
			bCloser++
		default:
			tie++
		}
	}
	return
}

// RootCmd is the command.
func RootCmd() *cobra.Command {
	var expr string
	cmd := &cobra.Command{
		Use:           "ulpflips --expr EXPR file",
		Short:         "judge the points where two implementations differ against mpmath",
		Args:          cobra.ExactArgs(1),
		SilenceUsage:  true,
		SilenceErrors: false,
		RunE: func(cmd *cobra.Command, args []string) error {
			f, err := os.Open(args[0])
			if err != nil {
				return err
			}
			defer f.Close()
			ps, err := parse(f)
			if err != nil {
				return err
			}
			if len(ps) == 0 {
				return fmt.Errorf("%s holds no points", args[0])
			}
			o, err := mporacle.Start()
			if err != nil {
				return err
			}
			defer o.Close()
			reqs := make([]mporacle.Req, len(ps))
			for i, p := range ps {
				reqs[i] = mporacle.Req{Expr: expr, Prec: 200, Vars: map[string]*big.Float{"x": new(big.Float).SetFloat64(p.x), "y": new(big.Float).SetFloat64(p.y)}}
			}
			truth, _, err := o.Batch(reqs)
			if err != nil {
				return err
			}
			wa, wb, ac, bc, tie := judge(ps, truth)
			fmt.Fprintf(cmd.OutOrStdout(), "%d points: a worst %.3f ulp, b worst %.3f ulp; a closer %d, b closer %d, equal %d\n", len(ps), wa, wb, ac, bc, tie)
			return nil
		},
	}
	cmd.Flags().StringVar(&expr, "expr", "", "Python expression in x and y (mp, mpf, mpmath in scope)")
	_ = cmd.MarkFlagRequired("expr")
	return cmd
}

func main() {
	if err := RootCmd().Execute(); err != nil {
		osExit(1)
	}
}
