package main

import (
	"fmt"
	"math/big"
	"os"
	"strconv"
	"strings"

	"github.com/go-numerics/special/tools/internal/mporacle"
)

// The oracle of the octave, cell and Temme-sum jobs: the shared mporacle
// process, each evaluation an mpmath expression at the requested precision
// on exactly bound arguments, the same calls the former fitgo server made
// (its op table is opExpr below).

// mpmathVersion is the oracle's version, printed in the generated headers.
const mpmathVersion = mporacle.MpmathVersion

type oracle struct{ o *mporacle.Oracle }

func newOracle() *oracle {
	o, err := mporacle.Start()
	check(err)
	return &oracle{o}
}

// opExpr maps an operation to its mpmath expression (the arguments are
// substituted for %s in order: a *big.Float as a bound variable, an int
// as a literal) and the bits added to the working precision.
var opExpr = map[string]struct {
	expr  string
	extra uint
}{
	"pi":          {"+mp.pi", 0},
	"cos":         {"mp.cos(%s)", 0},
	"ellipk":      {"mp.ellipk(%s)", 0},
	"ellipe":      {"mp.ellipe(%s)", 0},
	"i0e":         {"mp.besseli(0, %[1]s) * mp.exp(-%[1]s)", 40},
	"k0e":         {"mp.besselk(0, %[1]s) * mp.exp(%[1]s)", 40},
	"besselj":     {"mp.besselj(%s, %s, derivative=%s)", 0},
	"bessely":     {"mp.bessely(%s, %s, derivative=%s)", 0},
	"besseljzero": {"mp.besseljzero(%s, %s, derivative=%s)", 0},
	"besselyzero": {"mp.besselyzero(%s, %s, derivative=%s)", 0},
}

// req is one evaluation: op at prec bits on args (*big.Float or int).
type req struct {
	op   string
	prec uint
	args []any
}

// eval runs a batch of evaluations and returns their exact results.
func (o *oracle) eval(rs []req) []*big.Float {
	reqs := make([]mporacle.Req, len(rs))
	for i, r := range rs {
		e, ok := opExpr[r.op]
		if !ok {
			check(fmt.Errorf("oracle: unknown op %q", r.op))
		}
		vars := map[string]*big.Float{}
		subst := make([]any, len(r.args))
		for k, a := range r.args {
			switch v := a.(type) {
			case *big.Float:
				name := "x" + strconv.Itoa(k)
				vars[name] = v
				subst[k] = name
			case int:
				subst[k] = strconv.Itoa(v)
			default:
				check(fmt.Errorf("oracle: argument %v of %s", a, r.op))
			}
		}
		reqs[i] = mporacle.Req{Expr: fmt.Sprintf(e.expr, subst...), Prec: r.prec + e.extra, Vars: vars}
	}
	vals, _, err := o.o.Batch(reqs)
	check(err)
	return vals
}

// one evaluates a single op.
func (o *oracle) one(op string, prec uint, args ...any) *big.Float {
	return o.eval([]req{{op, prec, args}})[0]
}

func check(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

// pyRepr formats a float64 as Python's repr does (mporacle.Repr).
func pyRepr(f float64) string { return mporacle.Repr(f) }

// nstr3 formats like mpmath.nstr(v, 3) for the small positive error
// bounds printed in the headers: three significant digits, trailing zeros
// stripped down to one.
func nstr3(v *big.Float) string {
	if v.Sign() == 0 {
		return "0.0"
	}
	s := v.Text('e', 2)
	i := strings.IndexByte(s, 'e')
	m, ex := s[:i], s[i:]
	m = strings.TrimRight(m, "0")
	if strings.HasSuffix(m, ".") {
		m += "0"
	}
	if ex[1] == '+' {
		ex = "e+" + strings.TrimLeft(ex[2:], "0")
	} else {
		ex = "e-" + strings.TrimLeft(ex[2:], "0")
	}
	return m + ex
}
