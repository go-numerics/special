package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math/big"
	"os"
	"os/exec"
	"strconv"
	"strings"
)

// The oracle is mpmath, run once through uv as a small evaluation server:
// every request is a JSON array of [op, prec, args...] evaluations, every
// argument and result an exact binary number [mantissa in hex, exponent]
// (no decimal conversion on the way in or out), the evaluation done at
// the requested precision exactly as the mpmath expressions of the former
// Python generators did.
const oracleSource = `
import sys, json
from mpmath import mp
from mpmath.libmp import from_man_exp
def D(a):
    return mp.make_mpf(from_man_exp(int(a[0], 16), a[1])) if isinstance(a, list) else a
def E(v):
    s, m, e, b = v._mpf_
    return [('-' if s else '') + format(m, 'x'), e]
def wp(p, f):
    with mp.workprec(p):
        return f()
F = {
    'pi': lambda: +mp.pi,
    'cos': lambda x: mp.cos(x),
    'ellipk': lambda m: mp.ellipk(m),
    'ellipe': lambda m: mp.ellipe(m),
    'i0e': lambda x: wp(mp.prec + 40, lambda: mp.besseli(0, x) * mp.exp(-x)),
    'k0e': lambda x: wp(mp.prec + 40, lambda: mp.besselk(0, x) * mp.exp(x)),
    'besselj': lambda n, x, d: mp.besselj(n, x, derivative=d),
    'bessely': lambda n, x, d: mp.bessely(n, x, derivative=d),
    'besseljzero': lambda n, k, d: mp.besseljzero(n, k, derivative=d),
    'besselyzero': lambda n, k, d: mp.besselyzero(n, k, derivative=d),
}
for line in sys.stdin:
    out = []
    for r in json.loads(line):
        mp.prec = r[1]
        out.append(E(F[r[0]](*[D(a) for a in r[2:]])))
    print(json.dumps(out), flush=True)
`

// mpmathVersion is the oracle's version, printed in the generated headers.
const mpmathVersion = "1.4.1"

type oracle struct {
	in  *bufio.Writer
	out *bufio.Scanner
}

func newOracle() *oracle {
	cmd := exec.Command("uv", "run", "-q", "--with", "mpmath=="+mpmathVersion, "python", "-c", oracleSource)
	cmd.Stderr = os.Stderr
	w, err := cmd.StdinPipe()
	check(err)
	r, err := cmd.StdoutPipe()
	check(err)
	check(cmd.Start())
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1<<20), 1<<30)
	return &oracle{in: bufio.NewWriter(w), out: sc}
}

// req is one evaluation: op at prec bits on args (*big.Float or int).
type req struct {
	op   string
	prec uint
	args []any
}

// eval runs a batch of evaluations and returns their exact results.
func (o *oracle) eval(rs []req) []*big.Float {
	batch := make([][]any, len(rs))
	for i, r := range rs {
		e := []any{r.op, r.prec}
		for _, a := range r.args {
			switch v := a.(type) {
			case *big.Float:
				e = append(e, encode(v))
			default:
				e = append(e, v)
			}
		}
		batch[i] = e
	}
	b, err := json.Marshal(batch)
	check(err)
	o.in.Write(b)
	o.in.WriteByte('\n')
	check(o.in.Flush())
	if !o.out.Scan() {
		check(fmt.Errorf("oracle: %v", o.out.Err()))
	}
	var res [][2]json.RawMessage
	check(json.Unmarshal(o.out.Bytes(), &res))
	out := make([]*big.Float, len(res))
	for i, r := range res {
		var m string
		var e int
		check(json.Unmarshal(r[0], &m))
		check(json.Unmarshal(r[1], &e))
		out[i] = decode(m, e)
	}
	return out
}

// one evaluates a single op.
func (o *oracle) one(op string, prec uint, args ...any) *big.Float {
	return o.eval([]req{{op, prec, args}})[0]
}

// encode returns [hex mantissa, exponent] with v = mantissa·2^exponent.
func encode(v *big.Float) [2]any {
	if v.Sign() == 0 {
		return [2]any{"0", 0}
	}
	m := new(big.Float)
	e := v.MantExp(m) // v = m·2^e, |m| in [0.5, 1)
	p := int(v.MinPrec())
	m.SetMantExp(m, p)
	i, acc := m.Int(nil)
	if acc != big.Exact {
		panic("encode: inexact")
	}
	return [2]any{i.Text(16), e - p}
}

func decode(m string, e int) *big.Float {
	i, ok := new(big.Int).SetString(strings.TrimPrefix(m, "+"), 16)
	if !ok {
		panic("decode " + m)
	}
	f := new(big.Float).SetPrec(uint(max(i.BitLen(), 1))).SetInt(i)
	return f.SetMantExp(f, e)
}

func check(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

// pyRepr formats a float64 as Python's repr does (shortest round-trip
// digits; fixed notation for decimal exponents in [-4, 16), with ".0" for
// integers; else d.ddde±XX).
func pyRepr(f float64) string {
	e := strconv.FormatFloat(f, 'e', -1, 64)
	i := strings.IndexByte(e, 'e')
	exp, _ := strconv.Atoi(e[i+1:])
	if exp >= -4 && exp < 16 {
		s := strconv.FormatFloat(f, 'f', -1, 64)
		if !strings.ContainsAny(s, ".") {
			s += ".0"
		}
		return s
	}
	return e
}

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
