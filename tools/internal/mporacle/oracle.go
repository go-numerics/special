// Package mporacle drives mpmath as an arbitrary-precision oracle for the
// table generators under tools/. mpmath runs in one Python process started
// through uv (`uv run --with mpmath==1.4.1 python -c <script>`, the script
// below being the only Python involved); requests and replies are JSON
// lines over a pipe, and every number crosses it exactly, as an integer
// mantissa and a binary exponent.
package mporacle

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/big"
	"os"
	"os/exec"
	"strconv"
	"strings"
)

// MpmathVersion is the mpmath release the generated tables are made with.
const MpmathVersion = "1.4.1"

// The oracle: for each request, set the working precision, bind the
// variables exactly, evaluate the expression with mpmath, and reply with
// the exact value (or str(value) when asked).
const script = `
import sys, json
import mpmath
from mpmath import mp, mpf
for line in sys.stdin:
    r = json.loads(line)
    mp.prec = 4096
    env = {"mp": mp, "mpf": mpf, "mpmath": mpmath}
    for k, (m, e) in r["v"].items():
        env[k] = mpf((int(m), e))
    mp.prec = r["p"]
    v = eval(r["e"], env)
    if r.get("s"):
        out = {"s": v if isinstance(v, str) else str(v)}
    else:
        v = mpf(v)
        if v == 0:
            out = {"m": "0", "x": 0}
        elif not mp.isfinite(v):
            out = {"n": str(v)}
        else:
            sgn, m, e, _ = v._mpf_
            out = {"m": str(-m if sgn else m), "x": int(e)}
    sys.stdout.write(json.dumps(out) + "\n")
    sys.stdout.flush()
`

// Oracle is a running mpmath process.
type Oracle struct {
	cmd *exec.Cmd
	in  io.WriteCloser
	out *bufio.Scanner
}

// Req is one evaluation: expression, working precision in bits, variables.
type Req struct {
	Expr string
	Prec uint
	Vars map[string]*big.Float
	Str  bool // reply with str(value)
}

type wireReq struct {
	E string            `json:"e"`
	P uint              `json:"p"`
	V map[string][2]any `json:"v"`
	S bool              `json:"s,omitempty"`
}

type wireRep struct {
	M string `json:"m"`
	X int    `json:"x"`
	N string `json:"n"`
	S string `json:"s"`
}

func uvPath() string {
	if p, err := exec.LookPath("uv"); err == nil {
		return p
	}
	home, _ := os.UserHomeDir()
	return home + "/.local/bin/uv"
}

// Start launches the oracle.
func Start() (*Oracle, error) {
	cmd := exec.Command(uvPath(), "run", "-q", "--with", "mpmath=="+MpmathVersion, "python", "-c", script)
	cmd.Stderr = os.Stderr
	in, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	out, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	sc := bufio.NewScanner(out)
	sc.Buffer(make([]byte, 1<<20), 1<<24)
	return &Oracle{cmd, in, sc}, nil
}

// Close stops the oracle.
func (o *Oracle) Close() error {
	o.in.Close()
	return o.cmd.Wait()
}

// exact returns x as (integer mantissa, binary exponent).
func exact(x *big.Float) (string, int) {
	if x.Sign() == 0 {
		return "0", 0
	}
	e := x.MantExp(nil)
	p := int(x.MinPrec())
	m := new(big.Float).SetPrec(x.Prec()).SetMantExp(x, p-e)
	mi, acc := m.Int(nil)
	if acc != big.Exact {
		panic("mporacle: inexact mantissa")
	}
	return mi.String(), e - p
}

// Batch evaluates the requests in order (pipelined).
func (o *Oracle) Batch(reqs []Req) ([]*big.Float, []string, error) {
	errc := make(chan error, 1)
	go func() {
		w := bufio.NewWriter(o.in)
		for _, r := range reqs {
			wr := wireReq{E: r.Expr, P: r.Prec, V: map[string][2]any{}, S: r.Str}
			for k, v := range r.Vars {
				m, e := exact(v)
				wr.V[k] = [2]any{m, e}
			}
			b, err := json.Marshal(wr)
			if err != nil {
				errc <- err
				return
			}
			w.Write(b)
			w.WriteByte('\n')
		}
		errc <- w.Flush()
	}()
	vals := make([]*big.Float, len(reqs))
	strs := make([]string, len(reqs))
	for i := range reqs {
		if !o.out.Scan() {
			return nil, nil, fmt.Errorf("mporacle: no reply to %q: %v", reqs[i].Expr, o.out.Err())
		}
		var rep wireRep
		if err := json.Unmarshal(o.out.Bytes(), &rep); err != nil {
			return nil, nil, err
		}
		if reqs[i].Str {
			strs[i] = rep.S
			continue
		}
		if rep.N != "" {
			return nil, nil, fmt.Errorf("mporacle: %q = %s", reqs[i].Expr, rep.N)
		}
		mi, ok := new(big.Int).SetString(rep.M, 10)
		if !ok {
			return nil, nil, fmt.Errorf("mporacle: bad mantissa %q", rep.M)
		}
		f := new(big.Float).SetPrec(uint(max(mi.BitLen(), 1))).SetInt(mi)
		vals[i] = f.SetMantExp(f, rep.X)
	}
	if err := <-errc; err != nil {
		return nil, nil, err
	}
	return vals, strs, nil
}

// Eval evaluates one expression.
func (o *Oracle) Eval(prec uint, expr string, vars map[string]*big.Float) *big.Float {
	v, _, err := o.Batch([]Req{{Expr: expr, Prec: prec, Vars: vars}})
	if err != nil {
		panic(err)
	}
	return v[0]
}

// String evaluates str(expr).
func (o *Oracle) String(prec uint, expr string, vars map[string]*big.Float) string {
	_, s, err := o.Batch([]Req{{Expr: expr, Prec: prec, Vars: vars, Str: true}})
	if err != nil {
		panic(err)
	}
	return s[0]
}

// Float64 rounds x to the nearest float64 (ties to even), as mpmath's float().
func Float64(x *big.Float) float64 {
	f, _ := x.Float64()
	return f
}

// Repr formats f as Python's repr(float) does: the shortest round-trip
// digits, fixed notation for decimal exponents in [-4, 16), scientific
// otherwise, and always a decimal point or an exponent.
func Repr(f float64) string {
	switch {
	case math.IsNaN(f):
		return "nan"
	case math.IsInf(f, 1):
		return "inf"
	case math.IsInf(f, -1):
		return "-inf"
	case f == 0:
		if math.Signbit(f) {
			return "-0.0"
		}
		return "0.0"
	}
	s := strconv.FormatFloat(f, 'e', -1, 64) // d.ddde±XX
	neg := strings.HasPrefix(s, "-")
	s = strings.TrimPrefix(s, "-")
	mant, exps, _ := strings.Cut(s, "e")
	exp, _ := strconv.Atoi(exps)
	digits := strings.Replace(mant, ".", "", 1)
	var out string
	if exp >= -4 && exp < 16 {
		if exp >= 0 {
			if len(digits) <= exp+1 {
				out = digits + strings.Repeat("0", exp+1-len(digits)) + ".0"
			} else {
				out = digits[:exp+1] + "." + digits[exp+1:]
			}
		} else {
			out = "0." + strings.Repeat("0", -exp-1) + digits
		}
	} else {
		m := digits[:1]
		if len(digits) > 1 {
			m += "." + digits[1:]
		}
		sign := "+"
		if exp < 0 {
			sign = "-"
			exp = -exp
		}
		out = fmt.Sprintf("%se%s%02d", m, sign, exp)
	}
	if neg {
		out = "-" + out
	}
	return out
}
