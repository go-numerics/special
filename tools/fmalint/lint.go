// Command fmalint finds the floating-point products that the Go compiler
// may fuse with an addition or subtraction, and can round them explicitly.
//
// The Go specification lets an implementation fuse x*y + z into one
// operation, "possibly across statements", and the compiler does so on
// arm64, ppc64le, s390x, riscv64 and loong64 (cmd/compile ARM64.rules:
// (FADDD a (FMULD x y)) => (FMADDD a x y), with no condition on the number
// of uses of the product) but not on amd64 v1 or 386. The same source then
// rounds differently on different architectures. An explicit conversion,
// float64(x*y), rounds the product and forbids the fusion.
//
// A float product is accepted only where it cannot reach an addition
// without passing through a rounding: as a factor of another product or a
// divisor/dividend, in a comparison, as the operand of float64()/float32(),
// or as an argument of a function of package math (intrinsics, or bodies
// too large to inline; math.FMA is the explicit fusion). Everywhere else
// (an operand of + or -, a negation, an assigned value, a returned value,
// an argument of a function that may be inlined, a composite literal
// element) it is reported, because SSA forwards the unrounded product to
// whatever addition consumes it, through variables and inlined calls.
package main

import (
	"fmt"
	"go/ast"
	"go/constant"
	"go/parser"
	"go/token"
	"go/types"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"

	"golang.org/x/tools/go/packages"
)

// Finding is one product open to fusion.
type Finding struct {
	Pos      token.Position
	Start    int    // byte offset of the product (or math.F selector) in its file
	End      int    // byte offset just past it
	TypeName string // float64 or float32; "" for a math call
	Math     string // the math function called, for a call whose bits depend on the architecture
	// MulAssign is set for `x *= y`: a product stored unrounded like any
	// other. Start..End span the whole statement; OpStart..OpEnd the `*=`.
	MulAssign        bool
	OpStart, OpEnd   int
	LHSStart, LHSEnd int
	RHSStart, RHSEnd int
}

// exactMath lists the functions of package math whose results are the same
// bits on every architecture (exact, or correctly rounded by IEEE 754).
var exactMath = map[string]bool{
	"Abs": true, "Ceil": true, "Copysign": true, "Dim": true, "FMA": true,
	"Float32bits": true, "Float32frombits": true, "Float64bits": true,
	"Float64frombits": true, "Floor": true, "Frexp": true, "Ilogb": true,
	"Inf": true, "IsInf": true, "IsNaN": true, "Ldexp": true, "Logb": true,
	"Max": true, "Min": true, "Mod": true, "Modf": true, "NaN": true,
	"Nextafter": true, "Nextafter32": true, "Remainder": true, "Round": true,
	"RoundToEven": true, "Signbit": true, "Sqrt": true, "Trunc": true,
}

// exactCmplx lists the functions of math/cmplx that do no arithmetic; the
// others call math's architecture-dependent functions (Hypot, Atan2,
// Sincos, Exp, Log, ...) and are reported like them.
var exactCmplx = map[string]bool{"Conj": true, "Inf": true, "NaN": true, "IsInf": true, "IsNaN": true}

// libmFuncs lists the functions internal/libm provides in place of math's.
var libmFuncs = map[string]bool{
	"Acos": true, "Asin": true, "Atan": true, "Atan2": true, "Atanh": true,
	"Cos": true, "Cosh": true, "Exp": true, "Exp2": true, "Expm1": true,
	"Hypot": true, "Log": true, "Log1p": true, "Pow": true, "Sin": true,
	"Sincos": true, "Sinh": true, "Tan": true,
}

// Lint loads the packages matched by patterns (relative to dir), their
// tests included (a test's inputs computed with a fused product differ by
// architecture as much as the code under test), and returns the findings
// sorted by position, each file once.
func Lint(dir string, patterns ...string) ([]Finding, error) {
	cfg := &packages.Config{
		Mode: packages.NeedName | packages.NeedFiles | packages.NeedSyntax |
			packages.NeedTypes | packages.NeedTypesInfo,
		Dir:   dir,
		Tests: true,
	}
	pkgs, err := packages.Load(cfg, patterns...)
	if err != nil {
		return nil, err
	}
	var out []Finding
	done := map[string]bool{} // with Tests, a file appears in several package variants
	for _, p := range pkgs {
		if len(p.Errors) > 0 {
			return nil, fmt.Errorf("%s: %v", p.PkgPath, p.Errors[0])
		}
		for _, f := range p.Syntax {
			name := p.Fset.Position(f.Pos()).Filename
			if done[name] {
				continue
			}
			done[name] = true
			out = append(out, lintFile(p.Fset, p.TypesInfo, f)...)
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Pos.Filename != out[j].Pos.Filename {
			return out[i].Pos.Filename < out[j].Pos.Filename
		}
		return out[i].Start < out[j].Start
	})
	return out, nil
}

// isPow2Const reports whether e is a constant ±2^k, the divisors the
// compiler turns into a multiplication by the (exact) reciprocal.
func isPow2Const(info *types.Info, e ast.Expr) bool {
	tv, ok := info.Types[e]
	if !ok || tv.Value == nil {
		return false
	}
	f, ok := constant.Float64Val(constant.ToFloat(tv.Value))
	if !ok || f == 0 || math.IsInf(f, 0) {
		return false
	}
	frac, _ := math.Frexp(math.Abs(f))
	return frac == 0.5
}

func floatKind(info *types.Info, e ast.Expr) (string, bool) {
	t := info.TypeOf(e)
	if t == nil {
		return "", false
	}
	if tv, ok := info.Types[e]; ok && tv.Value != nil {
		return "", false // constant: folded at compile time
	}
	b, ok := t.Underlying().(*types.Basic)
	if !ok {
		return "", false
	}
	switch b.Kind() {
	case types.Float64:
		return "float64", true
	case types.Float32:
		return "float32", true
	}
	return "", false
}

func lintFile(fset *token.FileSet, info *types.Info, f *ast.File) []Finding {
	var out []Finding
	isTest := strings.HasSuffix(fset.Position(f.Pos()).Filename, "_test.go")
	// parents maps every node to its parent, skipping nothing.
	parents := map[ast.Node]ast.Node{}
	var stack []ast.Node
	ast.Inspect(f, func(n ast.Node) bool {
		if n == nil {
			stack = stack[:len(stack)-1]
			return true
		}
		if len(stack) > 0 {
			parents[n] = stack[len(stack)-1]
		}
		stack = append(stack, n)
		return true
	})
	ast.Inspect(f, func(n ast.Node) bool {
		if as, ok := n.(*ast.AssignStmt); ok && as.Tok == token.MUL_ASSIGN && len(as.Lhs) == 1 {
			if tn, ok := floatKind(info, as.Lhs[0]); ok {
				off := func(p token.Pos) int { return fset.Position(p).Offset }
				out = append(out, Finding{
					Pos: fset.Position(as.Pos()), Start: off(as.Pos()), End: off(as.End()),
					TypeName: tn, MulAssign: true,
					OpStart: off(as.TokPos), OpEnd: off(as.TokPos) + 2,
					LHSStart: off(as.Lhs[0].Pos()), LHSEnd: off(as.Lhs[0].End()),
					RHSStart: off(as.Rhs[0].Pos()), RHSEnd: off(as.Rhs[0].End()),
				})
			}
			return true
		}
		if sel, ok := n.(*ast.SelectorExpr); ok && !isTest {
			// (tests may call math as an oracle; their products still count)
			if fn, ok := info.Uses[sel.Sel].(*types.Func); ok && fn.Pkg() != nil &&
				((fn.Pkg().Path() == "math" && !exactMath[fn.Name()]) ||
					(fn.Pkg().Path() == "math/cmplx" && !exactCmplx[fn.Name()])) {
				out = append(out, Finding{
					Pos:   fset.Position(sel.Pos()),
					Start: fset.Position(sel.Pos()).Offset,
					End:   fset.Position(sel.End()).Offset,
					Math:  fn.Name(),
				})
			}
			return true
		}
		be, ok := n.(*ast.BinaryExpr)
		if !ok || (be.Op != token.MUL && !(be.Op == token.QUO && isPow2Const(info, be.Y))) {
			return true
		}
		// x / 2^k is compiled as x * 2^-k (exact), a product like any other.
		tn, ok := floatKind(info, be)
		if !ok {
			return true
		}
		if accepted(info, parents, be) {
			return true
		}
		out = append(out, Finding{
			Pos:      fset.Position(be.Pos()),
			Start:    fset.Position(be.Pos()).Offset,
			End:      fset.Position(be.End()).Offset,
			TypeName: tn,
		})
		return true
	})
	return out
}

// accepted reports whether the product e sits where it cannot be fused.
func accepted(info *types.Info, parents map[ast.Node]ast.Node, e ast.Expr) bool {
	p := parents[e]
	for {
		pe, ok := p.(*ast.ParenExpr)
		if !ok {
			break
		}
		p = parents[pe]
	}
	switch p := p.(type) {
	case *ast.BinaryExpr:
		switch p.Op {
		case token.MUL, token.QUO:
			return true
		case token.EQL, token.NEQ, token.LSS, token.LEQ, token.GTR, token.GEQ:
			return true
		}
		return false
	case *ast.CallExpr:
		if tv, ok := info.Types[p.Fun]; ok && tv.IsType() {
			return true // conversion
		}
		if sel, ok := p.Fun.(*ast.SelectorExpr); ok {
			if id, ok := sel.X.(*ast.Ident); ok {
				if pn, ok := info.Uses[id].(*types.PkgName); ok && pn.Imported().Path() == "math" {
					return true
				}
			}
		}
		return false
	}
	return false
}

// MathUses returns, for the packages matched by patterns, how many times
// each function of package math is called. Several of them (Exp, Log,
// Log1p, Pow, Sin, ...) have assembly versions on some architectures, and
// their pure-Go versions are themselves compiled with fusion on others, so
// their bits differ by architecture even when the caller's do not.
func MathUses(dir string, patterns ...string) (map[string]int, error) {
	cfg := &packages.Config{
		Mode: packages.NeedName | packages.NeedFiles | packages.NeedSyntax |
			packages.NeedTypes | packages.NeedTypesInfo,
		Dir: dir,
	}
	pkgs, err := packages.Load(cfg, patterns...)
	if err != nil {
		return nil, err
	}
	out := map[string]int{}
	for _, p := range pkgs {
		if len(p.Errors) > 0 {
			return nil, fmt.Errorf("%s: %v", p.PkgPath, p.Errors[0])
		}
		for id, obj := range p.TypesInfo.Uses {
			fn, ok := obj.(*types.Func)
			if !ok || fn.Pkg() == nil || fn.Pkg().Path() != "math" {
				continue
			}
			_ = id
			out[fn.Name()]++
		}
	}
	return out, nil
}

// Fix wraps every product finding in an explicit conversion to its type
// and, when libmPath is set, replaces every call of a math function that
// internal/libm provides by the libm one (importing libmPath, and dropping
// "math" where nothing else uses it). It edits the files in place and
// returns the number of findings fixed; findings it cannot fix (a math
// function libm lacks) are left for the report.
func Fix(findings []Finding, libmPath string) (int, error) {
	byFile := map[string][]Finding{}
	for _, f := range findings {
		switch {
		case f.Math == "":
		case libmPath != "" && libmFuncs[f.Math]:
		default:
			continue
		}
		byFile[f.Pos.Filename] = append(byFile[f.Pos.Filename], f)
	}
	n := 0
	for name, fs := range byFile {
		src, err := os.ReadFile(name)
		if err != nil {
			return n, err
		}
		// Insert from the end so that earlier offsets stay valid; a product
		// nested in another reported one is wrapped too (harmless).
		type ins struct {
			at   int
			text string
		}
		var all []ins
		replaced := false
		for _, f := range fs {
			if f.Math != "" {
				// "math.Exp" -> "libm.Exp": drop the 4 bytes of "math",
				// insert "libm" (same length, so no offset moves).
				copy(src[f.Start:f.Start+4], "libm")
				replaced = true
				continue
			}
			if f.MulAssign {
				// x *= y  ->  x = float64(x * (y))
				// (same length for the operator, so no offset moves; gofmt
				// then normalises the spacing)
				lhs := string(src[f.LHSStart:f.LHSEnd])
				copy(src[f.OpStart:f.OpEnd], " =")
				all = append(all,
					ins{f.RHSEnd, "))"},
					ins{f.RHSStart, f.TypeName + "(" + lhs + " * ("})
				continue
			}
			all = append(all, ins{f.Start, f.TypeName + "("}, ins{f.End, ")"})
		}
		sort.SliceStable(all, func(i, j int) bool { return all[i].at > all[j].at })
		for _, in := range all {
			src = append(src[:in.at], append([]byte(in.text), src[in.at:]...)...)
		}
		if replaced {
			if src, err = fixImports(name, src, libmPath); err != nil {
				return n, err
			}
		}
		if err := os.WriteFile(name, src, 0o644); err != nil {
			return n, err
		}
		n += len(fs)
	}
	return n, nil
}

// fixImports makes src import libmPath, and drops "math" when no
// math.X selector is left.
func fixImports(name string, src []byte, libmPath string) ([]byte, error) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, name, src, 0)
	if err != nil {
		return nil, err
	}
	var mathSpec *ast.ImportSpec
	for _, is := range f.Imports {
		switch strings.Trim(is.Path.Value, `"`) {
		case libmPath:
			return src, nil // already imported
		case "math":
			if is.Name == nil {
				mathSpec = is
			}
		}
	}
	if mathSpec == nil {
		return nil, fmt.Errorf("%s: math.X replaced but no plain \"math\" import", name)
	}
	mathUsed := false
	ast.Inspect(f, func(n ast.Node) bool {
		if sel, ok := n.(*ast.SelectorExpr); ok {
			if id, ok := sel.X.(*ast.Ident); ok && id.Name == "math" {
				mathUsed = true
			}
		}
		return !mathUsed
	})
	start := fset.Position(mathSpec.Path.Pos()).Offset
	end := fset.Position(mathSpec.Path.End()).Offset
	lit := strconv.Quote(libmPath)
	if !mathUsed {
		return append(src[:start:start], append([]byte(lit), src[end:]...)...), nil
	}
	// Keep "math"; add libm on its own spec. In a parenthesised block a new
	// line after "math" does; a lone `import "math"` becomes a block.
	var gd *ast.GenDecl
	for _, d := range f.Decls {
		if g, ok := d.(*ast.GenDecl); ok && g.Tok == token.IMPORT {
			for _, s := range g.Specs {
				if s == mathSpec {
					gd = g
				}
			}
		}
	}
	if gd != nil && gd.Lparen.IsValid() {
		return append(src[:end:end], append([]byte("\n\t"+lit), src[end:]...)...), nil
	}
	return append(src[:start:start], append([]byte("(\n\t\"math\"\n\t"+lit+"\n)"), src[end:]...)...), nil
}
