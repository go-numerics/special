package main

import (
	"fmt"
	"os"
	"runtime"
	"sort"

	"github.com/spf13/cobra"
)

var osExit = os.Exit

// RootCmd returns the fmalint command.
func RootCmd() *cobra.Command {
	var dir, libmPath, genlibm, objdump string
	var fix, mathUses bool
	cmd := &cobra.Command{
		Use:   "fmalint [--fix] [-C dir] [packages]",
		Short: "Report (or round) float products the compiler may fuse into a multiply-add",
		Long: `fmalint lists every float64/float32 product that may reach an addition
without an explicit rounding, so that Go may fuse it into a multiply-add on
arm64, ppc64le, s390x, riscv64 and loong64 but not on amd64: the source
would then give different bits on different architectures. With --fix it
wraps each one in float64(...) (or float32(...)). Exit status 1 when
anything is reported without --fix.`,
		SilenceUsage: true,
		RunE: func(cmd *cobra.Command, args []string) error {
			if len(args) == 0 {
				args = []string{"./..."}
			}
			if objdump != "" {
				f, err := os.Open(objdump)
				if err != nil {
					return err
				}
				defer f.Close()
				mod := "github.com/go-numerics/special"
				if len(args) > 0 && args[0] != "./..." { // ./... is the package default
					mod = args[0]
				}
				fs, err := ImplicitFusions(f, mod, dir)
				if err != nil {
					return err
				}
				for _, l := range fs {
					fmt.Fprintf(cmd.OutOrStdout(), "%s: fused multiply-add without math.FMA\n", l)
				}
				if len(fs) > 0 {
					return fmt.Errorf("%d implicit fusion(s)", len(fs))
				}
				return nil
			}
			if genlibm != "" {
				if err := GenLibm(runtime.GOROOT(), genlibm); err != nil {
					return err
				}
				fmt.Fprintf(cmd.OutOrStdout(), "wrote %d file(s) into %s\n", len(libmSources), genlibm)
				return nil
			}
			if mathUses {
				m, err := MathUses(dir, args...)
				if err != nil {
					return err
				}
				names := make([]string, 0, len(m))
				for n := range m {
					names = append(names, n)
				}
				sort.Strings(names)
				for _, n := range names {
					fmt.Fprintf(cmd.OutOrStdout(), "math.%s %d\n", n, m[n])
				}
				return nil
			}
			fs, err := Lint(dir, args...)
			if err != nil {
				return err
			}
			if fix {
				n, err := Fix(fs, libmPath)
				if err != nil {
					return err
				}
				fmt.Fprintf(cmd.OutOrStdout(), "fixed %d finding(s)\n", n)
				return nil
			}
			for _, f := range fs {
				if f.Math != "" {
					fmt.Fprintf(cmd.OutOrStdout(), "%s: math.%s gives different bits on different architectures\n", f.Pos, f.Math)
					continue
				}
				fmt.Fprintf(cmd.OutOrStdout(), "%s: product open to fusion\n", f.Pos)
			}
			if len(fs) > 0 {
				return fmt.Errorf("%d finding(s)", len(fs))
			}
			return nil
		},
	}
	cmd.Flags().StringVarP(&dir, "dir", "C", ".", "directory the package patterns are relative to")
	cmd.Flags().BoolVar(&fix, "fix", false, "wrap each reported product in an explicit conversion (and, with --libm, move math calls to libm)")
	cmd.Flags().StringVar(&libmPath, "libm", "", "import path of the portable libm that replaces math's architecture-dependent functions under --fix")
	cmd.Flags().StringVar(&objdump, "objdump", "", "read `go tool objdump` output of an FMA-architecture build and list fused multiply-adds whose source line does not call math.FMA (args: module path; -C: source root)")
	cmd.Flags().StringVar(&genlibm, "genlibm", "", "regenerate the portable libm (pure-Go src/math bodies, products rounded) into this directory")
	cmd.Flags().BoolVar(&mathUses, "math", false, "list the functions of package math the packages call, with counts")
	return cmd
}

func main() {
	if err := RootCmd().Execute(); err != nil {
		osExit(1)
	}
}
