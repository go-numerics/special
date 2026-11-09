package main

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
)

var osExit = os.Exit

// RootCmd returns the fmalint command.
func RootCmd() *cobra.Command {
	var dir string
	var fix bool
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
			fs, err := Lint(dir, args...)
			if err != nil {
				return err
			}
			if fix {
				n, err := Fix(fs)
				if err != nil {
					return err
				}
				fmt.Fprintf(cmd.OutOrStdout(), "rounded %d product(s)\n", n)
				return nil
			}
			for _, f := range fs {
				fmt.Fprintf(cmd.OutOrStdout(), "%s: product open to fusion\n", f.Pos)
			}
			if len(fs) > 0 {
				return fmt.Errorf("%d product(s) open to fusion", len(fs))
			}
			return nil
		},
	}
	cmd.Flags().StringVarP(&dir, "dir", "C", ".", "directory the package patterns are relative to")
	cmd.Flags().BoolVar(&fix, "fix", false, "wrap each reported product in an explicit conversion")
	return cmd
}

func main() {
	if err := RootCmd().Execute(); err != nil {
		osExit(1)
	}
}
