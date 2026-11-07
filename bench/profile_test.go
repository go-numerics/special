package bench

import "testing"

// BenchmarkRow runs one registry row under the Go benchmark driver, for pprof:
//
//	go test -run xxx -bench 'Row/BetaInc$' -cpuprofile cpu.out .
//
// Its times are not a verdict (no reference, no load gate); use TestPerf.
func BenchmarkRow(b *testing.B) {
	for _, c := range cases(nil) {
		b.Run(c.Name, func(b *testing.B) {
			b.ReportAllocs()
			c.Go(b.N)
		})
	}
}
