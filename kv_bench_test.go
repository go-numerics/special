package special

import (
	"math"
	"strings"
	"testing"
)

// BenchmarkKvRows times Kv, Kve (and Kn, the unchanged control) over the
// finite rows of their mpmath fixtures, per region and over all of them: the
// inputs the refbench smoke test times. ns/op is per row.
func BenchmarkKvRows(b *testing.B) {
	type fnT struct {
		name  string
		rows  []fixtureRow
		call  func(a []float64) float64
		slice func(r fixtureRow) bool
	}
	finite := func(r fixtureRow) bool {
		for _, a := range r.args {
			if math.IsNaN(a) || math.IsInf(a, 0) {
				return false
			}
		}
		return !math.IsNaN(r.hi) && !math.IsInf(r.hi, 0)
	}
	for _, f := range []fnT{
		{"Kv", append(loadFixture(b, "kv"), loadFixture(b, "kv_edge")...), func(a []float64) float64 { return Kv(a[0], a[1]) }, finite},
		{"Kve", append(loadFixture(b, "kve"), loadFixture(b, "kve_edge")...), func(a []float64) float64 { return Kve(a[0], a[1]) }, finite},
		{"Kn", append(loadFixture(b, "kn"), loadFixture(b, "kn_edge")...), func(a []float64) float64 { return Kn(int(a[0]), a[1]) }, finite},
	} {
		byReg := map[string][][]float64{"all": nil}
		for _, r := range f.rows {
			if !f.slice(r) {
				continue
			}
			reg := strings.ReplaceAll(strings.ReplaceAll(r.region, " ", "_"), "/", "_")
			byReg[reg] = append(byReg[reg], r.args)
			byReg["all"] = append(byReg["all"], r.args)
		}
		for reg, rows := range byReg {
			b.Run(f.name+"/"+reg, func(b *testing.B) {
				var sink float64
				for i := 0; i < b.N; i++ {
					sink += f.call(rows[i%len(rows)])
				}
				if sink == 0.123 {
					b.Log(sink)
				}
			})
		}
	}
}
