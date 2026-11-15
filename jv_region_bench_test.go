package special

import (
	"math/rand"
	"testing"
)

// Regions of the refbench distribution of Jv and Yv (v in [0, 20), x in
// [0.1, 50)), named by the branch of besselJYPos that answers them, for the
// time profile of #33 (BenchmarkJvRegion, BenchmarkYvRegion).
var jvRegions = []struct {
	name     string
	vlo, vhi float64
	xlo, xhi float64
	neg      bool // order -v
}{
	{"temme x<=1/4", 0, 20, 0.1, 0.25, false},
	{"small x series", 0, 20, 0.25, 1, false},
	{"x<v", 1, 20, 1, 20, false},
	{"x>=v x<=20", 0, 20, 1, 20, false},
	{"x>20 asym", 0, 20, 20, 50, false},
	{"negative v", 0, 20, 0.1, 50, true},
}

func jvRegionPoints(vlo, vhi, xlo, xhi float64, neg bool, onlyBelow bool) [][2]float64 {
	r := rand.New(rand.NewSource(33))
	var pts [][2]float64
	for len(pts) < 2048 {
		v := vlo + float64(r.Float64()*(vhi-vlo))
		x := xlo + float64(r.Float64()*(xhi-xlo))
		if onlyBelow && x >= v {
			continue
		}
		if !onlyBelow && vlo == 0 && vhi == 20 && xhi <= 20 && xlo >= 1 && x < v && !neg {
			continue
		}
		if neg {
			v = -v
		}
		pts = append(pts, [2]float64{v, x})
	}
	return pts
}

func benchRegions(b *testing.B, f func(v, x float64) float64) {
	for _, rg := range jvRegions {
		pts := jvRegionPoints(rg.vlo, rg.vhi, rg.xlo, rg.xhi, rg.neg, rg.name == "x<v")
		b.Run(rg.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p := pts[i&2047]
				besselSink += f(p[0], p[1])
			}
		})
	}
}

func BenchmarkJvRegion(b *testing.B) { benchRegions(b, Jv) }
func BenchmarkYvRegion(b *testing.B) { benchRegions(b, Yv) }

// BenchmarkJvMix and BenchmarkYvMix run the refbench distribution itself
// (v in [0, 20), x in [0.1, 50)), for a profile of where the time goes.
func benchMix(b *testing.B, f func(v, x float64) float64) {
	r := rand.New(rand.NewSource(7))
	pts := make([][2]float64, 4096)
	for i := range pts {
		pts[i] = [2]float64{float64(r.Float64() * 20), 0.1 + float64(r.Float64()*49.9)}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p := pts[i&4095]
		besselSink += f(p[0], p[1])
	}
}

func BenchmarkJvMix(b *testing.B) { benchMix(b, Jv) }
func BenchmarkYvMix(b *testing.B) { benchMix(b, Yv) }
