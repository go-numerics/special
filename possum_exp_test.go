package special

import (
	"math"
	"sort"
	"sync"
	"testing"

	"github.com/go-numerics/special/dd"
)

//go:generate go run ./internal/genpossum

type psState struct {
	dh, dl, sh, sl float64
	i              int
}

type psRec struct {
	a, b  float64
	x     dd.DD
	k0, n int
	eps   float64
	elem  int
}

// psHead is ibetaPosSum's head loop, copied verbatim.
func psHead(a, b, xh, xl float64, k0, n int) psState {
	abh, abl := dd.TwoSum(a, b)
	a1h, a1l := dd.TwoSum(a, 1)
	dh, dl := 1.0, 0.0
	sh, sl := 1.0, 0.0
	i := 0
	for ; i < n-1; i++ {
		fi := float64(i)
		s1h, e1 := dd.TwoSum(abh, fi)
		s1l := e1 + abl
		s2h, e2 := dd.TwoSum(a1h, fi)
		s2l := e2 + a1l
		th := float64(s1h * xh)
		tl := math.FMA(s1h, xh, -th) + (float64(s1h*xl) + float64(s1l*xh))
		r := 1 / s2h
		fh := float64(th * r)
		fl := float64((math.FMA(-fh, s2h, th) + tl - float64(fh*s2l)) * r)
		ph := float64(dh * fh)
		dl = math.FMA(dh, fh, -ph) + (float64(dh*fl) + float64(dl*fh))
		dh = ph
		var e float64
		sh, e = dd.TwoSum(sh, dh)
		sl += e + dl
		if i >= k0 && dh <= 0x1p-16*sh {
			i++
			break
		}
	}
	return psState{dh, dl, sh, sl, i}
}

// psTail is ibetaPosSum's remainder after the head loop, copied verbatim.
func psTail(a, b, xh float64, st psState, n int, eps float64) dd.DD {
	abh, _ := dd.TwoSum(a, b)
	a1h, _ := dd.TwoSum(a, 1)
	dh, sh, sl, i := st.dh, st.sh, st.sl, st.i
	tol := float64(eps * 0x1p-8)
	tail := 0.0
	for ; i+4 <= n-1 && dh > tol*sh && a1h+float64(i) < 0x1p200; i += 4 {
		fi := float64(i)
		xn0, xn1 := float64(xh*(abh+fi)), float64(xh*(abh+(fi+1)))
		xn2, xn3 := float64(xh*(abh+(fi+2))), float64(xh*(abh+(fi+3)))
		e0, e1, e2, e3 := a1h+fi, a1h+(fi+1), a1h+(fi+2), a1h+(fi+3)
		e23 := float64(e2 * e3)
		e123 := float64(e1 * e23)
		inner := e23 + float64(xn2*(e3+xn3))
		p := float64(xn0 * (e123 + float64(xn1*inner)))
		q := dh / float64(e0*e123)
		tail += float64(q * p)
		dh = float64(q * float64(float64(xn0*xn1)*float64(xn2*xn3)))
	}
	for ; i < n-1 && dh > tol*sh; i++ {
		fi := float64(i)
		dh = float64(dh * (xh * (abh + fi) / (a1h + fi)))
		tail += dh
	}
	sl += tail
	h, l := dd.FastTwoSum(sh, sl)
	return dd.DD{Hi: h, Lo: l}
}

func psFull(r psRec) dd.DD {
	st := psHead(r.a, r.b, r.x.Hi, r.x.Lo, r.k0, r.n)
	return psTail(r.a, r.b, r.x.Hi, st, r.n, r.eps)
}

func psFull4(r *[4]psRec) (out [4]dd.DD) {
	var a, b, xh, xl [4]float64
	var k0, n [4]int
	for l := range r {
		a[l], b[l], xh[l], xl[l], k0[l], n[l] = r[l].a, r[l].b, r[l].x.Hi, r[l].x.Lo, r[l].k0, r[l].n
	}
	st := psHead4(&a, &b, &xh, &xl, &k0, &n)
	for l := range r {
		out[l] = psTail(a[l], b[l], xh[l], st[l], n[l], r[l].eps)
	}
	return out
}

const boxN = 10000

var (
	recsOnce sync.Once
	recs     []psRec
	boxCalls int // BetaInc evaluations whose posSum path ran
)

// boxRecords runs BetaInc over the BenchmarkBetaIncBox points and records every
// ibetaPosSum call.
func boxRecords() []psRec {
	recsOnce.Do(func() {
		r := gapsRand(1)
		cur := 0
		psHook = func(a, b float64, x dd.DD, k0, n int, eps float64) {
			recs = append(recs, psRec{a, b, x, k0, n, eps, cur})
		}
		for i := 0; i < boxN; i++ {
			a := 0.5 + float64(r()*49.5)
			b := 0.5 + float64(r()*49.5)
			x := 0.01 + float64(r()*0.98)
			_ = 0.01 + float64(r()*0.98)
			cur = i
			BetaInc(a, b, x)
		}
		psHook = nil
	})
	return recs
}

func psSame(a, b dd.DD) bool {
	return math.Float64bits(a.Hi) == math.Float64bits(b.Hi) && math.Float64bits(a.Lo) == math.Float64bits(b.Lo)
}

func psCheck(t *testing.T, g [4]psRec) {
	t.Helper()
	got := psFull4(&g)
	for l := range g {
		r := g[l]
		want := ibetaPosSum(r.a, r.b, r.x, r.k0, r.n, r.eps)
		if !psSame(got[l], want) || !psSame(psFull(r), want) {
			t.Fatalf("lane %d %+v: lane4 %v scalar-split %v want %v", l, r, got[l], psFull(r), want)
		}
	}
}

func TestPosSumLanesBitIdentical(t *testing.T) {
	rs := boxRecords()
	for i := 0; i+4 <= len(rs); i += 4 {
		psCheck(t, [4]psRec{rs[i], rs[i+1], rs[i+2], rs[i+3]})
	}
	// edge cases: tiny n, huge k0, tiny a, huge b with tiny x, mixed in one group.
	e := []psRec{
		{0.5, 3, dd.F(0.3), 0, 1, ibetaEps, -1},
		{0.5, 3, dd.F(0.3), 0, 2, ibetaEps, -1},
		{0.5, 3, dd.F(0.3), 0, 3, ibetaEps, -1},
		{0.5, 3, dd.F(0.3), 0, 0, ibetaEps, -1},
		{2, 30, dd.F(0.69), 10, 40, ibetaEps, -1},
		{2, 30, dd.F(0.69), 25, 1 << 30, ibetaEps, -1},
		{1e-3, 5, dd.F(0.01), 0, 1 << 30, ibetaEps, -1},
		{0.7, 0x1p200, dd.F(0x1p-201), 0, 1 << 30, ibetaEps, -1},
		{0.7, 0x1p500, dd.F(0x1p-501), 0, 1 << 30, ibetaEps, -1},
		{40, 3, dd.F(0.999), 0, 5000, ibetaEps, -1},
		{1, 1, dd.F(1e-300), 0, 1 << 30, ibetaEps, -1},
		{0.5, 0.5, dd.F(0.9), 3, 4, ibetaEps, -1},
	}
	for i := 0; i+4 <= len(e); i += 4 {
		psCheck(t, [4]psRec{e[i], e[i+1], e[i+2], e[i+3]})
	}
	// every rotation of the edge list against box records
	for i := range e {
		psCheck(t, [4]psRec{e[i], rs[i], e[(i+5)%len(e)], rs[len(rs)-1-i]})
	}
}

func TestPosSumStats(t *testing.T) {
	rs := boxRecords()
	elems := map[int]bool{}
	for _, r := range rs {
		elems[r.elem] = true
	}
	t.Logf("box points %d; elements reaching ibetaPosSum %d (%.1f%%); calls %d", boxN, len(elems), 100*float64(len(elems))/boxN, len(rs))
	// head-loop iteration counts per call, from the head's own end index.
	cnt := make([]int, len(rs))
	var sum int
	hist := map[int]int{}
	for i, r := range rs {
		cnt[i] = psHead(r.a, r.b, r.x.Hi, r.x.Lo, r.k0, r.n).i
		sum += cnt[i]
		hist[cnt[i]]++
	}
	s := append([]int(nil), cnt...)
	sort.Ints(s)
	t.Logf("head iterations: mean %.1f min %d p10 %d p50 %d p90 %d max %d", float64(sum)/float64(len(rs)), s[0], s[len(s)/10], s[len(s)/2], s[len(s)*9/10], s[len(s)-1])
	util := func(c []int) float64 {
		var useful, spent int
		for i := 0; i+4 <= len(c); i += 4 {
			m := max(max(c[i], c[i+1]), max(c[i+2], c[i+3]))
			useful += c[i] + c[i+1] + c[i+2] + c[i+3]
			spent += 4 * m
		}
		return float64(useful) / float64(spent)
	}
	t.Logf("lane utilization, groups in arrival order: %.1f%%; groups of count-sorted calls: %.1f%%", 100*util(cnt), 100*util(s))
}

func benchRecs(b *testing.B) []psRec {
	rs := boxRecords()
	return rs[:len(rs)&^3]
}

var psSink dd.DD

// The two sides run the same benchmark names so benchstat rows match; the
// side is picked by -test.bench=PosSum.../scalar or .../lane4.
func BenchmarkPosSumHead(b *testing.B) {
	rs := benchRecs(b)
	b.Run("scalar", func(b *testing.B) {
		var s float64
		for j := 0; j < b.N; j++ {
			for i := range rs {
				r := &rs[i]
				s += psHead(r.a, r.b, r.x.Hi, r.x.Lo, r.k0, r.n).sh
			}
		}
		psSink.Hi = s
		b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N)/float64(len(rs)), "ns/elem")
	})
	b.Run("lane4", func(b *testing.B) {
		var s float64
		var a, bb, xh, xl [4]float64
		var k0, n [4]int
		for j := 0; j < b.N; j++ {
			for i := 0; i < len(rs); i += 4 {
				for l := 0; l < 4; l++ {
					r := &rs[i+l]
					a[l], bb[l], xh[l], xl[l], k0[l], n[l] = r.a, r.b, r.x.Hi, r.x.Lo, r.k0, r.n
				}
				st := psHead4(&a, &bb, &xh, &xl, &k0, &n)
				s += st[0].sh + st[1].sh + st[2].sh + st[3].sh
			}
		}
		psSink.Hi = s
		b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N)/float64(len(rs)), "ns/elem")
	})
}

// lane4sorted: the same records, scalar and lane4 both fed in head-count order
// (lanes then finish together: the utilization upper bound).
func BenchmarkPosSumHeadSorted(b *testing.B) {
	rs := append([]psRec(nil), benchRecs(b)...)
	cnt := func(r psRec) int { return psHead(r.a, r.b, r.x.Hi, r.x.Lo, r.k0, r.n).i }
	sort.Slice(rs, func(i, j int) bool { return cnt(rs[i]) < cnt(rs[j]) })
	b.Run("scalar", func(b *testing.B) {
		var s float64
		for j := 0; j < b.N; j++ {
			for i := range rs {
				r := &rs[i]
				s += psHead(r.a, r.b, r.x.Hi, r.x.Lo, r.k0, r.n).sh
			}
		}
		psSink.Hi = s
	})
	b.Run("lane4", func(b *testing.B) {
		var s float64
		var a, bb, xh, xl [4]float64
		var k0, n [4]int
		for j := 0; j < b.N; j++ {
			for i := 0; i < len(rs); i += 4 {
				for l := 0; l < 4; l++ {
					r := &rs[i+l]
					a[l], bb[l], xh[l], xl[l], k0[l], n[l] = r.a, r.b, r.x.Hi, r.x.Lo, r.k0, r.n
				}
				st := psHead4(&a, &bb, &xh, &xl, &k0, &n)
				s += st[0].sh + st[1].sh + st[2].sh + st[3].sh
			}
		}
		psSink.Hi = s
	})
}

func BenchmarkPosSumFull(b *testing.B) {
	rs := benchRecs(b)
	b.Run("scalar", func(b *testing.B) {
		var s float64
		for j := 0; j < b.N; j++ {
			for i := range rs {
				r := &rs[i]
				s += ibetaPosSum(r.a, r.b, r.x, r.k0, r.n, r.eps).Hi
			}
		}
		psSink.Hi = s
		b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N)/float64(len(rs)), "ns/elem")
	})
	b.Run("lane4", func(b *testing.B) {
		var s float64
		var g [4]psRec
		for j := 0; j < b.N; j++ {
			for i := 0; i < len(rs); i += 4 {
				copy(g[:], rs[i:i+4])
				o := psFull4(&g)
				s += o[0].Hi + o[1].Hi + o[2].Hi + o[3].Hi
			}
		}
		psSink.Hi = s
		b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N)/float64(len(rs)), "ns/elem")
	})
}
