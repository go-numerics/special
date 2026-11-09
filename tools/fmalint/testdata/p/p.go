package p

import "math"

func sq(x float64) float64 { return x * x } // want: returned product

func cases(a, b, c float64, f32 float32, n int) []float64 {
	r1 := a*b + c              // want
	r2 := c - a*b              // want
	r3 := float64(a*b) + c     // rounded
	r4 := math.FMA(a, b, c)    // explicit fusion
	r5 := (a * b) * c          // want: the outer product (the inner is a factor)
	r6 := c / (a * b)          // divisor
	p := a * b                 // want: assigned, may reach an addition
	r7 := sq(a) + p            // inlined product
	ok := a*b < c              // comparison
	r8 := math.Abs(a * b)      // math argument
	r9 := float64(f32*f32) + 0 // float32 product converted
	i := n*n + 1               // integer: not reported
	const k = 2.0 * 3.0        // constant: not reported
	r10 := -(a * b)            // want: negation
	_ = ok
	_ = i
	return []float64{r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, k}
}
