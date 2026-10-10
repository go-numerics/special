package nd

import (
	"github.com/go-ndarray/ndarray"
	"github.com/go-numerics/special"
)

// LogSumExp returns log Σ exp(x) over all elements of x (special.LogSumExp).
func LogSumExp(x *ndarray.Array) (float64, error) {
	d, _, err := operand(x)
	if err != nil {
		return 0, err
	}
	return special.LogSumExp(d), nil
}

// Softmax returns exp(x)/Σ exp(x) over all elements of x, with x's shape
// (special.Softmax).
func Softmax(x *ndarray.Array) (*ndarray.Array, error) {
	d, f32, err := operand(x)
	if err != nil {
		return nil, err
	}
	return result(special.Softmax(nil, d), x.Shape(), f32), nil
}

// LogSoftmax returns x - LogSumExp(x) over all elements of x, with x's shape
// (special.LogSoftmax).
func LogSoftmax(x *ndarray.Array) (*ndarray.Array, error) {
	d, f32, err := operand(x)
	if err != nil {
		return nil, err
	}
	return result(special.LogSoftmax(nil, d), x.Shape(), f32), nil
}
