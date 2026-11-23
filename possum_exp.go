package special

import "github.com/go-numerics/special/dd"

// psHook is set by the measurement-A experiment to record ibetaPosSum calls.
var psHook func(a, b float64, x dd.DD, k0, n int, eps float64)
