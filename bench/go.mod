module github.com/go-numerics/special/bench

go 1.27.3

require (
	github.com/go-ndarray/ndarray v0.34.0
	github.com/go-numerics/special v0.0.0
	github.com/go-scikit/refbench v0.3.0
)

require github.com/go-fleettools/fleettools v0.81.0 // indirect

replace github.com/go-numerics/special => ../
