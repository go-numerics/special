# Public functions without a performance case

Read by `refbench` in the CI mode (`cd bench && go test -run TestPerf .`): every
public function of the module has a case in `bench/perf_test.go` or is named here
with its reason. The method and the table are in [perf.md](perf.md). A row here
is a debt, not a verdict: the integer-order and combinatorial functions are the
next rows to add (go-numerics/special#33).

## Integer-order Bessel and combinatorics (SciPy has them as Python wrappers or integer ufuncs; follow-up rows)

- `Jn`: integer order; same code as `Jv` (measured as `Jv`); SciPy's `jn` takes integer arrays, not yet a row.
- `Yn`: as `Jn`, against `yn`.
- `In`: one-line wrapper of `Iv`.
- `Kn`: one-line wrapper of `Kv`.
- `nd.Jn`: as `Jn`.
- `nd.Yn`: as `Jn`.
- `nd.In`: as `In`.
- `nd.Kn`: as `Kn`.
- `Factorial`: scalar lookup/Gamma; SciPy's `factorial` is a Python wrapper (call overhead, not a ufunc loop); follow-up row.
- `LogFactorial`: as `Factorial`.
- `Factorial2`: as `Factorial`.
- `Comb`: integer combinatorics; SciPy's `comb` is a Python wrapper; follow-up row.
- `Perm`: as `Comb`.
- `nd.Factorial`: as `Factorial`.
- `nd.LogFactorial`: as `Factorial`.
- `nd.Factorial2`: as `Factorial`.
- `nd.Comb`: as `Comb`.
- `nd.Perm`: as `Comb`.

## Not an element-wise ufunc

- `LogGammaSign`: returns two values; its value part is measured as `LogGamma`, its sign as `GammaSign`.
- `JnZeros`: root finding returning a slice, not an element-wise function; SciPy's `jn_zeros` is Python; follow-up row.
- `LogSumExp`: array reduction; SciPy's `logsumexp` is a Python function with its own overhead; follow-up row.
- `Softmax`: as `LogSumExp`.
- `LogSoftmax`: as `LogSumExp`.
- `nd.LogSumExp`: as `LogSumExp`.
- `nd.Softmax`: as `LogSumExp`.
- `nd.LogSoftmax`: as `LogSumExp`.
- `nd.Map1`: generic driver; every `nd.<Function>` row measures it with a real function.
- `nd.Map2`: as `nd.Map1`.
- `nd.Map3`: as `nd.Map1`.

## Package cmplx: complex elementary functions (the reference is Go's math/cmplx, not SciPy; outside the SciPy campaign row set)

- `cmplx.Abs`: stdlib forwarder.
- `cmplx.Acos`: no SciPy reference.
- `cmplx.Acosh`: no SciPy reference.
- `cmplx.Asin`: no SciPy reference.
- `cmplx.Asinh`: no SciPy reference.
- `cmplx.Atan`: no SciPy reference.
- `cmplx.Atanh`: no SciPy reference.
- `cmplx.Conj`: stdlib forwarder.
- `cmplx.Cos`: no SciPy reference.
- `cmplx.Cosh`: no SciPy reference.
- `cmplx.Cot`: no SciPy reference.
- `cmplx.Div`: no SciPy reference.
- `cmplx.Exp`: no SciPy reference.
- `cmplx.Expm1`: no SciPy reference.
- `cmplx.Inf`: stdlib forwarder.
- `cmplx.IsInf`: stdlib forwarder.
- `cmplx.IsNaN`: stdlib forwarder.
- `cmplx.Log`: no SciPy reference.
- `cmplx.Log10`: no SciPy reference.
- `cmplx.Log1p`: no SciPy reference.
- `cmplx.Log2`: no SciPy reference.
- `cmplx.Mul`: no SciPy reference.
- `cmplx.NaN`: stdlib forwarder.
- `cmplx.Phase`: stdlib forwarder.
- `cmplx.Polar`: stdlib forwarder.
- `cmplx.Pow`: no SciPy reference.
- `cmplx.Proj`: no SciPy reference.
- `cmplx.Rect`: stdlib forwarder.
- `cmplx.Sin`: no SciPy reference.
- `cmplx.Sinh`: no SciPy reference.
- `cmplx.Sqrt`: no SciPy reference.
- `cmplx.Tan`: no SciPy reference.
- `cmplx.Tanh`: no SciPy reference.

## Package dd: double-double arithmetic (the engine behind the accuracy; no reference exists, it is measured through every function above)

- `dd.Abs`: building block.
- `dd.Add`: building block.
- `dd.AddF`: building block.
- `dd.AddProd`: building block.
- `dd.AddProdDD`: building block.
- `dd.Cmp`: building block.
- `dd.DD.Big`: conversion helper.
- `dd.DD.Float`: building block.
- `dd.DD.IsInf`: predicate.
- `dd.DD.IsNaN`: predicate.
- `dd.DD.Sign`: predicate.
- `dd.Div`: building block.
- `dd.DivF`: building block.
- `dd.Dot`: building block.
- `dd.DotDD`: building block.
- `dd.Exp`: building block.
- `dd.ExpMant`: building block.
- `dd.ExpReductionMargin`: bound helper.
- `dd.F`: constructor.
- `dd.FastTwoSum`: building block.
- `dd.FromBig`: conversion helper.
- `dd.FromInt64`: conversion helper.
- `dd.Ldexp`: building block.
- `dd.Log`: building block.
- `dd.LogAccurate`: building block.
- `dd.Mul`: building block.
- `dd.MulF`: building block.
- `dd.MulFF`: building block.
- `dd.Neg`: building block.
- `dd.Parse`: conversion helper.
- `dd.Sqr`: building block.
- `dd.Sqrt`: building block.
- `dd.Sub`: building block.
- `dd.Sum`: building block.
- `dd.TwoProd`: building block.
- `dd.TwoSum`: building block.
