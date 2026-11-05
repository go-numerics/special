# Accuracy of the complex functions (`special/cmplx`)

Each part of each result is measured in ulps of that part's exact value,
against mpmath 1.4.1 evaluated at 192 and 320 bits
(`cmplx/testdata/c_*.tsv.gz`, written by `go -C tools run ./gencomplex`;
a point whose two evaluations differ by more than 2^-80 relative is
dropped and counted in the fixture header). "wrong" is a NaN or an
infinity where the exact part is finite, or the converse. The regions are
those of [design-complex.md](design-complex.md), "Validation".

The table below is rewritten by the command in its marker line. "-" in the
Re/Im columns: the function is not implemented yet. The comparison column
is `math/cmplx` of Go 1.27.3 for the functions, and the builtin operators
`*` and `/` for `Mul` and `Div` (measured on darwin/arm64, where Go fuses
the products of `*` into multiply-adds).

<!-- table: go test -run TestAccuracy ./cmplx -args -cmplx.table=../docs/accuracy-complex.md -->

| function | region | points | Re (ulp) | Im (ulp) | math/cmplx Re | math/cmplx Im |
|---|---|---:|---:|---:|---:|---:|
| mul | moderate | 400 | 0.50 | 0.50 | 21.51 | 14.22 |
| mul | cancel | 400 | 0.50 | 0.50 | 7.9e+15 | 0.93 |
| mul | wide | 400 | 0.50 | 0.49 | 0.71 | 0.87 |
| mul | subnormal | 200 | 0.50 | 0.49 | 1.00 | 1.00 |
| mul | huge | 200 | 0.00 | 0.00 | wrong | wrong |
| div | moderate | 400 | 0.50 | 0.49 | 30.74 | 6.07 |
| div | cancel | 400 | 0.50 | 0.50 | 1.2e+16 | 1.49 |
| div | wide | 400 | 0.50 | 0.50 | 7606.91 | 2.4e+13 |
| div | subnormal | 200 | 0.50 | 0.50 | 2.1e+12 | 3.2e+12 |
| div | huge | 200 | 0.50 | 0.50 | 1.29 | 1.43 |
| exp | tiny | 200 | 0.26 | 0.33 | 0.26 | 0.33 |
| exp | moderate | 300 | 1.28 | 1.72 | 1.96 | 1.72 |
| exp | axes | 100 | 0.92 | 1.10 | 0.92 | 1.10 |
| exp | wide | 300 | 1.30 | 1.48 | 1.76 | 1.70 |
| exp | bigy | 150 | 9.94 | 7.22 | 9.94 | 7.22 |
| exp | overflow | 300 | 0.58 | 0.89 | 0.82 | wrong |
| cosh | tiny | 200 | 0.00 | 0.49 | 0.00 | 1.44 |
| cosh | moderate | 300 | 1.21 | 1.47 | 2.43 | 2.12 |
| cosh | axes | 100 | 0.81 | 0.00 | 0.94 | 0.00 |
| cosh | wide | 300 | 1.27 | 1.23 | 2.27 | 1.92 |
| cosh | bigy | 150 | 10.73 | 13.37 | 10.73 | 12.37 |
| cosh | overflow | 300 | 0.62 | 0.73 | wrong | wrong |
| sinh | tiny | 200 | 1.00 | 0.00 | 1.01 | 0.00 |
| sinh | moderate | 300 | 1.34 | 1.23 | 2.22 | 1.77 |
| sinh | axes | 100 | 0.49 | 0.97 | 1.75 | 0.97 |
| sinh | wide | 300 | 1.67 | 1.32 | wrong | wrong |
| sinh | bigy | 150 | 438.97 | 8.14 | 438.97 | 8.14 |
| sinh | overflow | 300 | 0.50 | 0.80 | wrong | wrong |
| cos | tiny | 200 | 0.02 | 0.50 | 0.02 | 1.34 |
| cos | moderate | 300 | 1.32 | 1.75 | 2.13 | 2.10 |
| cos | axes | 100 | 1.14 | 0.00 | 1.14 | 0.00 |
| cos | wide | 300 | 1.43 | 1.56 | 1.90 | 1.99 |
| cos | bigy | 150 | 15.54 | 211.37 | 16.54 | 211.37 |
| cos | overflow | 300 | 0.58 | 0.49 | wrong | wrong |
| sin | tiny | 200 | 0.22 | 1.00 | 0.22 | 1.00 |
| sin | moderate | 300 | 1.46 | 1.64 | 1.79 | 3.58 |
| sin | axes | 100 | 0.87 | 0.49 | 0.87 | 2.00 |
| sin | wide | 300 | 1.35 | 1.15 | 2.35 | 1.81 |
| sin | bigy | 150 | 11.27 | 2.25 | 10.27 | 3.82 |
| sin | overflow | 300 | 0.57 | 0.50 | wrong | wrong |
| tanh | tiny | 200 | - | - | 1.00 | 0.21 |
| tanh | moderate | 300 | - | - | 17.66 | 23.37 |
| tanh | axes | 100 | - | - | 1.64 | 819.74 |
| tanh | wide | 300 | - | - | 2.05 | 2.21 |
| tanh | huge | 300 | - | - | wrong | 2.1e+13 |
| tanh | pole | 200 | - | - | wrong | wrong |
| tan | tiny | 200 | - | - | 0.01 | 1.00 |
| tan | moderate | 300 | - | - | 3.48 | 3.03 |
| tan | axes | 100 | - | - | 1.24 | 2.39 |
| tan | wide | 300 | - | - | 2.27 | 2.41 |
| tan | huge | 300 | - | - | 1.8e+14 | wrong |
| tan | pole | 200 | - | - | 11.12 | 23.85 |
| cot | tiny | 200 | - | - | wrong | wrong |
| cot | moderate | 300 | - | - | 4.16 | 3.38 |
| cot | axes | 100 | - | - | 3.05 | 2.16 |
| cot | wide | 300 | - | - | 2.22 | 2.31 |
| cot | huge | 300 | - | - | 5.6e+14 | wrong |
| cot | pole | 200 | - | - | 12.04 | 1.51 |
