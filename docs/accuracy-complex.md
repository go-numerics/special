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
| exp | moderate | 300 | 1.06 | 1.20 | 1.96 | 1.72 |
| exp | axes | 100 | 0.49 | 0.50 | 0.92 | 1.10 |
| exp | wide | 300 | 1.00 | 1.06 | 1.76 | 1.70 |
| exp | bigy | 150 | 1.29 | 1.16 | 9.94 | 7.22 |
| exp | overflow | 300 | 0.49 | 0.89 | 0.82 | wrong |
| cosh | tiny | 200 | 0.00 | 0.49 | 0.00 | 1.44 |
| cosh | moderate | 300 | 1.15 | 1.05 | 2.43 | 2.12 |
| cosh | axes | 100 | 0.49 | 0.00 | 0.94 | 0.00 |
| cosh | wide | 300 | 0.97 | 1.22 | 2.27 | 1.92 |
| cosh | bigy | 150 | 1.03 | 1.06 | 10.73 | 12.37 |
| cosh | overflow | 300 | 0.62 | 0.73 | wrong | wrong |
| sinh | tiny | 200 | 1.00 | 0.00 | 1.01 | 0.00 |
| sinh | moderate | 300 | 1.02 | 1.15 | 2.22 | 1.77 |
| sinh | axes | 100 | 0.49 | 0.50 | 1.75 | 0.97 |
| sinh | wide | 300 | 0.90 | 1.28 | wrong | wrong |
| sinh | bigy | 150 | 0.92 | 0.94 | 438.97 | 8.14 |
| sinh | overflow | 300 | 0.50 | 0.80 | wrong | wrong |
| cos | tiny | 200 | 0.02 | 0.50 | 0.02 | 1.34 |
| cos | moderate | 300 | 0.97 | 1.15 | 2.13 | 2.10 |
| cos | axes | 100 | 0.50 | 0.00 | 1.14 | 0.00 |
| cos | wide | 300 | 1.07 | 0.97 | 1.90 | 1.99 |
| cos | bigy | 150 | 1.04 | 0.97 | 16.54 | 211.37 |
| cos | overflow | 300 | 0.58 | 0.49 | wrong | wrong |
| sin | tiny | 200 | 0.22 | 1.00 | 0.22 | 1.00 |
| sin | moderate | 300 | 0.99 | 0.91 | 1.79 | 3.58 |
| sin | axes | 100 | 0.49 | 0.49 | 0.87 | 2.00 |
| sin | wide | 300 | 1.15 | 1.02 | 2.35 | 1.81 |
| sin | bigy | 150 | 1.18 | 1.00 | 10.27 | 3.82 |
| sin | overflow | 300 | 0.57 | 0.50 | wrong | wrong |
| tanh | tiny | 200 | 0.29 | 0.21 | 1.00 | 0.21 |
| tanh | moderate | 300 | 1.30 | 1.44 | 17.66 | 23.37 |
| tanh | axes | 100 | 0.49 | 0.85 | 1.64 | 819.74 |
| tanh | wide | 300 | 0.66 | 1.63 | 2.05 | 2.21 |
| tanh | huge | 300 | 0.00 | 0.98 | wrong | 2.1e+13 |
| tanh | pole | 200 | 2.55 | 1.34 | wrong | wrong |
| tan | tiny | 200 | 0.01 | 0.00 | 0.01 | 1.00 |
| tan | moderate | 300 | 1.37 | 1.27 | 3.48 | 3.03 |
| tan | axes | 100 | 1.70 | 0.49 | 1.24 | 2.39 |
| tan | wide | 300 | 1.22 | 0.79 | 2.27 | 2.41 |
| tan | huge | 300 | 0.71 | 0.00 | 1.8e+14 | wrong |
| tan | pole | 200 | 1.62 | 2.55 | 11.12 | 23.85 |
| cot | tiny | 200 | 0.49 | 0.58 | wrong | wrong |
| cot | moderate | 300 | 1.63 | 1.25 | 4.16 | 3.38 |
| cot | axes | 100 | 1.58 | 0.49 | 3.05 | 2.16 |
| cot | wide | 300 | 1.55 | 0.50 | 2.22 | 2.31 |
| cot | huge | 300 | 0.48 | 0.00 | 5.6e+14 | wrong |
| cot | pole | 200 | 1.79 | 0.50 | 12.04 | 1.51 |
| log1p | tiny | 200 | 0.06 | 0.00 | 8.9e+15 | 0.00 |
| log1p | small | 300 | 0.50 | 0.50 | 8.8e+15 | 1.52 |
| log1p | moderate | 300 | 0.49 | 0.50 | 451.93 | 1.52 |
| log1p | axes | 100 | 0.49 | 0.50 | 455375.74 | 1.4e+16 |
| log1p | cancel | 300 | 0.50 | 0.50 | 2.5e+27 | 1.26 |
| log1p | near-1 | 200 | 0.49 | 0.48 | 0.71 | 0.72 |
| log1p | huge | 200 | 0.50 | 0.49 | 0.50 | 1.04 |
| log1p | cut | 200 | 0.45 | 0.28 | 0.55 | 1.4e+16 |
| expm1 | tiny | 200 | 0.00 | 0.00 | 9.0e+15 | 0.00 |
| expm1 | small | 300 | 0.50 | 0.50 | 9.0e+15 | 1.64 |
| expm1 | moderate | 300 | 0.50 | 0.50 | 125.46 | 2.16 |
| expm1 | axes | 100 | 0.49 | 0.50 | 352772.74 | 0.85 |
| expm1 | cancel | 300 | 0.53 | 0.50 | 3.9e+25 | 1.47 |
| expm1 | wide | 200 | 0.50 | 0.50 | 1.37 | 1.76 |
| log | tiny | 200 | 0.49 | 0.49 | 118438.07 | 0.96 |
| log | subnormal | 100 | 0.50 | 0.87 | 3.3e+06 | 0.87 |
| log | moderate | 300 | 0.50 | 0.50 | 64.97 | 0.96 |
| log | band | 300 | 0.50 | 0.50 | 465.70 | 1.07 |
| log | unit | 300 | 0.49 | 0.49 | 8.9e+15 | 1.12 |
| log | huge | 200 | 0.50 | 0.49 | 0.50 | 1.06 |
| log | cut | 300 | 0.41 | 0.49 | 0.41 | 1.4e+16 |
| log | axes | 100 | 0.49 | 0.28 | 0.57 | 0.28 |
| log10 | tiny | 200 | 0.50 | 0.57 | 7.0e+10 | 1.71 |
| log10 | subnormal | 100 | 0.50 | 1.03 | 26.38 | 1.14 |
| log10 | moderate | 300 | 0.50 | 0.50 | 87.83 | 2.00 |
| log10 | band | 300 | 0.50 | 0.49 | 165.44 | 1.96 |
| log10 | unit | 300 | 0.50 | 0.50 | 8.9e+15 | 1.93 |
| log10 | huge | 200 | 0.50 | 0.50 | 1.53 | 1.97 |
| log10 | cut | 300 | 0.49 | 0.42 | 0.76 | 1.2e+16 |
| log10 | axes | 100 | 0.50 | 0.31 | 0.95 | 0.69 |
| log2 | tiny | 200 | 0.49 | 0.53 | 3.8e+10 | 1.32 |
| log2 | subnormal | 100 | 0.50 | 0.82 | 7.2e+08 | 1.27 |
| log2 | moderate | 300 | 0.50 | 0.50 | 41.11 | 1.45 |
| log2 | band | 300 | 0.50 | 0.50 | 84.23 | 1.49 |
| log2 | unit | 300 | 0.50 | 0.50 | 8.8e+15 | 1.39 |
| log2 | huge | 200 | 0.50 | 0.50 | 1.38 | 1.43 |
| log2 | cut | 300 | 0.50 | 0.39 | 0.93 | 1.0e+16 |
| log2 | axes | 100 | 0.50 | 0.39 | 0.96 | 0.39 |
| sqrt | tiny | 200 | 0.50 | 0.50 | 1.56 | 1.26 |
| sqrt | subnormal | 100 | 0.50 | 0.50 | 0.87 | 0.88 |
| sqrt | moderate | 300 | 0.50 | 0.50 | 1.65 | 1.35 |
| sqrt | band | 300 | 0.50 | 0.50 | 1.37 | 1.51 |
| sqrt | unit | 300 | 0.49 | 0.50 | 1.22 | 1.13 |
| sqrt | huge | 200 | 0.50 | 0.49 | 1.37 | 1.15 |
| sqrt | cut | 300 | 0.50 | 0.50 | 1.06 | 1.8e+16 |
| sqrt | axes | 100 | 0.50 | 0.50 | 0.50 | 0.50 |
