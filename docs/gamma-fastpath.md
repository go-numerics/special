# Gamma: certified float64 fast paths

`Gamma` (gamma.go) evaluates x >= 1 with float64 kernels (`gammaMid`,
`gammaLarge`) and, before this change, evaluated 0 < x < 1 and x < 0 only in
double-double (`gammaPos`, `gammaNegDD`). gamma_fast.go adds a float64
evaluation for the double-double regions whose result is **the same float64,
bit for bit**, as the double-double path: Ziv's strategy (A. Ziv, "Fast
evaluation of elementary mathematical functions with correctly rounded last
bit", ACM TOMS 17(3), 1991) with the existing path as the accurate step.

## The contract kept

The reference result is `RN(D)`, D being the exact value of the
double-double the reference computes (`dd.DD.Float` is `Hi + Lo`, the
rounding of D for a normalised pair; `dd.Exp` is `math.Ldexp(RN(M), k)`).
D is not Γ(x): it carries the reference's own errors, among them float64
roundings (the Horner evaluation of the fitted polynomial in `evalPiecesDD`,
`stirlingSeries`). A fast value h + l with a proven bound
|h + l - D| <= e decides RN(D) = h whenever every real within e of h + l
rounds to h (`zivRound`); otherwise the reference runs.

Two choices make the bounds small:

1. **Reuse the reference's float64-rounded values bit for bit.** The fast
   paths call `evalPiecesDD(gamma1pPieces, t)` and `sinPiDD(x)` themselves,
   so the rounding errors of those values (about 2^-57 relative, far above a
   double-double's) are common to both sides and do not enter e.
2. **Bound against D, not against Γ.** e is the sum of the fast path's error
   and the reference's error with respect to the same ideal value.

u = 2^-53 below. Each bound is stated with respect to an ideal value V₀; the
decision uses e = E·|h| with E at least the sum of both sides' relative
bounds times (1 + 2^-50) (|V₀| <= |h|(1 + 2u + ...)). The double-double
operation bounds are those of Joldes, Muller and Popescu, "Tight and rigorous
error bounds for basic building blocks of double-word arithmetic", ACM TOMS
44(2), 2017, as quoted in dd/dd.go: AddF (DWPlusFP) 2u², Add 3u², MulF
(DWTimesFP) 2u², Mul (DWTimesDW, FMA) 5u², Div (DWDivDW) 15u² + 56u³.

## `zivRound(h, l, e)`

True iff h + l + [-e, e] lies strictly inside the rounding cell of h: half
an ulp on either side, a quarter ulp on the side towards zero when |h| is a
power of two. The tests `l + e < half` and `l - e > -low` are floating-point
sums compared with representable bounds; rounding is monotonic, so a
computed sum below a representable bound implies the exact sum is below it.
Strict inequalities exclude ties. h subnormal or within 2^54 of it, infinite
or NaN: refused (the reference runs).

## Region A: 2^-56 <= x < 1 and -1/2 < x <= -2^-56 (`gammaRecipFast`)

Reference: `dd.DivF(gamma1p(x), x)`, gamma1p(t) = AddF(Mul(MulF(TwoSum(t, -1), t), F), 1),
F = fh + fl = `evalPiecesDD(gamma1pPieces, t)` (common to both sides).
Ideal: Q₀ = (1 + t(t-1)F)/x, t = x.

Facts used: on [-1/2, 1], F lies in [0.42, 1.03] and every piece's constant
c_n lies in the same range, so |c_n| <= 2.5|fh|; fl = TwoSum low part
(<= u|fh|) plus c0lo (<= u|c_n|) rounded, so |fl| <= 4u|fh|. With
G = t(t-1)F and Γ1 = 1 + G, |G/Γ1| <= 0.7725/1.7725 < 0.436.

Fast side (`gamma1pFast`, then one division):

| step | error, relative to the ideal value |
|---|---|
| P = t(t-1): uh·t exact (TwoProd), ul·t and the sum rounded | 2.02u² |
| G = P·F: ph·fh exact; ph·fl, pl·fh and two sums rounded; pl·fl dropped | 4.1u² + 5.1u² + P's 2.02u² + O(u³) = 11.3u² |
| Γ1 = 1 + G: Fast2Sum exact, the low sum (below 3.3u·Γ1) rounded | 11.3u²·0.436 + 3.3u² = 8.3u² |
| Q = Γ1/x: qh, exact remainder (FMA), (r + lo) and /x rounded | 4.1u² |

Total fast: 12.4u². Reference: MulF 2u² and Mul 20u² (5u² scaled by 4 for
the non-normalised F) on G, times 0.436, + AddF 2u², + Div 16u²: 27.6u².
Sum 40u², times (1 + 2^-50): **E = 2^-99** (128u², margin 3x).

## Region B: -34 < x <= -1/2 (`gammaProdFast`)

Reference: Γ(1+t) / (t Π_{k<n} (x+k)), n = ceil(-x) <= 34, t = x + n, every
x + k exact; the denominator by n MulF from t, then Div.

Fast side: the product in two interleaved hi + lo chains (k even and k odd,
m_a = ceil(n/2), m_b = floor(n/2) steps), each step p = h·f exact (TwoProd)
and lo' = err + lo·f. After j steps |lo| <= j·u(1+u)|h|; a step adds at most
u|lo·f| + u|lo'| <= (2j+1)u²(1+3u) relative, so a chain of m steps is within
m²u²(1+3u). Combining the chains: ah·bh exact; ah·bl, al·bh and two sums
rounded, al·bl dropped: (3n + 1 + m_a·m_b)u². For n = 34: 578u² + 392u² =
970u². Normalising the denominator is exact (Fast2Sum); the division
qh + ((r + lo) - qh·dl)/dh with the exact remainder adds 12u² (first-order
expansion in dl/dh <= u included); Γ(1+t) as in region A, 8.3u². Total fast
991u².

Reference: n MulF (2u² each, 68u²) + Div 16u² + gamma1p 11.6u²: 96u².
Sum 1087u², times (1 + 2^-50): **E = 2^-94** (4096u², margin 3.7x).

## Region C: -190 <= x <= -34 (`gammaReflFast`)

Reference: s = sinPiDD(x) (common), q = Div(π, MulF(s, x)),
lq = Log(|q.Hi|) + q.Lo/q.Hi, lg = log Γ(y) for y = -x by Stirling in
double-double with S(y) = `stirlingSeries(y)` in float64, result
`dd.Exp(lq - lg)` = math.Ldexp(RN(M), k), sign of s.
Ideal: L₀ = log π - log|x·s| - log Γ(y); Γ = sign(s)·e^{L₀}.

Bounds on the double-double building blocks, derived from dd/dd.go:

- `dd.Log`, absolute: atanh tail 2s³/3 + ... in float64 with |s| <= 2^-8.5,
  3u·2^-26.1 = 2^-77.5; truncation after 2s⁷/7, 2s⁹/9 <= 2^-78.7;
  k·Ln2Lo32 rounded and represented, 2^-84·|k| <= 2^-78.5 (|k| <= 46);
  double-double sums below 2^-100. Sum 2^-76.5; used: **2^-75**.
- `dd.ExpMant`, relative: P(s) in float64, |p| <= 2^-25.1, error 2^-76.5;
  the low-part sum (four roundings of a value below 2^-25) 2^-76;
  truncation after s⁹/9!, 2^-96.8; AddF, Mul by the table, table: below
  2^-100. Sum 2^-75.2; used: **2^-74**.
- `stirlingSeries(y)`, y >= 34, against the 9-term series S9: the sum
  c1 + z·(...) rounded (u/12), c1 = RN(1/12) (u/16), z = 1/(y·y) rounded
  twice (2u·3·10^-5/12), the division by y (u·S): |ΔS| <= 0.25u/y = 2^-55/y,
  fused or not. S9 against log Γ: the first omitted term, |B20|/(380 y^19)
  <= 2^-96.2 (DLMF 5.11.ii: the error of the truncated series is bounded by
  the first omitted term for real y > 0).

Reference, |L_D - L₀|: lq: Div and MulF 19u² + Log 2^-75 + AddF 64u² +
(q.Lo/q.Hi)² terms 1.5u², about 2^-74.9; lg: (y - 1/2)·2^-75 + 2^-55/y +
Mul/AddF/Add below 2^-93; the final Sub below 2^-94.7.

Fast side, with the documented kernel bounds of kernels.go (logHL 2^-68
absolute, expHL 2^-68 relative):

- log|x·s|: x·s.Hi exact (TwoProd) + x·s.Lo, 3u²; logHL 2^-68;
  log(1 + dl/dh) = dl/dh to 4u².
- `logGammaLargeHL(y)`: (y - 1/2)·log y with log y from the log kernel,
  (y - 1/2)·2^-68; mh·ll rounded with |ll| <= 2^-17.79 (|q| <= r²/2,
  |r| <= 2^-8.4) so |pl| <= 2^-10.19, two roundings 2^-62.2; the final sum
  of low parts (three roundings of a value below 2^-10.19) 2^-61.6;
  S as sh + sl, 2^-73.4. Sum: (y - 1/2)·2^-68 + 2^-60.87.
- L = log π - log|x·s| - log Γ(y): two TwoSums exact; the low parts
  (below 2^-10.18) summed with four roundings, 2^-61.1; the Fast2Sum
  normalisation exact.
- e^L: expHL, 2^-68 relative (|L| <= 745, |l| <= u|h| after normalisation).

Sum: |L_fast - L_D| <= y·(2^-68 + 2^-75) + 2^-55/y + 2^-59.94 + 2^-74.9.
On [34, 190] the first two terms are convex in y and largest at y = 34,
2^-59.91; the total is 2^-58.92, and with both exponentials
|h + l - D|/|D| <= 2^-58.91. **E = 2^-58.5** (margin 1.33x).

Range: L < -746 gives |D| < 2^-1076, and the reference's math.Ldexp returns
a signed zero, which the fast path returns. -746 <= L < -745 (outside the
exp kernel's range) goes to the reference. For L >= -745 the decision is
made on the mantissa (eh, el, k) of the exp kernel: RN(D·2^-k) = eh. For
k >= -1021 the result eh·2^k is exact. Below, the reference returns
math.Ldexp(RN(M), k') with RN(M) = eh·2^(k-k') (the scaling between the two
kernels' mantissas is an exact power of two); math.Ldexp depends only on the
normalised fraction and the total exponent, so math.Ldexp(eh, k) is the
same float64, subnormal or zero included.

## `mod2`

`sinPiDD`/`cosPiDD` reduced x with `math.Mod(|x|, 2)`, which loops over the
exponent difference and took 40 % of region C. `mod2(a) = a - 2⌊a/2⌋` is
exact (proof in poly.go) and therefore returns the same value; the test
compares it with math.Mod on 2·10^6 random arguments, around every power of
two, and under fuzzing. Every caller of sinPi/cosPi is unchanged in value.

## Witness

`TestGammaFastEqualsReference` (gamma_fast_test.go) compares fast with
reference, and `Gamma` with the pre-change function, bit for bit, on
10.9·10^6 points (uniform, log-uniform down to 2^-56, and within 512 ulps
or a power of two of every pole), and logs the fallback rate;
`TestGammaReflSubnormal` covers subnormal and zero results; `FuzzGammaFast`
and `FuzzMod2` replay their committed corpus (testdata/fuzz).

Fallback rate measured (darwin/arm64, 2026-11-09):

| region | points | fallbacks |
|---|--:|--:|
| 2^-56 <= x < 1 | 2 000 000 | 0 |
| -1/2 < x <= -2^-56 | 2 000 000 | 0 |
| -34 < x <= -1/2 | 2 977 145 | 0 |
| -171.6 < x <= -34 | 2 956 300 | 3.21 % |
| -190 <= x <= -171.6 | 983 202 | 2.39 % |

The rate of region C is the predicted one: P(fallback) = 2E·eh/ulp(eh),
2^-5.5·eh for eh in [1, 2), about 3.3 % on average.

## Binom (beta branch), v0.6.1

`Binom(n, k)` for n+1 > 0, n-k+1 > 0, k > -1 evaluates
exp(-(log(n+1) + log B(a, b) + corrections)) in double-double
(`binomBetaDD`). `binomFast` forms the same sum as hi + lo from the float64
kernels and returns it when `zivRound` certifies it, `binomErr` = 2^-58
relative; otherwise the reference runs, so every output is the reference's
bits.

Covered: p = min(a, b), q = max(a, b) with 1/2 <= p < 2 and 10 <= q <= 2^40,
or q < 10 and p + q < 10 (the refbench row is all of it). The pieces are
those of the reference where it rounds to float64: log Γ from the Γ(1+t) core
`gamma1pFast` (the same polynomial value as `gamma1pm1`), the Stirling
difference S(q) - S(p+q) as the float64 `stirlingSeries(q) - stirlingSeries(s)`
that `logGammaDiff` forms (it carries a 2^-60 error near q = 10 that the fast
path must reproduce, not improve on: using the more accurate value from
`logGammaDiffHL` gave a worst distance of 2^-58), and the first-order
corrections for the rounding of a and b from the same `Digamma` calls.

Evidence (`binom_fast_test.go`): 2·10^7 points over the covered region,
0 mismatches with the reference, worst |h + l - D|/D = 2^-60.62 against
the 2^-58 bound (margin 5), 4.5 % of the points fall back (the suite runs a
tenth of that sweep: it is near the CI timeout under coverage); 10^5 wide
random points and the refbench grid through the public function, equal to
the reference.

## Factorial 0 < n < 10, bound for 10 <= n < 30, Xlog1py order

Factorial now runs the float64 path for 0 < n < 1 (`gamma1pFast`, Ziv bound
2^-99, measured worst 2^-105.8) and 1 < n < 10 (`logGammaMidHL` of Binom,
bound 2^-63, measured worst 2^-68.8), and shares the logarithm between
log Γ(n) and log n for n >= 10 (`logGammaLargeHLL`, bit for bit the same
values). The sweep (2·10^7 points, `SPECIAL_SWEEP`) showed that the earlier
2^-59 bound of n >= 10 was exceeded just above 10 (2^-58.79 on [10, 10.6]);
it is now 2^-57.5 on [10, 12), 2^-58.5 on [12, 30), 2^-59 above. Fallback
about 1.2 %. `zivRound` was trimmed to inline; Xlog1py tries its fast path
first. Outputs equal the double-double reference's. Against v0.10.0 they
differ on [10, 10.67]: its 2^-59 bound accepted about 6.6·10^-6 of the points
there (26 of 3.9·10^6) with a value one ulp from the reference's, e.g.
Factorial(10.43482661274922) was 0x41636a7d1b271eba and is now ...eb9 (#67);
all other results are unchanged.
