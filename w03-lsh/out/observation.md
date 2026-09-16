# Week 3 · Observations

## Task 1 · Minhash and LSH from the matrix up

**One pass over rows, not one per column.** The signature matrix is
`columns × hashes`, a size that does not depend on how many rows there are, so
a row can be read, used to update every column holding a 1 in it, and then
thrown away forever. Looping per column instead re-reads the whole matrix once
per document — correct in memory, and `n_cols` full disk scans when the matrix
does not fit, which is the case this course is about. Walking rows also lets
`h_k(r)` be computed once per row rather than once per (row, column) pair.
My input arrives column-oriented, so `row_stream()` regroups it by row once,
touching every 1 exactly once, and `minhash_signatures` then consumes that
stream in a single forward sweep; in a real pipeline the stream is just the
file and the adapter disappears.

**R5 — leftover rows are kept, not dropped.** When `bands` does not divide the
signature length, `band_bounds()` gives the first `n % bands` bands one extra
row each, so band widths differ by at most 1 and no hash is discarded.
Truncating instead would silently throw away evidence and move the S-curve
without saying so — the discarded rows would still cost you to compute, and the
effective `r` would no longer be the `r` you did the arithmetic with.

**The 1.0-vs-2/3 gap is sampling error, not a bug.** With two hashes the
estimator can only return 0, 0.5 or 1, and S1/S4 at true similarity 2/3 landed
on 1.0 by agreeing in both positions — probability (2/3)² = 44%. The estimate is
a mean of `k` Bernoulli(s) draws, so its standard error is `sqrt(s(1-s)/k)`:
0.33 at k=2, 0.047 at k=100. **Narrowing it means more hashes, and the exchange
rate is bad**: accuracy improves as `1/sqrt(k)` while cost — signature storage,
hash evaluations per row, band construction — grows as `k`. Buying 10× accuracy
costs 100× the hashing. That asymmetry is why you do not fix this by brute-force
accuracy; you use §3.4 to avoid needing precision on pairs you were never going
to compare.

## Task 2 · The crossover on this machine

**Crossover n ≈ 570** (brute force wins at n=500 by 0.46 s; LSH wins at n=600 by
0.10 s; interpolating the difference gives 582, solving the fitted curves gives
533). Machine: **Intel Core i7-9750H @ 2.60 GHz, 6c/12t, 16 GB RAM, macOS
26.6.2, CPython 3.9.6, single-threaded pure Python, no numpy**, editor and a
terminal open but nothing CPU-heavy. LSH loses below that because it pays a
**1.2 s fixed cost** — evaluating 100 hash functions over the ~5,000-shingle row
space, which is nearly full-length even at n=125 — before it makes a single
comparison. At n=125 it made **zero** comparisons and still took 1.34 s, while
brute force did all 7,750 in 0.08 s.

**The quadratic check (A4) held, over 64× of range.** Six consecutive doublings
gave ×4.02, ×4.28, ×4.26, ×4.25, ×3.88, ×4.03 (mean ×4.12), and `t/n²` stayed
flat at 5.3–6.5 µs from n=125 to n=8,000. Fitting `t = 6.38e-6·n²` predicts
408.3 s at n=8,000 against a measured 408.35 s. LSH over the same doublings gave
×1.24 → ×1.85, i.e. linear: `t = 1.20 + 0.00115·n` predicts 10.40 s at n=8,000,
measured 10.40 s.

**It became unpleasant at n = 8,000: brute force took 408 s (6.8 minutes) versus
10.4 s for LSH, a 39× gap.** What ran out was **time, not memory, and not
close** — peak traced allocation was 87.6 KB for brute force and 29.8 MB for
LSH (plus a 30.7 MB corpus) on a 16 GB machine, so memory sat at ~0.2% while one
core stayed pinned for seven minutes. I did not run n=16,000: the fit that just
predicted n=8,000 to within 0.02% predicts 1,633 s ≈ 27 minutes for one point,
and confirming a seventh doubling is not worth half an hour. That is the real
failure mode of quadratic work — it does not crash, it just makes the wait
quadruple while the data merely doubles.

## Task 3 · Comparing far less

**n = 100 hashes, b = 25 bands, r = 4 rows per band → step at
`(1/25)^(1/4) = 0.447`.** The threshold is 0.6, and I put the step deliberately
*below* it. The planted near-duplicates change 4–14 of 60 shingles, so the
worst of them sits at about 46/74 = 0.62 — right on top of the threshold, and
a step placed *at* 0.6 catches such pairs only ~50% of the time by definition
(the step is where P = 0.5). At step 0.447 the arithmetic
`P(s) = 1 - (1 - s^r)^b` gives **P(0.62) = 98.2%** and **P(0.875) = 100.0%**,
while unrelated documents (60 shingles drawn from a 5,000 vocabulary, s ≈ 0.006)
give **P = 3.2e-8** — under 0.1 expected spurious candidates across 2.2 M pairs.
A low step normally costs false positives; here it is nearly free because the
corpus has no middle ground between 0.62 and 0.006. Result:
**126 comparisons instead of 2,246,140 — 99.99% avoided, recall 100.0%,
precision 100.0% (`out/bench.txt`, grade: strong)**, i.e. 126 comparisons to
find 121 pairs.

**Moving the step the wrong way, measured.** Raising `r` pushes the step above
the threshold and recall collapses exactly as the formula says:

| n | b | r | step | predicted P(0.62) | comparisons | **recall** |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 25 | 4 | 0.447 | 98.2% | 126 | **100.0%** |
| 120 | 20 | 6 | 0.607 | 68.9% | 112 | **92.6%** |
| 100 | 10 | 10 | 0.794 | 8.1% | 58 | **47.9%** |
| 100 | 5 | 20 | 0.923 | 0.04% | 9 | **7.4%** |

Note the trap: every bad configuration made *fewer* comparisons, so on the
"comparisons avoided" number alone they all look better. b=10 avoids 99.997%
and loses half the real pairs. Recall is the first number to read, not the
second. (Measured recall sits above P(0.62) because most planted pairs are well
above 0.62; 0.62 is the worst case, not the typical one.)

**Is "hashing is free" fair at scale?** It is fair in shape, not in constants.
Signatures and banding cost `O(n · k)`; comparisons cost `O(candidates)`, which
is `O(n²)` for brute force and near-linear for LSH — so as n grows the charged
term dominates and the accounting becomes *more* accurate, not less. It stops
being fair in two places. First, at small n it is already unfair in the other
direction: at n=125 my finder spent 1.34 s and made **zero** charged
comparisons, so the harness would score it a perfect zero while it ran 17×
slower than the brute force it beat on paper — the uncharged work was 100% of
the runtime. Second, at genuinely large n the hashing stops being a cheap linear
term and becomes an I/O and shuffle problem: 100 hashes × 3 M documents is 300 M
signature entries that will not sit in one machine's memory, banding becomes a
distributed shuffle by band key, and skewed buckets (one popular band value
pulling in a huge candidate list) can make the "free" step cost more wall-clock
than the comparisons it saved. The honest version of the assumption is that
hashing is linear and comparison is quadratic; below the crossover and above the
single-machine limit, linear-and-cheap stops being the same statement.
