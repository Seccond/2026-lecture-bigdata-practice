# Task 2 · The crossover on this machine

Raw numbers: `out/crossover.json` (10 sizes, 125 → 8,000, a 64× span).
Produced by `python3 task2_crossover.py --sizes ...` in three invocations
(`125,250,500,1000,2000`, then `600,700,800,4000`, then `8000` alone).

## A6 · The machine

| | |
|---|---|
| CPU | Intel Core i7-9750H @ 2.60 GHz, 6 cores / 12 threads |
| RAM | 16 GB (17,179,869,184 bytes) |
| OS | macOS 26.6.2 (build 25G83), x86_64 |
| Python | 3.9.6, CPython, stock interpreter, **no numpy** — everything below is pure Python |
| Recorded by the script | `macOS-26.6.2-x86_64-i386-64bit` / `i386` / `3.9.6` (`platform.processor()` reports `i386` on macOS; the CPU line above is from `sysctl machdep.cpu.brand_string`) |
| Also running | editor and a terminal, nothing CPU-heavy. The n=8,000 run was done on its own with nothing else started. |

Single-threaded throughout — both methods are plain Python loops, so the other
11 hardware threads do nothing and the 6-core count buys nothing here.

### One change to the harness, stated up front

`task2_crossover.py` originally took `bench.build()[:n]`, but `bench.build()`
always returns exactly `N_DOCS + PLANTED = 2,120` documents. Every size above
2,120 would therefore have silently re-measured 2,120 documents and the curve
would have gone flat. I replaced that with `corpus(bench, n)`, which uses the
harness corpus verbatim at or below 2,120 and above it rebuilds with the same
`VOCAB` (5,000), the same `SHINGLES` (60) and the same rate of planted
near-duplicates (120 per 2,000). `bench.py` itself is untouched.

## A3 · Time against n

| n | brute force | LSH (n=100 hashes, b=25 bands) | brute comparisons | LSH comparisons |
|---:|---:|---:|---:|---:|
| 125 | 0.08 s | 1.34 s | 7,750 | 0 |
| 250 | 0.34 s | 1.66 s | 31,125 | 1 |
| 500 | 1.44 s | 1.90 s | 124,750 | 7 |
| **600** | **2.20 s** | **2.10 s** | 179,700 | 11 |
| 700 | 3.13 s | 2.27 s | 244,650 | 14 |
| 800 | 3.80 s | 2.21 s | 319,600 | 17 |
| 1,000 | 6.14 s | 2.46 s | 499,500 | 28 |
| 2,000 | 26.10 s | 4.10 s | 1,999,000 | 110 |
| 4,000 | 101.34 s | 5.63 s | 7,998,000 | 223 |
| 8,000 | **408.35 s** | 10.40 s | 31,996,000 | 478 |

```
time (s), log scale
 500 |                                                        B
     |                                       B
 100 |
     |                        B
  10 |            B                    L            L         L
     |      B  L        L        L
   1 |   B     L
     |B  L
 0.1 +---------------------------------------------------------
     125  250 500  1k      2k       4k            8k        n
     B = brute force   L = LSH        (crossover between 500 and 600)
```

Brute force spans 0.08 s → 408 s, a factor of **5,100**, while n grows 64×.
LSH spans 1.34 s → 10.40 s, a factor of **7.8** over the same 64×.

## A4 · Is brute force actually quadratic?

Two checks, both against the measured numbers rather than against the claim.

**Doubling test** — n doubles, time should ×4:

| doubling | ratio |
|---|---:|
| 125 → 250 | ×4.02 |
| 250 → 500 | ×4.28 |
| 500 → 1,000 | ×4.26 |
| 1,000 → 2,000 | ×4.25 |
| 2,000 → 4,000 | ×3.88 |
| 4,000 → 8,000 | ×4.03 |

Six doublings, all between ×3.88 and ×4.28, mean ×4.12. **It holds.**

**Constant test** — if `t = c·n²` then `t/n²` should be flat:

| n | 125 | 250 | 500 | 1,000 | 2,000 | 4,000 | 8,000 |
|---|---|---|---|---|---|---|---|
| t/n² (µs) | 5.35 | 5.37 | 5.74 | 6.14 | 6.52 | 6.33 | 6.38 |

Flat to within ±10% across a 64× range of n. The slow drift upward from 5.3 to
6.4 µs between n=125 and n=1,000 is real and is not a departure from n²: the
per-comparison cost itself creeps up as the working set (8,000 shingle sets,
~31 MB) stops fitting in cache, and `tracemalloc` is instrumenting every
allocation the whole time. Above n=2,000 the constant settles at ≈6.4 µs and
stays there. Fitting `t = 6.38e-6 · n²` predicts 408.3 s at n=8,000 against a
measured 408.35 s.

The comparison counts are exactly `n(n-1)/2` at every size, as they must be,
which confirms the count is quadratic independently of the clock.

**LSH, for contrast** — the same doublings give ×1.24, ×1.14, ×1.67, ×1.37,
×1.85, i.e. approaching ×2, not ×4. A straight-line fit `t = 1.20 + 0.00115·n`
predicts 10.40 s at n=8,000 against a measured 10.40 s. Linear with a
**1.2 second fixed cost**, and that fixed cost is the whole story of A8.

## A5 · Peak memory at the largest n (8,000)

| | peak (tracemalloc) |
|---|---:|
| brute force | **87.6 KB** |
| LSH | **29.8 MB** |
| the corpus itself (allocated before the timer starts, so counted by neither) | 30.7 MB |

`tracemalloc.start()` is called inside `timed()`, after the documents already
exist, so these are the *working sets of the algorithms*, not the data.

Brute force is essentially free on memory: it holds two transient sets per
comparison (`a | b`, `a & b`) and throws them away, so its peak is just the
result set of found pairs — 87.6 KB at n=8,000, up from 7.5 KB at n=125 only
because there are more planted pairs to store.

LSH pays 29.8 MB, which is roughly the size of the corpus again. That is the
signature matrix: 8,000 documents × 100 hashes = 800,000 Python ints in 8,000
lists, plus the row-to-document index and the band buckets. It scales linearly
(1.25 MB at n=125, 4.86 MB at n=1,000, 15.66 MB at n=4,000, 29.8 MB at
n=8,000 — the marginal cost settles at about 3.7 KB per document, on top of a
fixed ~0.6 MB for the row index over the 5,000-shingle vocabulary, which is why
the per-document figure looks inflated at the small sizes).

So the two methods trade the expensive resource: **brute force spends time and
almost no memory; LSH spends memory to buy back the time.** At this scale that
trade is a bargain — 30 MB against 16 GB is nothing, and it bought a 39×
speedup.

## A2 · Where it became unpleasant

**n = 8,000, brute force, 408 seconds — 6.8 minutes of waiting.** That is the
recorded unpleasant point. It is well past "a minute of waiting"; the terminal
sat there long enough that I checked twice whether it had hung.

**What ran out first: time, not memory, and not close.** At n=8,000 the whole
process peaked around 31 MB of traced allocation on a machine with 16 GB. There
was never any memory pressure, no swapping, no paging. Memory was ~0.2% used
while the CPU had been pinned at 100% of one core for seven minutes.

**n=16,000 was not run, deliberately.** The quadratic fit that just predicted
n=8,000 to within 0.02% predicts `6.38e-6 × 16000² = 1,633 s`, i.e. **27
minutes** for one brute-force point. Extrapolating a curve that has held over
six consecutive doublings is better evidence than spending half an hour to
confirm it a seventh time. Memory would still have been fine there (~60 MB
corpus + working set); it is the clock that makes the size unreachable, and
that is the actual shape of the problem — quadratic algorithms do not fail by
crashing, they fail by making you wait, and the waiting quadruples while your
data merely doubles.

For scale: at 3 million documents the same constant gives 4.5 × 10¹² comparisons
× 6.4 µs ≈ 2.9 × 10⁷ seconds ≈ **333 days** on one core.

## A7 · The crossover

**Between n = 500 and n = 600; approximately n ≈ 570.**

| n | brute | LSH | winner |
|---:|---:|---:|---|
| 500 | 1.44 s | 1.90 s | brute force, by 0.46 s |
| **600** | **2.20 s** | **2.10 s** | **LSH, by 0.10 s** — first size where LSH wins |
| 700 | 3.13 s | 2.27 s | LSH, by 0.86 s |
| 800 | 3.80 s | 2.21 s | LSH, by 1.59 s |

Linear interpolation on the difference (−0.46 s at 500, +0.10 s at 600) puts
the crossing at **n ≈ 582**. Solving the two fitted curves,
`6.38e-6·n² = 1.20 + 0.00115·n`, gives **n ≈ 533**. Both land in the same place:
somewhere around 550.

After that the gap opens fast, because one side is linear and the other is not:

| n | speedup of LSH over brute |
|---:|---:|
| 600 | 1.05× |
| 1,000 | 2.5× |
| 2,000 | 6.4× |
| 4,000 | 18× |
| 8,000 | **39×** |

## A8 · Why LSH loses at small n

It loses because it has already paid before it compares anything, and at small
n there is nothing to compare anyway.

Look at the flat part of the LSH curve: **1.34 s at n=125, when it made zero
comparisons.** Brute force at n=125 did 7,750 real comparisons in 0.08 s — one
sixteenth of the time — and got the right answer. LSH's 1.34 s bought nothing
at all at that size.

That 1.2 s intercept is three specific things, none of which shrink with n:

1. **Hashing the row space.** `minhash_signatures` walks every row of the
   characteristic matrix and evaluates all 100 hash functions on it. The row
   space is the *vocabulary*, not the document count — `VOCAB = 5,000` — and
   125 documents × 60 shingles already touch about 4,000 distinct shingles. So
   the row loop is nearly full length at n=125 and barely longer at n=8,000.
   That is ~500,000 Python-level lambda calls before a single pair is looked at.
2. **Building the signature matrix.** 100 min-updates for every 1 in the
   matrix. This part *is* linear in n (60 ones per document), and it is what
   makes the line slope upward at 0.00115 s/doc.
3. **Banding and bucketing.** 25 dictionary insertions per document, plus
   tuple construction for each band key.

None of that touches the scoring function, which is exactly why the harness in
Task 3 charges zero for it — and at n=125 that accounting is a fiction, because
the un-charged work is the *entire* runtime.

The real point is the shape, not the constant. Brute force pays `6.4 µs × n²/2`
and nothing else. LSH pays `1.2 s + 1.15 ms × n` and then a handful of real
comparisons. Below n≈550 the fixed setup is larger than the whole quadratic bill,
so you have bought machinery for a job you could have done by hand. Above it,
the quadratic term runs away from the linear one and never comes back: at
n=8,000 LSH's fixed 1.2 s is 0.3% of what brute force spends, and at n=3,000,000
it would be invisible against 333 days.

Tuning the setup cost would move the crossover but not the conclusion. Halving
the hash count to 50 would roughly halve the 1.2 s intercept and pull the
crossover down to around n≈400, at the cost of a coarser S-curve. The crossover
is a property of the constants; the reason to use LSH at all is a property of
the exponents.
