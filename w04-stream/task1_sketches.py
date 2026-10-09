#!/usr/bin/env python3
"""Week 4 · Task 1 — Answer questions about a stream you cannot store.

Textbook §4.3 (sampling), §4.4 (Bloom filter), §4.5 (Flajolet-Martin).

The premise of the whole chapter: the stream is longer than your memory, it
goes past once, and you still have to answer. Every method here trades an exact
answer for a bounded amount of space, and the job is to know exactly what you
traded.

You build three, and the harness checks each against the truth it is
approximating.

    python3 task1_sketches.py --verify
"""
import argparse, hashlib, math, random, statistics


def _base_hash(item, seed, size=16):
    """A keyed blake2b digest of the item, as one integer."""
    d = hashlib.blake2b(str(item).encode(), digest_size=size,
                        key=str(seed).encode()).digest()
    return int.from_bytes(d, "big")


class BloomFilter:
    """Membership, with one-sided error.

    A Bloom filter never says "no" about something you inserted. It sometimes
    says "yes" about something you did not. That asymmetry is the entire design
    and it is why it is useful for "have I seen this before" and useless for
    "is this definitely in the set".

    `m` bits, `k` hash functions.
    """

    def __init__(self, m, k, seed=246):
        self.m = m
        self.k = k
        self.seed = seed
        self.bits = bytearray((m + 7) // 8)     # m bits, packed 8 to a byte

    def _positions(self, item):
        """k bit positions for one item.

        One 128-bit digest split into two halves h1, h2, then g_i = h1 + i*h2
        (Kirsch-Mitzenmacher double hashing). Behaves like k independent hashes
        for the purposes of the §4.4.2 analysis, at the cost of one digest.
        """
        h = _base_hash(item, self.seed)
        h1, h2 = h >> 64, (h & 0xFFFFFFFFFFFFFFFF) | 1
        return [(h1 + i * h2) % self.m for i in range(self.k)]

    def add(self, item):
        for p in self._positions(item):
            self.bits[p >> 3] |= 1 << (p & 7)

    def __contains__(self, item):
        # add() set every one of these bits and nothing ever clears a bit,
        # so an inserted item always finds all k of them set: no false negatives.
        return all(self.bits[p >> 3] & (1 << (p & 7))
                   for p in self._positions(item))

    def expected_fp_rate(self, n_inserted):
        """The textbook's predicted false-positive rate after n insertions.

        §4.4.2 derives it. Return the number, do not measure it - the harness
        measures separately and compares the two.
        """
        # A given bit stays 0 after k*n darts with probability e^(-kn/m).
        # A false positive needs all k of an absent item's bits to be 1.
        return (1 - math.exp(-self.k * n_inserted / self.m)) ** self.k


_P61 = (1 << 61) - 1       # Mersenne prime for h(x) = (a*x + b) mod p


def _trailing_zeros(v):
    return (v & -v).bit_length() - 1 if v else 0


def _combine(rs, how, group_size=8):
    """Turn the per-hash maxima R into one estimate."""
    ests = [2.0 ** r for r in rs]
    if how == "mean":
        return statistics.mean(ests)
    if how == "median":
        return statistics.median(ests)
    # §4.5.3: average within small groups, then take the median of the averages.
    groups = [ests[i:i + group_size] for i in range(0, len(ests), group_size)]
    return statistics.median(statistics.mean(g) for g in groups)


def flajolet_martin(stream, n_hashes=64, seed=246, combine="median"):
    """Estimate how many DISTINCT items went past, in almost no memory.

    §4.5. Hash each item, count trailing zeros in the hash, keep the maximum.
    A maximum of R suggests about 2^R distinct items, because seeing R trailing
    zeros is a 1-in-2^R event.

    One hash gives an estimate with enormous variance, so you use many and
    combine them. How you combine them matters a great deal:

      * averaging 2^R directly is dominated by whichever hash got lucky - the
        values are exponential, so one outlier swamps the rest
      * the median is robust but can only ever be a power of two
      * §4.5.3 suggests grouping, and combining twice

    The harness accepts anything **within a factor of two** of the truth. That is
    not a generous tolerance, it is an honest one: this method really is that
    crude, and HyperLogLog exists because of it. Getting inside a factor of two
    reliably is the requirement; getting closer than that is not expected here.

    Return your estimate as a float.

    `combine` is "median" (the default), "mean" or "group" (§4.5.3), so the
    three rules can be compared on the same stream.

    Why the median: for one hash, P(R <= r) = exp(-n / 2^(r+1)), so the median
    of R sits where 2^r first reaches n/(2 ln 2) = 0.72n. With enough hashes the
    median estimate therefore lands in [0.72n, 1.44n), inside the factor of two.
    Anything that averages 2^R inherits its long tail: P(R = r) ~ n / 2^(r+1),
    so every r above log2(n) adds about n/2 to E[2^R], bounded only by the
    hash width. "mean" and "group" both overestimate for that reason.
    """
    rng = random.Random(seed)
    coeffs = [(rng.randrange(1, _P61), rng.randrange(_P61)) for _ in range(n_hashes)]

    # The only state: one running maximum per hash. Its size is set by
    # n_hashes, not by the length of the stream or the number of distinct items.
    rs = [0] * n_hashes
    for item in stream:
        x = _base_hash(item, seed, size=8)
        for i, (a, b) in enumerate(coeffs):
            r = _trailing_zeros((a * x + b) % _P61)
            if r > rs[i]:
                rs[i] = r
    return float(_combine(rs, combine))


def reservoir_sample(stream, k, seed=246):
    """Keep k items uniformly at random from a stream of unknown length.

    §4.3. Every item that went past must end up with the same probability k/n
    of being in your sample, and you only ever hold k of them.

    Return a list of k items (or fewer if the stream was shorter).
    """
    rng = random.Random(seed)
    sample = []
    for i, item in enumerate(stream):
        if i < k:
            sample.append(item)
            continue
        # This is where the unknown length is handled: the (i+1)-th item gets
        # in with probability k/(i+1), using only the count seen so far. Each
        # earlier item survives this step with probability i/(i+1), which keeps
        # every item at exactly k/n whenever the stream happens to end.
        j = rng.randrange(i + 1)
        if j < k:
            sample[j] = item
    return sample


# ------------------------------------------------------------------- harness
def verify():
    fails = 0
    rng = random.Random(246)

    def check(label, ok, detail=""):
        nonlocal fails
        print(f"  {'ok  ' if ok else 'FAIL'}  {label:<46} {detail}")
        fails += not ok

    # --- Bloom: no false negatives, ever
    try:
        bf = BloomFilter(m=8192, k=5)
    except NotImplementedError:
        print("  BloomFilter is still a stub"); return 1
    inserted = [f"item-{i}" for i in range(800)]
    for x in inserted:
        bf.add(x)
    check("no false negatives", all(x in bf for x in inserted))

    absent = [f"other-{i}" for i in range(20_000)]
    fp = sum(1 for x in absent if x in bf) / len(absent)
    predicted = bf.expected_fp_rate(len(inserted))
    close = abs(fp - predicted) < max(0.02, predicted * 0.5)
    check("measured false-positive rate matches theory", close,
          f"measured {fp:.3%}, predicted {predicted:.3%}")

    # --- Flajolet-Martin: a factor of two is what this method gives you
    try:
        distinct = 20_000
        stream = [f"k{rng.randrange(distinct)}" for _ in range(120_000)]
        est = flajolet_martin(stream)
    except NotImplementedError:
        print("  flajolet_martin is still a stub"); return 1
    true_distinct = len(set(stream))
    ratio = est / true_distinct
    check("distinct estimate within a factor of 2", 0.5 <= ratio <= 2.0,
          f"estimated {est:,.0f}, true {true_distinct:,} ({ratio:.2f}x)")

    # --- Reservoir: uniform over many trials
    try:
        counts = [0] * 20
        trials = 4000
        for t in range(trials):
            s = reservoir_sample(range(20), 5, seed=t)
            for i in s:
                counts[i] += 1
    except NotImplementedError:
        print("  reservoir_sample is still a stub"); return 1
    expected = trials * 5 / 20
    spread = (max(counts) - min(counts)) / expected
    check("reservoir is uniform across items", spread < 0.15,
          f"spread {spread:.1%} around {expected:.0f}")

    print(f"\n  {'all ok' if not fails else str(fails) + ' failed'}")
    return 1 if fails else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true")
    a = p.parse_args()
    raise SystemExit(verify() if a.verify else p.print_help())
