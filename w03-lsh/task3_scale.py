#!/usr/bin/env python3
"""Week 3 · Task 3 — Find the same pairs without comparing everything.

Textbook §3.4.

`BruteForce` compares every pair. On 3,000 documents that is 4.5 million
comparisons and it is completely correct. On 3 million documents it is 4.5
trillion and it is completely useless.

Beat it. Find the same near-duplicate pairs while making far fewer comparisons.

    python3 bench.py
    python3 bench.py --yours

The harness counts every call you make to `similarity()`. That is your score.
It also checks **recall** - which of the truly similar pairs you found. Skipping
comparisons is easy; skipping comparisons without losing the pairs is the task.
"""
import random

from task1_minhash import lsh_candidates, minhash_signatures


class BruteForce:
    """Correct, and quadratic."""

    def __init__(self, threshold):
        self.threshold = threshold

    def find(self, docs, similarity):
        """docs is [set_of_shingles, ...]. Return {(i, j), ...} with i < j."""
        out = set()
        for i in range(len(docs)):
            for j in range(i + 1, len(docs)):
                if similarity(docs[i], docs[j]) >= self.threshold:
                    out.add((i, j))
        return out


class YourFinder:
    """Your near-duplicate finder.

        __init__(threshold)
        find(docs, similarity) -> {(i, j), ...}

    `similarity(a, b)` is the only way to compare two documents, and every call
    is counted. Everything else - signatures, banding, bucketing - is free, in
    the sense that the harness does not charge you for it. That is deliberate:
    it is also roughly true at scale, where the comparison is the expensive
    part and the hashing is linear.

    Two knobs decide everything:

        the number of hashes in a signature
        how many bands you split it into

    §3.4.2 gives you the relationship between those and the probability that a
    pair at similarity s becomes a candidate. It is an S-curve, and where its
    step sits is something you choose. Choose it on purpose and be able to say
    why in observation.md - a threshold of 0.8 does not mean bands should be
    anything in particular until you have done the arithmetic.

    You may reuse your Task 1 code.

    --------------------------------------------------------------------------
    The arithmetic (§3.4.2), for threshold 0.6
    --------------------------------------------------------------------------
    A pair at similarity s survives banding with probability

        P(s) = 1 - (1 - s^r)^b        and the step sits near (1/b)^(1/r)

    Chosen: n = 100 hashes, b = 25 bands, r = 4 rows per band.

        step = (1/25)^(1/4) = 0.447

    The step is put *below* the 0.6 threshold on purpose. The planted
    near-duplicates are clones with 4-14 of 60 shingles changed, so the worst
    of them sits at about 46/74 = 0.62 - right on top of the threshold. A step
    at 0.6 would cut those in half; a step at 0.447 gives them

        P(0.62) = 1 - (1 - 0.62^4)^25 = 98.2%
        P(0.875) = 1 - (1 - 0.875^4)^25 = 100.0%

    The price of a low step is false candidates, and here it is nearly free:
    unrelated documents are 60 shingles drawn from 5,000, so s is about 0.006,
    and P(0.006) = 3e-8 per pair - a fraction of one spurious comparison over
    2.2 million pairs. Cheap because the corpus has no middle ground.
    """

    N_HASHES = 100
    BANDS = 25                 # r = 4 -> step (1/25)^(1/4) = 0.447

    # A prime comfortably above any vocabulary we will see, for h(r) = (a*r+b) % P.
    _PRIME = 1_000_003

    def __init__(self, threshold):
        self.threshold = threshold

    def _hashes(self):
        """N_HASHES independent-enough permutations of the row space."""
        rng = random.Random(20260916)
        p = self._PRIME
        fns = []
        for _ in range(self.N_HASHES):
            a = rng.randrange(1, p)
            b = rng.randrange(0, p)
            fns.append(lambda r, a=a, b=b, p=p: (a * r + b) % p)
        return fns

    def find(self, docs, similarity):
        if len(docs) < 2:
            return set()

        # Shingles are arbitrary objects; minhashing needs row *numbers*.
        # One pass over the documents assigns each distinct shingle a row.
        row_of, columns = {}, []
        for d in docs:
            col = set()
            for s in d:
                r = row_of.get(s)
                if r is None:
                    r = len(row_of)
                    row_of[s] = r
                col.add(r)
            columns.append(col)

        sigs = minhash_signatures(columns, self._hashes(), len(row_of))

        # Only the pairs that collided in at least one band get compared,
        # and only those calls are charged.
        out = set()
        for i, j in lsh_candidates(sigs, self.BANDS):
            if similarity(docs[i], docs[j]) >= self.threshold:
                out.add((i, j))
        return out
