#!/usr/bin/env python3
"""Week 4 · Task 3 — Same memory, fewer mistakes.

Textbook §4.4 (Bloom filters), §4.5 (counting distinct).

`NaiveFilter` is a membership filter in a fixed number of bits. It works. It
also makes far more mistakes than it has to with the memory it was given, and
it does so for a reason you can find by reading §4.4.2 and doing one derivative.

You get **exactly the same number of bits**. Make fewer mistakes.

    python3 bench.py
    python3 bench.py --yours

The rule that makes this interesting: a false negative is not allowed. Ever.
The whole point of this structure is that "no" means no. A filter that gets a
better score by occasionally forgetting something it was given has not improved
anything, it has broken the contract.
"""
import hashlib, math


class NaiveFilter:
    """One hash function, and the bits it was given."""

    def __init__(self, n_bits, seed=246):
        self.n_bits = n_bits
        self.seed = seed
        self.bits = bytearray(n_bits)

    def _index(self, item):
        d = hashlib.blake2b(str(item).encode(), digest_size=8,
                            key=str(self.seed).encode()).digest()
        return int.from_bytes(d, "big") % self.n_bits

    def add(self, item):
        self.bits[self._index(item)] = 1

    def __contains__(self, item):
        return bool(self.bits[self._index(item)])

    def memory_bits(self):
        return self.n_bits


class YourFilter:
    """Your filter.

        __init__(n_bits, seed=246)
        add(item)
        item in filter  ->  bool
        memory_bits()   ->  how many bits you are using

    `memory_bits()` must not exceed the `n_bits` you were given. The harness
    checks. Counting only some of your memory is not an optimisation.

    §4.4.2 gives the false-positive rate of a filter with m bits, k hashes and
    n items inserted. There is a k that minimises it, and it depends on m/n.
    The harness tells you n before you start, so you have no excuse for guessing.

    Then there is a second question, which is worth more: the harness inserts
    a **known** number of items, but a real stream does not tell you n in
    advance. What would you do then? You do not have to implement it - but
    observation.md asks.
    """

    # The harness inserts 8,000 items into 80,000 bits (bench.N_INSERT).
    EXPECTED_ITEMS = 8_000

    def __init__(self, n_bits, seed=246):
        self.n_bits = n_bits
        self.seed = seed
        # §4.4.2: FP(k) = (1 - e^(-kn/m))^k. Setting d/dk ln FP = 0 gives
        # k* = (m/n) ln 2. Here m/n = 10, so k* = 6.93 -> 7 hashes, and the
        # rate there is (1/2)^k* = 0.6185^(m/n) = 0.82%.
        # (The naive filter is k = 1: 1 - e^(-0.1) = 9.5%.)
        ratio = n_bits / self.EXPECTED_ITEMS
        self.k = max(1, round(ratio * math.log(2)))
        self.bits = bytearray((n_bits + 7) // 8)    # really n_bits, not n_bits bytes

    def _positions(self, item):
        # One 128-bit digest, split into h1 and h2, gives k positions
        # h1 + i*h2 (double hashing) instead of k separate digests.
        d = hashlib.blake2b(str(item).encode(), digest_size=16,
                            key=str(self.seed).encode()).digest()
        h1 = int.from_bytes(d[:8], "big")
        h2 = int.from_bytes(d[8:], "big") | 1
        return [(h1 + i * h2) % self.n_bits for i in range(self.k)]

    def add(self, item):
        for p in self._positions(item):
            self.bits[p >> 3] |= 1 << (p & 7)

    def __contains__(self, item):
        return all(self.bits[p >> 3] & (1 << (p & 7))
                   for p in self._positions(item))

    def memory_bits(self):
        # The bit array is the only state that grows; k and the seed are
        # parameters, the same kind NaiveFilter does not count either.
        return len(self.bits) * 8
