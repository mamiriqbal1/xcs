import math

from random import random, randint
from typing import List

from xcs.bitstrings import BitString
from xcs.input_encoding import real


class EncoderDecoder(real.EncoderDecoder):

    def __init__(self, min_value: float, max_value: float, encoding_bits: int):
        assert max_value >= min_value
        assert encoding_bits >= 1
        self.extremes = (float(min_value), float(max_value))
        self.encoding_bits = encoding_bits

    def __str__(self):
        return "Encoder, %d bits, in [%.2f,%.2f]" % (self.encoding_bits, self.extremes[0], self.extremes[1])

    def random(self) -> float:
        return random() * (self.extremes[1] - self.extremes[0]) + self.extremes[0]

    def choice(self, length: int) -> List[float]:
        return [self.random() for _ in range(length)]

    def clip(self, d: float) -> float:
        return max(self.extremes[0], min(self.extremes[1], d))

    def encode_as_int(self, d: float) -> int:
        """Does encoding of a float into one of 'm' ints, where m = 2^k (k=number of bits used for representation)"""
        assert (d >= self.extremes[0]) and (d <= self.extremes[1]), "%.2f is not in correct interval [%.2f,%.2f]" % (d, self.extremes[0], self.extremes[1])
        return int(
            math.floor(
                (math.pow(2, self.encoding_bits) - 1) * (d - self.extremes[0]) /
                (self.extremes[1] - self.extremes[0])))

    def encode(self, d: float) -> BitString:
        assert (d >= self.extremes[0]) and (d <= self.extremes[1]), "Can't encode %.2f because interval is [%.2f,%.2f]" % (d, self.extremes[0], self.extremes[1])
        as_bitstring = ('{0:0%db}' % self.encoding_bits).format(self.encode_as_int(d))
        return BitString(as_bitstring)

    def decode(self, b: BitString) -> int:
        return int(str(b), 2)

    def mutate_float(self, d: float, factor: float) -> float:
        """Mutates a bitstring encoded with this specific encoder, by a certain factor (in [0,1])"""
        assert (factor >= 0) and (factor <= 1)
        mutation_value = (random() * factor) * (self.extremes[1] - self.extremes[0])
        mutation_sign = 1 if random() > 0.5 else -1
        r = d + mutation_value * mutation_sign
        return max(self.extremes[0], min(r, self.extremes[1]))

    def mutate(self, b: BitString, factor: float) -> BitString:
        """Mutates a bitstring encoded with this specific encoder, by a certain factor (in [0,1])"""
        return self.encode(self.mutate_float(self.decode(b), factor))

    def is_in_interval(self, a_value) -> bool:
        """Returns true if the value is in interval defined by this encoder."""
        return (a_value >= self.extremes[0]) and (a_value <= self.extremes[1])

    def min_distance_to_ends(self, a_value: float) -> float:
        """Minimum of distances to either ends."""
        assert self.is_in_interval(a_value)
        return min(a_value - self.extremes[0], self.extremes[1] - a_value)


def random_in(bottom: float, top: float) -> float:
    """Returns random value in interval"""
    if bottom == top:
        return bottom
    bottom = min(bottom, top)
    top = max(bottom, top)
    return random() * (top - bottom) + bottom


def add_and_rebound(v: float, delta: float, m: float, M: float) -> float:
    """Rebound from walls."""
    mutated_v = v + delta
    if mutated_v > M:
        return 2 * M - mutated_v
    elif mutated_v < m:
        return 2 * m - mutated_v
    else:
        return mutated_v

