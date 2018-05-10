
from typing import List, Tuple, Union
from random import sample, random
from functools import reduce

from xcs.bitstrings import BitConditionBase, BitString as CoreBitString
from xcs.input_encoding.real.center_spread.util import EncoderDecoder, random_in,add_and_rebound


class BitConditionRealEncoding(BitConditionBase):
    """See Documentation of base class."""

    def _encode(self, center_spreads: List[Tuple[float, float]]) -> CoreBitString:
        result = CoreBitString('')
        for ((center, spread), encoder) in zip(center_spreads, self.real_translators):
            result += encoder.encode(center)
            result += encoder.encode(spread)
        return result

    def __init__(self, encoders: Union[EncoderDecoder, List[EncoderDecoder]], center_spreads: List[Tuple[float, float]], mutation_strength: float, mutation_prob: float):
        assert len(center_spreads) > 0
        assert (mutation_strength > 0) and (mutation_strength <= 1)
        assert (mutation_prob >= 0) and (mutation_prob <= 1)
        assert all(map(lambda t: t[1] >= 0, center_spreads))  # spreads are all positive
        # did I get just 1 encoder, applicable for all reals?
        if isinstance(encoders, EncoderDecoder):
            encoders = [encoders] * len(center_spreads)
        assert len(center_spreads) == len(encoders)
        assert all(
            [(center >= encoder.extremes[0]) and
             (center <= encoder.extremes[1]) and  # centers are all in interval
             (spread >= 0) and
             (spread <= encoder.extremes[1] - encoder.extremes[0])  # spreads are all valid
             for (center, spread), encoder in zip(center_spreads, encoders)])
        self.real_translators = encoders
        self.mutation_strength = mutation_strength
        # clipped_center_spreads = list(map(self.clip_center_spread, center_spreads))
        # assert clipped_center_spreads == center_spreads, "Center-spreads do not respect conditions"
        BitConditionBase.__init__(self, bits=self._encode(center_spreads), mask=None, mutation_prob=mutation_prob)
        self.center_spreads = center_spreads

    @classmethod
    def random(cls, encoders: Union[EncoderDecoder, List[EncoderDecoder]], mutation_strength: float, mutation_prob: float, length: int):
        # did I get just 1 encoder, applicable for all reals?
        if isinstance(encoders, EncoderDecoder):
            encoders = [encoders] * length
        assert length == len(encoders)
        return cls.random_from_center_list(
            center_list=[encoder.random() for encoder in encoders],
            encoders=encoders,
            mutation_strength=mutation_strength, mutation_prob=mutation_prob)

    @classmethod
    def random_from_center_list(
            cls,
            center_list: List[float],
            encoders: Union[EncoderDecoder, List[EncoderDecoder]],
            mutation_strength: float,
            mutation_prob: float):
        # did I get just 1 encoder, applicable for all reals?
        if isinstance(encoders, EncoderDecoder):
            encoders = [encoders] * len(center_list)
        assert len(center_list) == len(encoders)
        assert all([(center <= encoder.extremes[1]) and (center >= encoder.extremes[0]) for center, encoder in zip(center_list, encoders)])
        center_spread_list = []
        for center, encoder in zip(center_list, encoders):
            spread = random_in(0, (encoder.extremes[1] - encoder.extremes[0]) / 2)  # relatively small spread
            center, spread = BitConditionRealEncoding.clip_center_spread_class(encoder, (center, spread))
            center_spread_list.append((center, spread))
        return cls(encoders=encoders, center_spreads=center_spread_list, mutation_strength=mutation_strength, mutation_prob=mutation_prob)

    @classmethod
    def clip_center_spread_class(cls, encoder: EncoderDecoder, center_spread: Tuple[float, float]) -> Tuple[float, float]:
        """Clips an interval in a way that it represents a valid range of values."""
        center, spread = center_spread
        center = encoder.clip(center)
        # spread = encoder.clip(spread)
        spread = min(spread, min(center - encoder.extremes[0], encoder.extremes[1] - center))
        return center, spread

    def clip_center_spread(self, center_spread: Tuple[float, float], encoder: EncoderDecoder) -> Tuple[float, float]:
        """Clips an interval in a way that it represents a valid range of values."""
        return BitConditionRealEncoding.clip_center_spread_class(encoder, center_spread)

    def __str__(self):
        """Overloads str(condition)"""
        return ','.join(["(%.2f, %.2f)" % (center, spread) for (center, spread) in self])

    def __len__(self):
        """Overloads len(condition)"""
        return len(self.center_spreads)

    def __getitem__(self, index):
        """Overloads condition[index]. The values yielded by the index
        operator are True (1), False (0), or None (#)."""
        if isinstance(index, slice):
            return BitConditionRealEncoding(self.real_translators, self.center_spreads[index], self.mutation_strength)
            # return BitCondition(self._bits[index], self._mask[index])
        # return self._bits[index] if self._mask[index] else None
        return self.center_spreads[index]

    def __contains__(self, item):
        """Overloads 'in' operator."""
        # assert isinstance(item, Tuple[float, float])
        return item in self.center_spreads

    def __call__(self, other):
        """Overloads condition(situation). Returns a Boolean value that
        indicates whether the other value satisfies this condition."""

        assert isinstance(other, (BitString, BitConditionRealEncoding)), "Type is => " + str(type(other))
        assert len(self) == len(other)

        if isinstance(other, BitString):
            situation = other

            center_spreads = [(center, spread) for (center, spread) in self]
            values = [value for value in situation]
            return all(
                [((value >= center - spread) and (value <= center + spread))
                 for ((center, spread), value) in zip(center_spreads, values)])
        else:  # 'other' is a condition
            other_condition = other

            my_center_spreads = [(my_center, my_spread) for (my_center, my_spread) in self]
            other_center_spreads = [(other_center, other_spread) for (other_center, other_spread) in other_condition]
            return all(
                [((my_center - my_spread <= other_center - other_spread) and (my_center + my_spread >= other_center + other_spread))
                 for ((my_center, my_spread), (other_center, other_spread)) in zip(my_center_spreads, other_center_spreads)])

    def __iter__(self):
        """Overloads iter(bitstring), and also, for bit in bitstring"""
        return iter(self.center_spreads)

    def _mutate_interval_by_translation(self, interval: Tuple[float, float], value: float) -> Tuple[float, float]:
        center, spread = interval
        bottom, top = center - spread, center + spread  # interval is [bottom, top]
        # let's do a translate
        if (spread == 0) or ((top - bottom) == (self.real_translators.extremes[1] - self.real_translators.extremes[0])):
            # there is nothing else to do:
            new_center, new_spread = center, spread
        else:
            delta_min_max = (
                max(self.real_translators.extremes[0] - bottom, value - top),
                min(self.real_translators.extremes[1] - top, value - bottom))
            # let's choose a delta - preventing a 0 which won't translate anything:
            delta = random_in(delta_min_max[0], delta_min_max[1])
            while delta == 0:
                delta = random_in(delta_min_max[0], delta_min_max[1])
            new_center = center + delta
            new_spread = spread
        return new_center, new_spread

    def _mutate_interval_by_stretching(self, interval: Tuple[float, float], value: float) -> Tuple[float, float]:
        center, spread = interval
        # let's do a shrinking\stretching of the interval
        if spread == 0 and (
                (center == self.real_translators.extremes[0]) or (center == self.real_translators.extremes[1])):
            # there is nothing else to do:
            new_center, new_spread = center, spread
        else:
            # Stretching can't make the interval too large. So:
            max_spread = min(center - self.real_translators.extremes[0], self.real_translators.extremes[1] - center)
            min_spread = abs(center - value)
            new_spread = random_in(min_spread, max_spread)
            while new_spread == spread:  # I actually want to mutate!
                new_spread = random_in(min_spread, max_spread)
            new_center = center
        return new_center, new_spread

    def mutate_intervals(self, situation):
        center_spread_list = []
        for (center, spread), value in zip(self, situation):
            if random() <= self.mutation_prob:
                do_translation = random() < 0.5
                do_stretching = not do_translation
                if do_translation:
                    new_center, new_spread = self._mutate_interval_by_translation(interval=(center, spread), value=value)
                    do_stretching = (new_center, new_spread) == (center, spread)
                if do_stretching:
                    new_center, new_spread = self._mutate_interval_by_stretching(interval=(center, spread), value=value)
            else:
                new_center, new_spread = center, spread
            center_spread_list.append((new_center, new_spread))
        return BitConditionRealEncoding(
            encoders=self.real_translators,
            center_spreads=center_spread_list,
            mutation_strength=self.mutation_strength, mutation_prob=self.mutation_prob)

    def mutate_ints2(self, situation):
        def get_mutation_value(m: float, encoder: EncoderDecoder) -> float:
            return (-1 if random() < 0.5 else 1) * \
                   random_in(0, m) * (encoder.extremes[1] - encoder.extremes[0])

        m = 0.1
        center_spread_list = []
        for (center, spread), value, encoder in zip(self, situation, self.real_translators):
            tries_left = 1  # 3
            mutation_valid = False
            while (tries_left > 0) and not mutation_valid:
                if random() <= .5:  # TODO self.mutation_prob:
                    # do 'center' mutation:
                    min_value_d = encoder.min_distance_to_ends(value)
                    min_delta = -min_value_d - center + value
                    max_delta = min(min_value_d - center + value, (encoder.extremes[1] - 2*center + value) / 2)
                    unclipped_center = center + random_in(min_delta, max_delta)
                    unclipped_spread = random_in(abs(unclipped_center - value),
                                                 encoder.min_distance_to_ends(unclipped_center))
                else:  # no mutation of the center
                    unclipped_center = center
                    if random() <= .5:  # TODO self.mutation_prob:
                        unclipped_spread = random_in(abs(unclipped_center - value), encoder.min_distance_to_ends(unclipped_center))
                    else:  # no mutation of the spread
                        unclipped_spread = spread
                new_center, new_spread = self.clip_center_spread((unclipped_center, unclipped_spread), encoder)
                # update termination condition
                tries_left -= 1
                mutation_valid = (value >= new_center - new_spread) and (value <= new_center + new_spread)
            if not mutation_valid:
                # stay:
                new_center, new_spread = center, spread
            center_spread_list.append((new_center, new_spread))
        return BitConditionRealEncoding(
            encoders=self.real_translators,
            center_spreads=center_spread_list,
            mutation_strength=self.mutation_strength, mutation_prob=self.mutation_prob)

    def mutate_as_in_paper(self, situation):
        def get_mutation_value(m: float, encoder: EncoderDecoder) -> float:
            return (-1 if random() < 0.5 else 1) * \
                   random_in(0, m) * (encoder.extremes[1] - encoder.extremes[0])

        m = 0.1
        center_spread_list = []
        for (center, spread), value, encoder in zip(self, situation, self.real_translators):
            tries_left = 1  # 3
            mutation_valid = False
            while (tries_left > 0) and not mutation_valid:
                if random() <= .5: # TODO self.mutation_prob:
                    # do 'center' mutation:
                    delta = get_mutation_value(m, encoder)
                    unclipped_center = add_and_rebound(center, delta, m=encoder.extremes[0], M=encoder.extremes[1])
                    unclipped_spread = random_in(0, min(unclipped_center - encoder.extremes[0], encoder.extremes[1] - unclipped_center))
                    # unclipped_center = encoder.clip(center + get_mutation_value(m, encoder))
                else:  # no mutation of the center
                    unclipped_center = center
                    if random() <= .5: # TODO self.mutation_prob:
                        delta = get_mutation_value(m, encoder)
                        unclipped_spread = add_and_rebound(spread, delta, m=0, M=encoder.extremes[1] - encoder.extremes[0])
                        # unclipped_spread = encoder.clip(spread + get_mutation_value(m, encoder))  # TODO: is this a bug?? Shouldn't 'spread, be in [0, M - m] ?
                    else:  # no mutation of the spread
                        unclipped_spread = spread
                new_center, new_spread = self.clip_center_spread((unclipped_center, unclipped_spread), encoder)
                # update termination condition
                tries_left -= 1
                mutation_valid = (value >= new_center - new_spread) and (value <= new_center + new_spread)
            if not mutation_valid:
                # stay:
                new_center, new_spread = center, spread

                #
                # TODO: this new way of doing misses lots of covering!
                #

            center_spread_list.append((new_center, new_spread))
        return BitConditionRealEncoding(
            encoders=self.real_translators,
            center_spreads=center_spread_list,
            mutation_strength=self.mutation_strength, mutation_prob=self.mutation_prob)

    def mutate(self, situation):
        # return self.mutate_as_in_paper(situation)
        return self.mutate_ints2(situation)


    def crossover_with(self, other, points):
        """Perform 2-point crossover on this bit condition and another of
        the same length, returning the two resulting children.

        Usage:
            offspring1, offspring2 = condition1.crossover_with(condition2)

        Arguments:
            other: A second BitCondition of the same length as this one.
            points: An int, the number of crossover points of the
                crossover operation.
        Return:
            A tuple (condition1, condition2) of BitConditions, where the
            value at each position of this BitCondition and the other is
            preserved in one or the other of the two resulting conditions.
        """
        assert isinstance(other, BitConditionRealEncoding)
        assert len(self) == len(other)
        assert points < len(self)

        # print(self)
        # print(other)
        if self == other:
            # nothing to do
            # print(" CROSSOVER =====> ARE THE SAME????????????????????????")  # TODO: take this out.
            return self, other
        else:
            # print(" CROSSOVER =====> not the same")
            pts = [-1] + sample(range(len(self) - 1), points) + [len(self) - 1]
            pts.sort()
            pts = list(map(lambda x: x + 1, pts))
            genome_1, genome_2 = self, other
            result = ([], [])
            result_alt = ([], [])
            for begin, end in zip(pts[:-1], pts[1:]):
                result = (result[0] + genome_1.center_spreads[begin: end], result[1] + genome_2.center_spreads[begin: end])
                strip_1 = genome_1.center_spreads[begin: end]
                strip_2 = genome_2.center_spreads[begin: end]
                if random() < 0.5:
                    orig_strip_1 = strip_1
                    orig_strip_2 = strip_2
                    # swap last allele
                    last_gene_1 = (strip_1[-1][0], strip_2[-1][1])
                    last_gene_2 = (strip_2[-1][0], strip_1[-1][1])
                    strip_1 = strip_1[:-1] + [last_gene_1]
                    strip_2 = strip_2[:-1] + [last_gene_2]
                result_alt = (result_alt[0] + strip_1, result_alt[1] + strip_2)
                genome_1, genome_2 = (self, other) if genome_1 == other else (other, self)
            return \
                BitConditionRealEncoding(self.real_translators, result[0], self.mutation_strength, mutation_prob=self.mutation_prob), \
                BitConditionRealEncoding(self.real_translators, result[1], self.mutation_strength, mutation_prob=self.mutation_prob)
            # return \
            #     BitConditionRealEncoding(self.real_translators, result_alt[0], self.mutation_strength, mutation_prob=self.mutation_prob), \
            #     BitConditionRealEncoding(self.real_translators, result_alt[1], self.mutation_strength, mutation_prob=self.mutation_prob)


class BitString(CoreBitString):

    def __init__(self, encoders: Union[EncoderDecoder, List[EncoderDecoder]], reals: List[float]):
        assert len(reals) > 0
        # did I get just 1 encoder, applicable for all reals?
        if isinstance(encoders, EncoderDecoder):
            encoders = [encoders] * len(reals)
        assert len(encoders) == len(reals)
        self.as_reals = reals
        self.real_translators = encoders
        as_bitstring = reduce(
            lambda x,y: x+y,
            [encoder.encode(a_real) for encoder, a_real in zip(self.real_translators, self.as_reals)])
        CoreBitString.__init__(self, bits=as_bitstring)


    @classmethod
    def random_from(cls, encoders: Union[EncoderDecoder, List[EncoderDecoder]], length):
        assert isinstance(length, int) and length >= 0
        # did I get just 1 encoder, applicable for all reals?
        if isinstance(encoders, EncoderDecoder):
            encoders = [encoders] * length
        list_of_reals = [encoder.random() for encoder in encoders]
        return cls(encoders=encoders, reals=list_of_reals)

    def __len__(self):
        """Overloads len(instance)"""
        return len(self.as_reals)

    def __iter__(self):
        """Overloads iter(bitstring), and also, for bit in bitstring"""
        return iter(self.as_reals)

    def __getitem__(self, index):
        """Overloads bitstring[index]"""
        if isinstance(index, slice):
            return BitString(encoders=self.real_translators, reals=self.as_reals[index])
        return self.as_reals[index]

    def __str__(self):
        """Overloads str(bitstring)"""
        return ','.join(["%.2f" % (value) for value in self])

    def cover(self, wildcard_probability: float):
        """Create a new bit condition that matches the provided bit string,
        with the indicated per-index wildcard probability.

        Usage:
            condition = BitCondition.cover(bitstring, .33)
            assert condition(bitstring)

        Arguments:
            bits: A BitString which the resulting condition must match.
            wildcard_probability: A float in the range [0, 1] which
            indicates the likelihood of any given bit position containing
            a wildcard.
        Return:
            A randomly generated BitCondition which matches the given bits.
        """
        assert (wildcard_probability >= 0) and (wildcard_probability <= 1)
        r = BitConditionRealEncoding.random_from_center_list(
            center_list=[value for value in self],
            encoders=self.real_translators,
            mutation_strength=0.1,
            mutation_prob=.2
        )  # TODO: value of mutation strenght!!!!! AND mutation prob!!!!!

        # r = BitConditionRealEncoding(encoders=self.real_translators, center_spreads=list(map(lambda v: (v,0), [value for value in self])), mutation_strength=0.1, mutation_prob=.2)

        # convert which inputs to 'match-all'?
        center_spreads = [
            (center, spread)
            if random() >= wildcard_probability  # TODO: put this back as wildcard_probability, once the experiment with spreads = 0 is over.
            else BitConditionRealEncoding.clip_center_spread_class(encoder, (encoder.extremes[0] + ((encoder.extremes[1] - encoder.extremes[0]) / 2), encoder.extremes[1] - encoder.extremes[0]))
            for (center, spread), encoder in zip(r, self.real_translators)]
        r2 = BitConditionRealEncoding(r.real_translators, center_spreads=center_spreads, mutation_strength=r.mutation_strength, mutation_prob=r.mutation_prob)
        if not r2(self):  # TODO: take this sanity check out!
            raise RuntimeError("does not cover anymore!!!")
        return r2
