"""Integer softmax approximation used by the 16-bit boot sampler.

The decoder chooses the original Min-P/noise sampler 3/4 of the time and an
approximate softmax sampler 1/4 of the time. This keeps high-confidence output
while allowing plausible alternatives excluded by the original hard cutoff.
The branch and its sample use distinct (but correlated) 16-bit LCG updates. Softmax logits are
signed Q8; 23/256 approximates 1/(16*ln(2)), and the exp2 table is Q14.
Probabilities below the integer grid can still round to zero.
"""

EXP2_Q14 = (16384, 15689, 15024, 14387, 13777, 13193, 12634, 12098,
            11585, 11094, 10624, 10173, 9742, 9329, 8933, 8555)


def weights_for(logits):
    maximum = max(map(int, logits))
    result = []
    for value in logits:
        # x86 calculates in 32 bits; signed-int16 logit differences and
        # multiplication by 23 cannot overflow a signed 32-bit integer.
        exponent = ((maximum - int(value)) * 23 + 128) >> 8
        result.append(0 if exponent >= 224 else
                      EXP2_Q14[exponent & 15] >> (exponent >> 4))
    return result


def lcg(rnd):
    return (rnd * 25173 + 13849) & 0xffff


def sample(logits, rnd):
    rnd = lcg(rnd)
    if (rnd >> 8) < 64:
        weights = weights_for(logits)
        rnd = lcg(rnd)
        index = (rnd * sum(weights)) >> 16
        for token, weight in enumerate(weights):
            if index < weight:
                return token, rnd
            index -= weight
        raise AssertionError('softmax weights must include the maximum')

    threshold = max(map(int, logits)) - 240
    best, token = -(1 << 31), 0
    for i, logit in enumerate(logits):
        rnd = lcg(rnd)
        r = rnd >> 8
        if r >= 128:
            r -= 256
        candidate = int(logit) + r * 6
        if logit >= threshold and candidate > best:
            best, token = candidate, i
    return token, rnd
