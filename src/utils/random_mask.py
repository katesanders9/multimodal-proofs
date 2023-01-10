''' From Nathaniel's NELLIE codebase.'''

from src.utils import flatten
import numpy as np
from IPython import embed

MASK="<extra_id_0>" # single special token in T5 vocab

def random_spans_noise_mask(length,
                            noise_density=0.6,
                            mean_noise_span_length=3.0):
    """Noise mask consisting of random spans of noise tokens.
    The number of noise tokens and the number of noise spans and non-noise spans
    are determined deterministically as follows:
      num_noise_tokens = round(length * noise_density)
      num_nonnoise_spans = num_noise_spans = round(
         num_noise_tokens / mean_noise_span_length)
    Spans alternate between non-noise and noise, beginning with non-noise.
    Subject to the above restrictions, all masks are equally likely.
    Args:
      length: an int32 scalar (length of the incoming token sequence)
      noise_density: a float - approximate density of output mask
      seeds: an int32 Tensor, shaped (2, 2)
      mean_noise_span_length: a number
    Returns:
      a boolean tensor with shape [length]
    """

    orig_length = length
    # increase length to avoid degeneracy
    length = max(length, 2)
    num_noise_tokens = int(np.round(float(length) * noise_density * np.random.normal(loc=1, scale=0.4)))
    # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
    num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
    num_noise_spans = int(np.round(float(num_noise_tokens) / mean_noise_span_length))
    # avoid degeneracy by ensuring positive number of noise spans
    num_noise_spans = max(num_noise_spans, 1)
    num_nonnoise_tokens = length - num_noise_tokens

    # pick the lengths of the noise spans and the non-noise spans
    def _random_segmentation(num_items, num_segments):
        """Partition a sequence of items randomly into non-empty segments.
        Args:
          num_items: an integer scalar > 0
          num_segments: an integer scalar in [1, num_items]
          seed: an integer seed
        Returns:
          a Tensor with shape [num_segments] containing positive integers that add
          up to num_items
        """
        num_segments = min(num_items, num_segments)
        barslots = (num_items - 1)
        idxs = np.random.choice(np.arange(1, 1 + barslots), num_segments - 1, replace=False)
        segment_length = []
        prev_idx = 0
        for idx in sorted(idxs) + [num_items]:
            new_length = idx - prev_idx
            segment_length.append(new_length)
            prev_idx = idx

        assert sum(segment_length) == num_items
        return segment_length

    noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
    nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens,
                                                 max(1, num_noise_spans + np.random.choice([-1, 0, 1])))
    maxlen = max(len(noise_span_lengths), len(nonnoise_span_lengths))

    noise_spans = [list(np.ones(noise_span_lengths[i] if i < len(noise_span_lengths) else 0, dtype=int))
                   for i in range(maxlen)]
    nonnoise_spans = [list(np.zeros(nonnoise_span_lengths[i] if i < len(nonnoise_span_lengths) else 0, dtype=int))
                      for i in range(maxlen)]
    mask = flatten((i + j) for (i,j) in (zip(noise_spans, nonnoise_spans)
                   if np.random.random() < 0.5 else
                   zip(nonnoise_spans, noise_spans)))
    return mask


def random_mask(instr):
    tokens = instr.split()
    num_tokens = len(tokens)
    mask = random_spans_noise_mask(num_tokens, noise_density=0.6)
    out = []
    last_mask_idx = -2
    for i, (tok_i, m_i) in enumerate(zip(tokens, mask)):
        if m_i:
            if i != last_mask_idx + 1:
                out.append(MASK)
            last_mask_idx = i
        else:
            out.append(tok_i)

    return ' '.join(out)


if __name__ == '__main__':
    from src.utils.worldtree_utils import WorldTree

    wt = WorldTree()
    instr = sorted(wt.to_sentence_corpus(), key=len)[-1]
    print(sorted(random_mask(instr) for _ in range(100)))
    embed(user_ns=locals())
