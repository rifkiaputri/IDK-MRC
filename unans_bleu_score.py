import math
import string
from collections import Counter
from fractions import Fraction
from nltk.translate.bleu_score import closest_ref_length, brevity_penalty, SmoothingFunction
from nltk.util import ngrams

STOPWORDS = set.union(set(['yang', 'di', 'dengan', 'dari', 'pun', 'the', 'of', 'kah']), set(string.punctuation))


def unans_corpus_bleu(
    list_of_references,
    hypotheses,
    answerables,
    weights=(0.25, 0.25, 0.25, 0.25),
    smoothing_function=None,
    auto_reweigh=False,
):
    """
    Calculate a single corpus-level UnAns-BLEU score (aka. system-level UnAns-BLEU) for all
    the hypotheses and their respective references.

    :param list_of_references: a corpus of lists of reference sentences, w.r.t. hypotheses
    :type list_of_references: list(list(list(str)))
    :param hypotheses: a list of hypothesis sentences
    :type hypotheses: list(list(str))
    :param answerables: a list of answerable questions
    :type answerables: list(list(str))
    :param weights: weights for unigrams, bigrams, trigrams and so on
    :type weights: list(float)
    :param smoothing_function:
    :type smoothing_function: SmoothingFunction
    :param auto_reweigh: Option to re-normalize the weights uniformly.
    :type auto_reweigh: bool
    :return: The corpus-level BLEU score.
    :rtype: float
    """
    # Before proceeding to compute BLEU, perform sanity checks.

    p_numerators = Counter()  # Key = ngram order, and value = no. of ngram matches.
    p_denominators = Counter()  # Key = ngram order, and value = no. of ngram in ref.
    hyp_lengths, ref_lengths = 0, 0

    assert len(list_of_references) == len(hypotheses), (
        "The number of hypotheses and their reference(s) should be the " "same "
    )

    # Iterate through each hypothesis and their corresponding references.
    for references, hypothesis, answerable in zip(list_of_references, hypotheses, answerables):
        # For each order of ngram, calculate the numerator and
        # denominator for the corpus-level modified precision.
        for i, _ in enumerate(weights, start=1):
            p_i = unans_modified_precision(references, hypothesis, answerable, i)
            p_numerators[i] += p_i.numerator
            p_denominators[i] += p_i.denominator
        
        # Calculate the hypothesis length and the closest reference length.
        # Adds them to the corpus-level hypothesis and reference counts.
        hyp_len = len(hypothesis)
        hyp_lengths += hyp_len
        ref_lengths += closest_ref_length(references, hyp_len)

    # Calculate corpus-level brevity penalty.
    bp = brevity_penalty(ref_lengths, hyp_lengths)

    # Uniformly re-weighting based on maximum hypothesis lengths if largest
    # order of n-grams < 4 and weights is set at default.
    if auto_reweigh:
        if hyp_lengths < 4 and weights == (0.25, 0.25, 0.25, 0.25):
            weights = (1 / hyp_lengths,) * hyp_lengths

    # Collects the various precision values for the different ngram orders.
    p_n = [
        Fraction(p_numerators[i], p_denominators[i], _normalize=False)
        for i, _ in enumerate(weights, start=1)
    ]

    # Returns 0 if there's no matching n-grams
    # We only need to check for p_numerators[1] == 0, since if there's
    # no unigrams, there won't be any higher order ngrams.
    if p_numerators[1] == 0:
        return 0

    # If there's no smoothing, set use method0 from SmoothinFunction class.
    if not smoothing_function:
        smoothing_function = SmoothingFunction().method0
    # Smoothen the modified precision.
    # Note: smoothing_function() may convert values into floats;
    #       it tries to retain the Fraction object as much as the
    #       smoothing method allows.
    p_n = smoothing_function(
        p_n, references=references, hypothesis=hypothesis, hyp_len=hyp_lengths
    )
    s = (w_i * math.log(p_i) for w_i, p_i in zip(weights, p_n))
    s = bp * math.exp(math.fsum(s))
    return s


def unans_modified_precision(references, hypothesis, answerable, n):
    """
    Calculate modified ngram precision.

    The normal precision method may lead to some wrong translations with
    high-precision, e.g., the translation, in which a word of reference
    repeats several times, has very high precision.

    This function only returns the Fraction object that contains the numerator
    and denominator necessary to calculate the corpus-level precision.
    To calculate the modified precision for a single pair of hypothesis and
    references, cast the Fraction object into a float.

    :param references: A list of reference translations.
    :type references: list(list(str))
    :param hypothesis: A hypothesis translation.
    :type hypothesis: list(str)
    :param answerable: An answerable questions.
    :type answerable: list(str)
    :param n: The ngram order.
    :type n: int
    :return: BLEU's modified precision for the nth order ngram.
    :rtype: Fraction
    """
    # Extracts all ngrams in hypothesis
    # Set an empty Counter if hypothesis is empty.
    counts = Counter(ngrams(hypothesis, n)) if len(hypothesis) >= n else Counter()
    
    if is_identical(hypothesis, answerable):
        # Give 0 value for all ngrams if the hyphotesis is identical to answerable question
        clipped_counts = {
            ngram: 0 for ngram, _ in counts.items()
        }
    else:
        # Extract a union of references' counts.
        # max_counts = reduce(or_, [Counter(ngrams(ref, n)) for ref in references])
        max_counts = {}
        for reference in references:
            reference_counts = (
                Counter(ngrams(reference, n)) if len(reference) >= n else Counter()
            )
            for ngram in counts:
                max_counts[ngram] = max(max_counts.get(ngram, 0), reference_counts[ngram])
        
        # Assigns the intersection between hypothesis and references' counts.
        clipped_counts = {
            ngram: min(count, max_counts[ngram]) for ngram, count in counts.items()
        }

    numerator = sum(clipped_counts.values())
    # Ensures that denominator is minimum 1 to avoid ZeroDivisionError.
    # Usually this happens when the ngram order is > len(reference).
    denominator = max(1, sum(counts.values()))

    return Fraction(numerator, denominator, _normalize=False)


def is_identical(tokens1, tokens2):
    t1 = set(tokens1)
    t2 = set(tokens2)
    diff = set.union((t1-t2), (t2-t1))
    
    if len(diff) > 0 and all([tok in STOPWORDS for tok in diff]):
        return True
    
    tokens1_no_sw = [tok.lower() for tok in tokens1 if tok.lower() not in STOPWORDS]
    tokens2_no_sw = [tok.lower() for tok in tokens2 if tok.lower() not in STOPWORDS]
    
    return tokens1_no_sw == tokens2_no_sw
