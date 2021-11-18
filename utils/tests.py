import numpy as np
from scipy.stats import chi2, kstwobign

np.random.seed(42)


def ks_test(samples, cdf, alpha=0.05, **kwargs):
    """
    Kolmogorov-Smirnov test for checking if samples are from cdf's distribution

    samples: numpy array of samples from specific distribution
    cdf: cumulative distribution function of this distribution
    alpha: 1st type error probability
    **kwargs: parameters to pass to cdf
    """
    n = len(samples)
    empirical_cdf = np.arange(n) / n
    theoretical_cdf = np.array([cdf(x, **kwargs) for x in sorted(samples)])
    Dn = np.max(np.abs(theoretical_cdf - empirical_cdf))
    ks_value = np.sqrt(n) * Dn

    significance_level = 1 - alpha
    critical_value = kstwobign.ppf(significance_level)

    print(
        f"KS test. Null hypothesis: no difference between samples and specified cdf. Significance_level: {significance_level}"
    )
    if ks_value < critical_value:
        print(f"Can't reject null hypothesis, {ks_value} < {critical_value}.")
    else:
        print(f"Reject null hypothesis, {ks_value} >= {critical_value}.")


def chi2_discrete(samples, pmf, alpha=0.05, **kwargs):
    """
    Chi2 test for checking if samples are from pmf's distribution
    Works with discrete data.

    samples: numpy array of samples from specific distribution
    pmf: probability mass function of this distribution
    alpha: 1st type error probability
    **kwargs: parameters to pass to pmf
    """
    n = len(samples)
    obs = np.unique(samples, return_counts=True)[1]

    dof = len(obs) - 1
    probas = np.array([pmf(i, **kwargs) for i in range(len(obs))])
    sum_of_probas = round(sum(probas), 2)
    assert sum_of_probas == 1, f"Sum of probas = {sum_of_probas} != 1"
    exp = n * probas
    chi2_value = sum((obs - exp) ** 2 / exp)

    significance_level = 1 - alpha
    critical_value = chi2.ppf(significance_level, dof)

    print(
        f"Chi2 test. Null hypothesis: no difference in expected and observed. Significance_level: {significance_level}"
    )
    if chi2_value < critical_value:
        print(f"Can't reject null hypothesis, {chi2_value} < {critical_value}.")
    else:
        print(f"Reject null hypothesis, {chi2_value} >= {critical_value}.")


def chi2_continious(samples, cdf, N_bins=10, alpha=0.05, **kwargs):
    """
    Chi2 test for checking if samples are from cdf's distribution
    Works with continious data.

    samples: numpy array of samples from specific distribution
    cdf: cumulative distribution function of this distribution
    N_bins: interger, which represents into how many bins samples are discretized
    alpha: 1st type error probability
    **kwargs: parameters to pass to cdf
    """
    n = len(samples)
    dof = N_bins - 1
    bins_counts, bins_borders = np.histogram(samples, bins=N_bins)

    probas = np.diff([cdf(x, **kwargs) for x in bins_borders])
    assert round(np.sum(probas), 1) == 1, "Sum of probas != 1"

    significance_level = 1 - alpha
    chi2_value = np.sum((bins_counts - n * probas) ** 2 / (n * probas))
    critical_value = chi2.ppf(significance_level, dof)

    print(
        f"Chi2 test. Null hypothesis: no difference in expected and observed. Significance_level: {significance_level}"
    )
    if chi2_value < critical_value:
        print(f"Can't reject null hypothesis, {chi2_value} < {critical_value}.")
    else:
        print(f"Reject null hypothesis, {chi2_value} >= {critical_value}.")
