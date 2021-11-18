from math import comb

import numpy as np
from scipy.special import erf

np.random.seed(42)


def uniform_cdf(x, a=0, b=1):
    return 0 if x < a else 1 if x > b else (x - a) / (b - a)


def binomial_sample(m=1, p=0.5):
    return np.sum(np.random.rand(m) < p)


def negative_binomial_sample(r=1, p=0.5):
    successes, failures = 0, 0
    while failures < r:
        is_success = np.random.rand() >= p
        successes, failures = successes + int(is_success), failures + int(not is_success)
    return successes


def normal_sample(N=12, loc=0, scale=1):
    return loc + (12 / N) ** 0.5 * (np.sum(np.random.rand(N)) - N / 2) * scale


def exponential_sample(a=1):
    return -np.log(np.random.rand()) / a


def logistic_sample(loc=0, scale=1):
    x = np.random.rand()
    return loc + scale * np.log(x / (1 - x))


def generate_samples(generate_sample, n=1000, **kwargs):
    return np.array([generate_sample(**kwargs) for _ in range(n)])


def binomial_pmf(k, m, p):
    return comb(m, k) * p ** k * (1 - p) ** (m - k)


def negative_binomial_pmf(k, r, p):
    return comb(k + r - 1, r - 1) * (1 - p) ** k * p ** r


def normal_cdf(x, loc=0, scale=1):
    return 0.5 * (1 + erf((x - loc) / (scale * 2 ** 0.5)))


def exponential_cdf(x, a=1):
    return 1 - np.exp(-a * x)


def logistic_cdf(x, loc=0, scale=1):
    return 1 / (1 + np.exp(-(x - loc) / scale))


def multiplicative_congruential(a0, beta, M=2 ** 31, n=1000):
    samples = np.ones(n + 1, dtype=float) * a0
    for i in range(1, len(samples)):
        samples[i] = beta * samples[i - 1] % M

    return samples[1:] / M


def maclaren_marsaglia(g1, g2, K, n=1000):
    samples = np.zeros(n)
    v = g1[:K]  # initialize v with first K values of g1
    s = np.array(K * g2, dtype=int)  # array of idxs shape: (n,)
    for i in range(n - K):
        samples[i] = v[s[i]]
        v = np.append(v[1:], g1[K + i])
    return samples
