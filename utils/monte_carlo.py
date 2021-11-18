import numpy as np
import seaborn as sns

np.random.seed(42)


def f1(x):
    return 1 / (x ** 4 + 3 * x ** 2 + 17)


def f2(x, y):
    return (x * y ** 2 + 1) * np.sin(x)


def calculate_I1(N, a=0, b=100):
    return 2 * (b - a) * np.mean([f1(x) for x in a + b * np.random.rand(N)])


def calculate_I2(N, a=-3, b=3):
    return (
        (b - a) ** 2
        * np.sum(
            [
                f2(x, y)
                for x, y in a + (b - a) * np.random.rand(N, 2)
                if abs(x) + abs(y) < 3
            ]
        )
        / N
    )


def calculate_integrals_for_Ns(
    calculate_integral, exact_value, Ns=[100, 1000, 10000], **kwargs
):
    values = []
    diffs = []
    for N in Ns:
        values.append(calculate_integral(N, **kwargs))
        diffs.append(np.abs(values[-1] - exact_value))

    return values, diffs, Ns


def plot_error_graph(Ns, diffs, title=None, ax=None):
    plot_xticks = range(len(Ns))
    g = sns.lineplot(x=plot_xticks, y=diffs, ax=ax)

    for i, txt in enumerate(diffs):
        g.annotate("{:.2e}".format(txt), (plot_xticks[i], diffs[i]))

    g.set_xticks(plot_xticks)
    g.set_xticklabels(Ns)

    g.set_title(title)
    return g