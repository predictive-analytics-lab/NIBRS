# %%
import numpy as np

def wilson_error(n_s: int, n: int, z=1.96):
    """
    Wilson score interval

    param n_s: number of successes
    param n: total number of events
    param z: The z-value

    return: The lower and upper bound of the Wilson score interval
    """
    n_f = np.max([1, n - n_s])
    denom = n + z ** 2
    adjusted_p = (n_s + z ** 2 * 0.5) / denom
    ci = (z / denom) * np.sqrt((n_s * n_f / n) + (z ** 2 / 4))
    return adjusted_p, ci

wilson_error(4.5, 3)
# %%
