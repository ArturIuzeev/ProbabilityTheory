import scipy
import time
import numpy as np
import matplotlib.pyplot as plt

rnd = np.random.default_rng(999)
sizes = [100, 500, 1000]


def make_distribution():
    class MyDistribution(scipy.stats.rv_continuous):
        def _pdf(self, x):
            return pdf(x)

        def _get_support(self):
            return 0, 10

    return MyDistribution(momtype=0)


distribution = make_distribution()


def bin_search(p):
    l = 0
    r = 400
    while r - l > 1e-6:
        m = (l + r) / 2
        if cdf(m) < p:
            l = m
        else:
            r = m
    return l


def pdf(x):
    return (np.e ** (-(((5 - x ** 5) / np.sqrt(2)) ** 2)) / np.sqrt(2 * np.pi)) * 5 * (x ** 4)


def cdf(x):
    return 1 / 2 * (scipy.special.erf(((x ** 5) - 5) / np.sqrt(2))) + 0.5


def plot(size, func, form):
    start = time.time_ns()
    r = func(size)
    e = time.time_ns() - start
    s = e / 1e9
    m = s / 60
    h = m / 60
    s %= 60
    m %= 60
    print("Time: %02dh %02dm %02ds" % (h, m, s))

    u, c = np.unique(np.round(r, decimals=2), return_counts=True)
    c = c / sum(c)
    c *= 30

    x = np.arange(0, 4, 0.01)
    plt.bar(u, c, color="yellow", width=0.05, label="Random values")
    plt.plot(u, c, color="red", alpha=0.5)
    plt.plot(x, distribution.pdf(x), label="PDF")
    plt.title(form % size)
    plt.xlim(1, 1.8)
    plt.legend()
    plt.show()


def rvs(size):
    return distribution.rvs(size=size, random_state=rnd)


def inverse(size):
    return np.array([bin_search(x) for x in rnd.uniform(size=size)])


def rejection(size):
    return np.array([next_rand() for x in range(size)])

count = 0
norm = 0
def next_rand():
    helper = distribution.pdf(1.37)
    left = 0
    right = 2.5
    global count
    global norm
    while True:
        y = rnd.uniform() * right
        y = y + left
        q = distribution.pdf(y) / helper / right
        count += 1
        if rnd.uniform() < q:
            norm += 1
            # print(count)
            # print("All")
            # print(norm)
            # print("Norm")
            return y

def countTime(func):
    start = time.time_ns()
    func()
    return time.time_ns() - start

plot(100, rvs, "RVS (size=%s)")
plot(1000, rvs, "RVS (size=%s)")

plot(100, inverse, "CDF (size=%s)")
plot(1000, inverse, "CDF (size=%s)")
plot(10000, inverse, "CDF (size=%s)")

plot(100, rejection, "Rej (size=%s)")
plot(1000, count, norm, rejection)

plt.plot(sizes, [countTime(lambda : rvs(size)) for size in sizes], label="RVS")
plt.plot(sizes, [countTime(lambda : inverse(size)) for size in sizes], label="Inverse CDF")
plt.plot(sizes, [countTime(lambda : rejection(size)) for size in sizes], label="Rejection sampling")
plt.legend()
plt.ylabel("Time")
plt.xlabel("Size")
plt.yscale("log")
plt.show()