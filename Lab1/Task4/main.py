import math
import scipy
import matplotlib.pyplot as plt

count = [10, 100, 1000, 10000]
probabilities = [0.001, 0.01, 0.1, 0.25, 0.5]
exactProbabilities = []
noExactProbabilities = []


def countX(Sn, c, n, p):
    return (Sn - n * p) / c


def countFi(x):
    a = 1 / math.sqrt(2 * math.pi)
    degree = (x ** 2) / 2 * -1
    eps = math.e ** degree
    return a * eps

def e(z):
    return math.e ** (((z**2) / 2) * -1)


def countFi2(x):
    shot = 1/(math.sqrt(math.pi * 2))
    integral, _ = scipy.integrate.quad(e, 0, x)
    return shot * integral

def func1(n, p):
    q = 1 - p
    result = 0
    for S in range(int(n / 2 - math.sqrt(n * p * q)), int(n / 2 + math.sqrt(n * p * q))):
        c = math.sqrt(n * p * q)
        x = countX(S, c, n, p)
        fi = countFi(x)

        shot = 1 / c
        result += shot * fi

    exactProbabilities.append(result)
    return result


def func2(n, p):
    q = 1 - p
    c = math.sqrt(n * p * q)

    x1 = countX(n / 2 - c, c, n, p)
    x2 = countX(n / 2 + c, c, n, p)

    fi1 = countFi2(x1)
    fi2 = countFi2(x2)

    result = fi2 - fi1
    noExactProbabilities.append(result)
    return result


for n in count:
    for p in probabilities:
        func1(n, p)
        func2(n, p)
        print(n, p)


diff = [exactProbabilities[i] - noExactProbabilities[i] for i in range(len(exactProbabilities))]
plt.plot(diff)
plt.xlabel('Index')
plt.ylabel('Difference')
plt.title('Difference between Exact Probabilities and Approximations')
plt.show()

