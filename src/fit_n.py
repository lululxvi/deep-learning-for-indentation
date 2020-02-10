import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def func(e, n, R):
    return R * e ** n


def fit_n_Al(E, sy, s033, s066, s1):
    e = sy / E
    e033 = e + 0.033
    e066 = e + 0.066
    e1 = e + 0.1
    print(e, e033, e066, e1)
    return curve_fit(func, [e, e033, e066, e1], [sy, s033, s066, s1])


def fit_n_Ti(E, sy, s008, s015, s033):
    e = sy / E
    e008 = e + 0.008
    e015 = e + 0.015
    e033 = e + 0.033
    print(e, e008, e015, e033)
    return curve_fit(func, [e, e008, e015, e033], [sy, s008, s015, s033])


def main():
    sy = np.loadtxt("B3090_peer/sigma_y.dat")
    s1 = np.loadtxt("B3090_peer/sigma_0.008.dat")
    s2 = np.loadtxt("B3090_peer/sigma_0.015.dat")
    s3 = np.loadtxt("B3090_peer/sigma_0.033.dat")
    print(np.mean(sy), np.std(sy))
    print(np.mean(s1), np.std(s1))
    print(np.mean(s2), np.std(s2))
    print(np.mean(s3), np.std(s3))

    sy = np.mean(sy)
    s1 = np.mean(s1)
    s2 = np.mean(s2)
    s3 = np.mean(s3)

    # E = 66.8
    # E = 70.1
    E = 110
    (n, R), pcov = fit_n_Ti(E, sy, s1, s2, s3)
    print(n, pcov[0, 0] ** 0.5)
    print(R)


if __name__ == "__main__":
    main()
