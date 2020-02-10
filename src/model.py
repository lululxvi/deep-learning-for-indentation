from __future__ import division
from __future__ import print_function

import csv
import random

import numpy as np
import pandas as pd
from scipy import optimize

from data import BerkovichData, ExpData, FEMData


def Pi1(Estar_sigma33):
    x = np.log(Estar_sigma33)
    return -1.131 * x ** 3 + 13.635 * x ** 2 - 30.594 * x + 29.267


def Pi2(Estar_sigma33, n):
    x = np.log(Estar_sigma33)
    return (
        (-1.40557 * n ** 3 + 0.77526 * n ** 2 + 0.1583 * n - 0.06831) * x ** 3
        + (17.93006 * n ** 3 - 9.22091 * n ** 2 - 2.37733 * n + 0.86295) * x ** 2
        + (-79.99715 * n ** 3 + 40.5562 * n ** 2 + 9.00157 * n - 2.54543) * x
        + 122.65069 * n ** 3
        - 63.88418 * n ** 2
        - 9.58936 * n
        + 6.20045
    )


def Pi4(hrhm):
    return 0.268536 * (0.9952495 - hrhm) ** 1.1142735


def Pi5(hrhm):
    return 1.61217 * (
        1.13111 - 1.74756 ** (-1.49291 * hrhm ** 2.535334) - 0.075187 * hrhm ** 1.135826
    )


def Pitheta(theta, Estar_sigma):
    x = np.log(Estar_sigma)
    if theta == 60:
        return -0.154 * x ** 3 + 0.932 * x ** 2 + 7.657 * x - 11.773
    if theta == 80:
        return -2.913 * x ** 3 + 44.023 * x ** 2 - 122.771 * x + 119.991
    if theta == 50:
        return 0.0394 * x ** 3 - 1.098 * x ** 2 + 9.862 * x - 11.837
    raise NotImplementedError


def epsilon_r(theta):
    return 2.397e-5 * theta ** 2 - 5.311e-3 * theta + 0.2884


def forward_model(E, n, sigma_y, nu, Pm=None, hm=None, nu_i=0.07, E_i=1100e9):
    assert (hm is None) ^ (Pm is None)

    cstar = 1.1957
    # cstar = 1.2370

    sigma_33 = sigma_y * (1 + E / sigma_y * 0.033) ** n
    Estar = 1 / ((1 - nu ** 2) / E + (1 - nu_i ** 2) / E_i)
    C = sigma_33 * Pi1(Estar / sigma_33)
    if hm is None:
        hm = (Pm / C) ** 0.5
    else:
        Pm = C * hm ** 2
    dPdh = Estar * hm * Pi2(Estar / sigma_33, n)
    Am = (1 / cstar / Estar * dPdh) ** 2
    p_ave = Pm / Am
    if p_ave / Estar > Pi4(0):
        hr = 0
    elif p_ave / Estar < 0:
        hr = hm
    else:
        hr = optimize.brentq(lambda x: Pi4(x) - p_ave / Estar, 0, 0.9952495) * hm
    WpWt = Pi5(hr / hm)
    return Estar, C, hr, dPdh, WpWt, p_ave


def inverse_model(C, WpWt, dPdh, nu, hm, nu_i=0.07, E_i=1100e9):
    cstar = 1.1957  # Conical
    # cstar = 1.2370  # Berkovich

    if WpWt < Pi5(0):
        hr = 1e-9
    else:
        hr = optimize.brentq(lambda x: Pi5(x) - WpWt, 0, 1) * hm
    Pm = C * hm ** 2
    Am = (Pm * cstar / dPdh / Pi4(hr / hm)) ** 2
    Estar = dPdh / cstar / Am ** 0.5
    p_ave = Pm / Am
    sigma_33 = optimize.brentq(lambda x: Pi1(Estar / x) - C / x, 1e7, 1e10)
    E = (1 - nu ** 2) / (1 / Estar - (1 - nu_i ** 2) / E_i)
    try:
        n = optimize.brentq(
            lambda x: Pi2(Estar / sigma_33, x) - dPdh / Estar / hm, 0, 0.5
        )
    except ValueError:
        n = 0
    if n > 0:
        sigma_y = optimize.brentq(
            lambda x: (1 + E / x * 0.033) ** n - sigma_33 / x, 1e7, 1e10
        )
    else:
        sigma_y = sigma_33
    return E, Estar, n, sigma_y, p_ave


def inverse_model_dual(Ca, Cb, theta, WpWt, dPdh, nu, hm, nu_i=0.07, E_i=1100e9):
    cstar = 1.1957

    if WpWt < Pi5(0):
        hr = 1e-9
    else:
        hr = optimize.brentq(lambda x: Pi5(x) - WpWt, 0, 1) * hm
    Pm = Ca * hm ** 2
    Am = (Pm * cstar / dPdh / Pi4(hr / hm)) ** 2
    Estar = dPdh / cstar / Am ** 0.5
    p_ave = Pm / Am
    sigma_33 = optimize.brentq(lambda x: Pi1(Estar / x) - Ca / x, 1e7, 1e10)
    for l, r in [[1e7, 1e9], [1e9, 5e9], [5e9, 1e10]]:
        if (Pitheta(theta, Estar / l) - Cb / l) * (
            Pitheta(theta, Estar / r) - Cb / r
        ) < 0:
            sigma_r = optimize.brentq(
                lambda x: Pitheta(theta, Estar / x) - Cb / x, l, r
            )
            break
    else:
        raise ValueError
    epsilon = epsilon_r(theta)
    E = (1 - nu ** 2) / (1 / Estar - (1 - nu_i ** 2) / E_i)
    if (epsilon > 0.033 and sigma_33 < sigma_r) or (
        epsilon < 0.033 and sigma_33 > sigma_r
    ):
        sigma_y = optimize.brentq(
            lambda x: np.log(sigma_33 / x) / np.log(sigma_r / x)
            - np.log(1 + E / x * 0.033) / np.log(1 + E / x * epsilon),
            1e7,
            min(sigma_33, sigma_r),
        )
        n = np.log(sigma_33 / sigma_y) / np.log(1 + E / sigma_y * 0.033)
    else:
        n = 0
        sigma_y = (sigma_33 + sigma_r) / 2
    return E, Estar, n, sigma_y, p_ave


def test_inverse():
    nu = 0.3
    hm = 0.2e-6

    # E*
    d = FEMData("E*", [70])
    # d = BerkovichData("E*")
    # d = ExpData("B3067.csv", "E*")
    y_pred = np.array(
        [inverse_model(x[0] * 1e9, x[2], x[1], nu, hm)[1] / 1e9 for x in d.X]
    )[:, None]
    ape = np.abs(y_pred - d.y) / d.y * 100
    print("E* APE:", np.mean(ape), np.std(ape))
    np.savetxt("E.dat", np.hstack((d.y, y_pred)))

    # sigma_y
    d = FEMData("sigma_y", [70])
    # d = BerkovichData("sigma_y")
    # d = ExpData("B3067.csv", "sigma_y")
    y_pred = np.array(
        [inverse_model(x[0] * 1e9, x[2], x[1], nu, hm)[3] / 1e9 for x in d.X]
    )[:, None]
    ape = np.abs(y_pred - d.y) / d.y * 100
    print("sigma_y APE:", np.mean(ape), np.std(ape))
    np.savetxt("sy.dat", np.hstack((d.y, y_pred)))

    # n
    # d = FEMData("n", [70])
    d = BerkovichData("n")
    y_pred = np.array([inverse_model(x[0] * 1e9, x[2], x[1], nu, hm)[2] for x in d.X])[
        :, None
    ]
    print(d.y)
    print(y_pred)
    ape = np.abs(y_pred - d.y) / d.y * 100
    print("n APE:", np.mean(ape), np.std(ape))
    np.savetxt("n.dat", np.hstack((d.y, y_pred)))


def test_inverse_dual():
    nu = 0.3
    hm = 0.2e-6

    # E*
    d = FEMData("E*", [70, 60])
    y_pred = np.array(
        [
            inverse_model_dual(x[0] * 1e9, x[-1] * 1e9, 60, x[2], x[1], nu, hm)[1] / 1e9
            for x in d.X
        ]
    )[:, None]
    mape = np.mean(np.abs(y_pred - d.y) / d.y) * 100
    print("E* MAPE:", mape)

    # sigma_y
    d = FEMData("sigma_y", [70, 60])
    y_pred = np.array(
        [
            inverse_model_dual(x[0] * 1e9, x[-1] * 1e9, 60, x[2], x[1], nu, hm)[3] / 1e9
            for x in d.X
        ]
    )[:, None]
    mape = np.mean(np.abs(y_pred - d.y) / d.y) * 100
    print("sigma_y MAPE:", mape)


def gen_forward():
    nu = 0.3
    hm = 0.2e-6
    nu_i = 0.07
    E_i = 1100e9
    with open("model_forward.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "E (GPa)", "sy (GPa)", "C (GPa)", "dP/dh (N/m)", "WpWt"])
        for _ in range(10000):
            E = random.uniform(10, 210)
            n = random.uniform(0, 0.5)
            # sigma_y = random.uniform(0.03, 5.3)
            sigma_y = random.uniform(0.0014, 0.04) * E
            if sigma_y < 0.03 or sigma_y > 5.3:
                continue

            Estar = 1 / ((1 - nu ** 2) / E + (1 - nu_i ** 2) / E_i)
            if n > 0.3 and sigma_y / Estar >= 0.03:
                continue

            _, C, _, dPdh, WpWt, _ = forward_model(E * 1e9, n, sigma_y * 1e9, nu, hm=hm)
            writer.writerow([n, E, sigma_y, C / 1e9, dPdh, WpWt])


def gen_inverse():
    nu = 0.3
    hm = 0.2e-6
    with open("model_inverse.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["E", "Estar", "n", "sigma_y", "C", "dPdh", "WpWt"])
        for _ in range(15000):
            C = random.uniform(2.7e3, 2.3e5) * 1e6
            dPdh = random.uniform(8.3e3, 3.4e5)
            WpWt = random.uniform(0.20, 0.98)
            try:
                E, Estar, n, sigma_y, _ = inverse_model(C, WpWt, dPdh, nu, hm)
            except:
                continue
            writer.writerow([E, Estar, n, sigma_y, C, dPdh, WpWt])


def main():
    # print(inverse_model(27.4e9, 0.902, 4768e3 * 0.2 * (27.4 / 3)**0.5 * 10**(-1.5), 0.3, 0.2e-6, nu_i=0.07, E_i=1100e9))
    # test_inverse()
    # test_inverse_dual()
    gen_forward()
    # gen_inverse()


if __name__ == "__main__":
    main()
