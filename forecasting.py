# footprint forecasting

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sstats
import scipy.special as sc
from scipy.integrate import quad_vec
from itertools import accumulate
from time import time

import multiprocessing as mp

from suppressedBD import *

import warnings

warnings.filterwarnings("ignore")

#######################
# fast ftpt evaluation
#######################

# continuous measure of orthogonality for firewalk polynomials
def arrcontMeasure(x, b, g):  
    if b == 1:
        prefactor = G(g / 2 + 2) / (np.sqrt(np.pi) * G((g + 3) / 2))
        xpart = (1 - x**2) ** ((g + 1) / 2)
        return prefactor * xpart
    else:
        prefactor = (b + g + 1) / (2 * np.pi * 1j)
        uvpart = (
            rtU(x, b) ** (cfA(x, b, g))
            * rtV(x, b) ** (cfB(x, b, g))
            * (rtU(x, b) - rtV(x, b)) ** (-1 * cfA(x, b, g))
            / ((rtV(x, b) - rtU(x, b)) ** (cfB(x, b, g) + 1))
        )
        betapart = G(-1 * cfA(x, b, g)) * G(-1 * cfB(x, b, g)) / G(g + 2)
        return np.real(prefactor * uvpart * betapart)


def fast_firewalkW(n, x, b, g):  # recursive evaluation of polynomials
    n = int(n)
    if isinstance(x, np.ndarray):
        wm1 = np.ones(x.shape)
        wp1 = np.zeros(x.shape)
    else:
        wm1 = 1
        wp1 = 1
    wc = (b + g + 1) / b * x
    k = 1

    while k < n:
        wp1 = (
            x * ((b + 1) * (k + 1) + g) / (b * (k + 1)) * wc
            - (k + 1 + g) / (b * (k + 1)) * wm1
        )
        wm1 = wc
        wc = wp1
        k += 1
    return wc


def fast_probF_inf(F, N, beta, gamma, F0=None, kmax=50):  # fast
    if F0 == None:
        F0 = N
    if F < F0:
        return 0
    else:
        prf = 1 / asymAbsorb(N, beta, gamma) * (gamma + 1) / (beta + gamma + 1)

        expon = 2 * (F - F0) + N - 1

        integrand = (
            lambda x: x ** (expon)
            * fast_firewalkW(N - 1, x, beta, gamma)
            * arrcontMeasure(x, beta, gamma)
        )
        ib = Ib(beta)
        res, err = quad_vec(integrand, -ib, ib)
        cont_part = prf * res

        if beta <= 1:
            return np.real(cont_part)
        else:
            func = lambda x: x ** (expon) * fast_firewalkW(N - 1, x, beta, gamma)
            kr = np.arange(kmax + 1)
            xvs = atomX(kr, beta, gamma)
            weights = weightD(kr, beta, gamma)
            summed1 = np.dot(func(xvs), weights)
            summed2 = np.dot(func(-1 * xvs), weights)
            return np.real(cont_part) + prf * np.real(summed1 + summed2)


def fast_ftptIntegrate(func, b, g, kmax=50):  # array footprint measure integration
    integrand = lambda x: func(x) * arrcontMeasure(x, b, g)
    ib = Ib(b)
    res, err = quadrature(integrand, -ib, ib, tol=1e-80)
    if b <= 1:
        return np.real(res)
    else:
        kr = np.arange(kmax + 1)
        xvs = atomX(kr, b, g)
        weights = weightD(kr, b, g)
        summed1 = np.dot(func(xvs), weights)
        summed2 = np.dot(func(-1 * xvs), weights)
        return np.real(res) + np.real(summed1 + summed2)


def physP(i, j, t, b, g):  # population distribution, in physical units
    if j == 0:
        return absorbProb(t, i, b, g)
    else:
        return P(i - 1, j - 1, t, b, g)


def exact_avgJ(t, N, b, g):  # average population, exact expr (only for integral N!)
    expFunc = np.exp((b - 1) * t)
    zt = z(t, b)
    if g == 0:
        summed = 0
    else:
        summed = np.sum([sc.betainc(n, g, zt) for n in range(1, N + 1)])
    return expFunc * (N - summed)


def avgJ(t, N, b, g):  # only valid for integer N
    expFunc = np.exp((b - 1) * t)
    zt = z(t, b)
    if g == 0:
        summed = 0
    else:
        summed = np.sum([sc.betainc(n, g, zt) for n in range(1, int(N) + 1)])
    return expFunc * (N - summed)


def asymAvgF(N, b, g, F0):  # asymptotic average footprint
    integrand = lambda x: x ** (N + 1) / (1 - x**2) ** 2 * firewalkW(N - 1, x, b, g)
    normalization = (g + 1) / (b + g + 1) / asymAbsorb(N, b, g)
    return F0 + (normalization * ftptIntegrate(integrand, b, g))


def BasicForecast(T, N, b, g, F0=None, init_dict=None, zeroB=False):
    if F0 == None:
        F0 = N

    if init_dict == None:
        init_dict = {"j": N, "F": F0, "oj": 0, "oF": 0, "cov": 0}
    # calibrate number of steps
    tscale = 1 / ((b + 1) * N + g) * 1 / 10
    Ns = int(T / tscale)
    tr = np.linspace(0, T, Ns)
    dt = T / Ns

    # compute averages
    absorbs = np.array(absorbProb(tr, N, b, g))  # absorption probabilities
    avgJs = np.array([avgJ(t, N, b, g) for t in tr])  # average pops

    avgFs = F0 + np.array(list(accumulate(b * dt * avgJs)))  # average footys
    forecast = {"t": tr, "abs": absorbs, "j": avgJs, "F": avgFs}
    asymF = asymAbsorb(N, b, g)
    asymA = asymAbsorb(N, b, g)
    dabs = np.gradient(absorbs, tr)

    # compute population variance
    oj_prefactor = np.exp(2 * (b - 1) * tr)
    oj_integrand = np.exp(-2 * (b - 1) * tr) * (
        (1 + b) * avgJs + g * (1 - absorbs - 2 * absorbs * avgJs)
    )
    oj_integral = np.array(list(accumulate(dt * oj_integrand)))
    ojs = oj_prefactor * (init_dict["oj"] + oj_integral)
    forecast["oj"] = ojs

    # compute pop/foot covariance
    cov_prefactor = np.exp((b - 1) * tr)
    davgJs = np.gradient(avgJs, tr)
    if g != 0:

        if not zeroB:
            Bterm = (avgFs - b * avgJs) * absorbs
        else:
            Bterm = 0

        gf = Bterm - (avgFs) * absorbs

        cov_integrand = np.exp(-1 * (b - 1) * tr) * (b * avgJs + b * ojs + g * gf)
    else:
        cov_integrand = np.exp(-1 * (b - 1) * tr) * (b * avgJs + b * ojs)
    cov_integral = np.array(list(accumulate(dt * cov_integrand)))
    cov = cov_prefactor * (init_dict["cov"] + cov_integral)
    forecast["cov"] = cov

    # compute footprint variance
    oF_integrand = 2 * b * cov + b * avgJs
    oF = np.array(list(accumulate(dt * oF_integrand)))
    forecast["oF"] = init_dict["oF"] + oF

    return forecast


def FtptForecast(T, b, g, N, F0=None, Fmax=500, normalize=False, force=None):
    if F0 == None:
        F0 = N
    # compute forecast
    if force != None:
        genForecast = force
    else:
        genForecast = BasicForecast(T, N, b, g, F0=F0)

    tarr = genForecast["t"]
    absorbs = genForecast["abs"]
    asymA = asymAbsorb(N, b, g)

    # compute parameters of gamma dist
    avgF = genForecast["F"] - F0
    varF = genForecast["oF"]

    epsilon = 0
    gammaShape = avgF**2 / (varF + epsilon)
    gammaScale = (varF + epsilon) / avgF

    fvals = np.arange(Fmax)
    asymF = np.array([fast_probF_inf(f, N, b, g, F0=F0) for f in tqdm(fvals)])

    f_dists = []
    for i in range(len(tarr)):
        gamma_part = sstats.gamma.pdf(fvals, gammaShape[i], F0 - 0.5, gammaScale[i])
        gamma_part[:F0] = np.zeros(F0)

        gamma_part = gamma_part / sum(gamma_part)

        absRatio = absorbs[i] / asymA

        dist = absRatio * asymF + (1 - absRatio) * gamma_part

        dist[dist < 1e-12] = 0

        if normalize:
            dist = dist / sum(dist[dist < np.inf])

        f_dists.append(dist)

    return tarr, fvals, f_dists


def generateHistories(tarr, beta, gamma, init, Nsims=10000):
    # Nsims = 10000
    J = 1e6
    proc = BDensemble(beta, 1, gamma, init, J, Nsims)
    Tmax = tarr[-1]
    F_functions = []
    j_functions = []
    maxt = 0
    for n in tqdm(range(Nsims)):
        traj = proc.run_finite_trajectory(Tmax)
        jj, ff, tt = traj
        f = interp1d(tt, ff, kind="zero", bounds_error=False, fill_value=ff[-1])
        F_functions.append(f)

        j = interp1d(tt, jj, kind="zero", bounds_error=False, fill_value=jj[-1])
        j_functions.append(j)

    return F_functions, j_functions


if __name__ == "__main__":

    tarr, fvs, fdist = FtptForecast(3, 1.2, 0.00001, 5, Fmax=50)

    print(fdist[-1])
