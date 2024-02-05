import random
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import math
from scipy.stats import norm

# simulate(n, L, e, k, gap)
#
# INPUT:
# n = number of elements
# e = comparison probability
# L = number of comparisons per pair
# k = number of top elements to separate from the rest
# gap = imposed gap between first element and rest
#
# OUTPUT:
# s_star = vector approximating ideal s
# l_inf_error = infinity norm error of the approximation vector
# D_w_error = weighted sum of out of order pairs in the approximation vector
def simulate(n, L, e, k, gap):
    s = make_s(n, k, gap)
    # normalize s
    s_norm = np.asarray(s) / sum(s)
    P = make_P(n, e, L, s)
    s_star = lp_algorithm(P, min(s_norm), max(s_norm))

    # l infinity error
    l_inf_err = max(abs(np.subtract(s_star, s_norm)))/max(s_norm)

    # D_w error
    D_w_err = 0
    for i in range(0, n):
        for j in range(i, n):
            if ((s_norm[i] - s_norm[j]) * (s_star[i] - s_star[j]) < 0):
                D_w_err += math.pow((s_norm[i] - s_norm[j]), 2)

    D_w_err = math.sqrt(D_w_err / (2*n*math.pow(np.linalg.norm(s_norm), 2)))

    return s_star, l_inf_err, D_w_err

# make_s(n, k, delta_k)
#
# INPUT:
# n = number of elements
# k = number of top elements to separate from the rest
# delta_k = imposed gap between first k and rest
#
# OUTPUT:
# s = ground truth score vector
def make_s(n, k, delta_k):
    s = [0] * n
    for i in range(0, n):
        s[i] = random.random() * 0.5 + 0.5

    s_sort = np.sort(s)

    for i in range(0, n):
        if (s[i] >= s_sort[n-k]):
            s[i] += delta_k

    return s

# make_P(n, e, L, s)
#
# INPUT:
# n = number of elements
# e = comparison probability
# L = number of comparisons per pair
# s = ground truth score vector
#
# OUTPUT:
# P_btl: synthetic data matrix of size nxn
def make_P(n, e, L, s):

    P_thu = np.zeros((n, n))

    # fill in probability matrix
    for i in range(0, n):
        for j in range(i, n):
            # diagonals (i beats i) should have probability of 1/2
            if (i==j):
                P_thu[i][j] = 1/2
            # if i and j get compared:
            elif (random.random() < e):
                # generate a number of wins according to each model's respective probability function
                wins = np.random.binomial(L,  norm.cdf(s[i] - s[j]))
                # record the probability of wins and the inverse in each matrix ([i][j] and [j][i])
                P_thu[i][j] = wins / L
                P_thu[j][i] = 1 - P_thu[i][j]
            # else: P_ij = 0
            else:
                P_thu[i][j] = 0

    return P_thu

# lp_algorithm(P, s_min, s_max)
#
# INPUT:
# P = synthetic data matrix
# s_min = minimum score in s vector
# s_max = maximum score in s vector
#
# OUTPUT:
# s_star = vector approximating ideal s
def lp_algorithm(P, s_min, s_max):

    n = P.shape[0]

    m_thu = gp.Model("Thurstone")
    # x: 1D vector (used to approximate s)
    x_thu = m_thu.addMVar((n, 1), lb=math.log(s_min), ub=math.log(s_max), vtype=GRB.CONTINUOUS, name="x")
    # z: nxn matrix, z = Y - P
    z_thu = m_thu.addMVar((n, n), lb=0.0, vtype=GRB.CONTINUOUS, name="z")

    zeroMat = np.zeros((n, n))
    eVec = np.full((n, 1), 1)
    # keep a map of nonzero entries; useful for speeding up algorithm
    mask = np.zeros((n, n))

    # apply link function to P
    P_link = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            # we assume no probabilities are zero unless they were not compared
            if (P[i][j] == 0):
                P_link[i][j] = 0
            else:
                # 1 / (1 - x) undefined
                if (P[i][j] == 1):
                    P_link[i][j] = 0
                    continue
                P_link[i][j] = norm.ppf(P[i][j])
                if (i != j):
                    mask[i][j] = 1

    # constraints
    m_thu.addConstr((x_thu @ eVec.T) - (eVec @ x_thu.T) + z_thu - P_link >= zeroMat)
    m_thu.addConstr((x_thu @ eVec.T) - (eVec @ x_thu.T) - z_thu - P_link <= zeroMat)

    # constraints include entire matrix, but we only care when link[i][j] != 0 and i != j; so apply a mask: np.multiply(z, mask)
    # this way, the optimization problem can ignore zero entries
    m_thu.setObjective((z_thu * mask).sum(), GRB.MINIMIZE)

    m_thu.optimize()

    # initialize approximation vector of ideal s
    s_star = []

    # read in values directly
    for i in range(0, n):
        s_star.append(x_thu.X[i][0])

    s_star = s_star / sum(s_star)
    return s_star