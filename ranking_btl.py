import random
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import math

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
# w_star = vector approximating ideal w
# l_inf_error = infinity norm error of the approximation vector
# D_w_error = weighted sum of out of order pairs in the approximation vector
def simulate(n, L, e, k, gap):
    w = make_w(n, k, gap)
    # normalize w
    w_norm = np.asarray(w) / sum(w)
    P = make_P(n, e, L, w)
    w_star = lp_algorithm(P, min(w_norm), max(w_norm))

    # l infinity error
    l_inf_err = max(abs(np.subtract(w_star, w_norm)))/max(w_norm)

    # D_w error
    D_w_err = 0
    for i in range(0, n):
        for j in range(i, n):
            if ((w_norm[i] - w_norm[j]) * (w_star[i] - w_star[j]) < 0):
                D_w_err += math.pow((w_norm[i] - w_norm[j]), 2)

    D_w_err = math.sqrt(D_w_err / (2*n*math.pow(np.linalg.norm(w_norm), 2)))

    return w_star, l_inf_err, D_w_err

# make_w(n, k, delta_k)
#
# INPUT:
# n = number of elements
# k = number of top elements to separate from the rest
# delta_k = imposed gap between first k and rest
#
# OUTPUT:
# w = ground truth score vector
def make_w(n, k, delta_k):
    w = [0] * n
    for i in range(0, n):
        w[i] = random.random() * 0.5 + 0.5

    w_sort = np.sort(w)

    for i in range(0, n):
        if (w[i] >= w_sort[n-k]):
            w[i] += delta_k

    return w

# make_P(n, e, L, w)
#
# INPUT:
# n = number of elements
# e = comparison probability
# L = number of comparisons per pair
# w = ground truth score vector
#
# OUTPUT:
# P_btl: synthetic data matrix of size nxn
def make_P(n, e, L, w):

    P_btl = np.zeros((n, n))

    # fill in probability matrix
    for i in range(0, n):
        for j in range(i, n):
            # diagonals (i beats i) should have probability of 1/2
            if (i==j):
                P_btl[i][j] = 1/2
            # if i and j get compared:
            elif (random.random() < e):
                # generate a number of wins according to each model's respective probability function
                wins = np.random.binomial(L, (w[i] / (w[i] + w[j])))
                # record the probability of wins and the inverse in each matrix ([i][j] and [j][i])
                P_btl[i][j] = wins / L
                P_btl[j][i] = 1 - P_btl[i][j]
            # else: P_ij = 0
            else:
                P_btl[i][j] = 0

    return P_btl         

# lp_algorithm(P, w_min, w_max)
#
# INPUT:
# P = synthetic data matrix
# w_min = minimum score in w vector
# w_max = maximum score in w vector
#
# OUTPUT:
# w_star = vector approximating ideal w
def lp_algorithm(P, w_min, w_max):

    n = P.shape[0]

    m_btl = gp.Model("BTL")
    # x: 1D vector (used to approximate s)
    x_btl = m_btl.addMVar((n, 1), lb=math.log(w_min), ub=math.log(w_max), vtype=GRB.CONTINUOUS, name="x")
    # z: nxn matrix, z = Y - P
    z_btl = m_btl.addMVar((n, n), lb=0.0, vtype=GRB.CONTINUOUS, name="z")

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
                P_link[i][j] = math.log(P[i][j] / (1 - P[i][j]))
                if (i != j):
                    mask[i][j] = 1

    # constraints
    m_btl.addConstr((x_btl @ eVec.T) - (eVec @ x_btl.T) + z_btl - P_link >= zeroMat)
    m_btl.addConstr((x_btl @ eVec.T) - (eVec @ x_btl.T) - z_btl - P_link <= zeroMat)

    # constraints include entire matrix, but we only care when link[i][j] != 0 and i != j; so apply a mask: np.multiply(z, mask)
    # this way, the optimization problem can ignore zero entries
    m_btl.setObjective((z_btl * mask).sum(), GRB.MINIMIZE)

    m_btl.optimize()

    # initialize approximation vector of ideal w
    w_star = []

    #undo log ->   x = log(w) => w = 2^x
    for i in range(0, n):
        w_star.append(pow(math.e, x_btl.X[i])[0])

    w_star = w_star / sum(w_star)
    return w_star