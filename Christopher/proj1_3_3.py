import numpy as np
import itertools

np.random.seed(0)

u_star = np.array([[0, .5, 0, .5, 0],
                   [.5, 0, .5, 0, .5],
                   [0, .5, 0, .5, 0],
                   [.5, 0, .5, 0, 0],
                   [0, .5, 0, 0, 0]])

def gibbs(u, y0, burn_in, N):
    y = y0.copy()
    samples = []
    for n in range(burn_in + N):
        for r in range(len(y)):
            S = u[r] @ y
            y[r] = 1 if np.random.rand() < 1 / (1 + np.exp(-2 * S)) else -1
        if n >= burn_in:
            samples.append(y.copy())
    return np.array(samples)

Y = gibbs(u_star, np.array([1, -1, -1, 1, 1]), 5000, 1000)
N, p = Y.shape
iu = np.triu_indices(p, 1)
states = np.array(list(itertools.product([-1, 1], repeat=p)))
T = (Y.T @ Y)[iu]
Ts = np.einsum("ni,nj->nij", states, states)[:, iu[0], iu[1]]
lam, burn_in, N_bar = 0.2, 10000, 1000

def log_post(ut):
    logZ = np.log(np.exp(Ts @ ut).sum())
    return ut @ T - N * logZ - np.abs(ut).sum() / lam

def mh(sigma2):
    ut = np.zeros(len(T))
    lp = log_post(ut)
    samples, accepted = [], 0
    for n in range(burn_in + N_bar):
        v = ut + np.sqrt(sigma2) * np.random.randn(len(ut))
        lp_v = log_post(v)
        if np.log(np.random.rand()) < lp_v - lp:
            ut, lp = v, lp_v
            accepted += 1
        if n >= burn_in:
            samples.append(ut.copy())
    return np.array(samples), accepted / (burn_in + N_bar)

def to_matrix(x):
    U = np.zeros((p, p))
    U[iu] = x
    return U + U.T

for sigma2 in [0.1, 0.001]:
    samples, acc = mh(sigma2)
    print(f"sigma^2 = {sigma2}, acceptance rate = {acc:.4f}")
    print("Mean:\n", to_matrix(samples.mean(axis=0)).round(3))
    print("Variance:\n", to_matrix(samples.var(axis=0, ddof=1)).round(4))