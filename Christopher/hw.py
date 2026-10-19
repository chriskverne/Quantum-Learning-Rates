# import numpy as np
# import matplotlib.pyplot as plt

# X1 = np.random.exponential(1, 500)
# X2 = np.random.exponential(1, 500)
# mn, mx = np.minimum(X1, X2), np.maximum(X1, X2)
# R = mx - mn
# D = X1 - X2


# plt.scatter(mn, X1)
# plt.xlabel("min(X1, X2)"); plt.ylabel("X1")
# plt.show()

# plt.scatter(D, mx)
# plt.xlabel("X1 - X2"); plt.ylabel("max(X1, X2)")
# plt.show()




import numpy as np

for n in (12, 25):
    i, j = np.indices((n, n)) + 1                            # 1-based row/col indices
    A = np.where(i <= j + 1, n + 1 - np.maximum(i, j), 0)    # build A from the a_ij rule
    rho = np.max(np.abs(np.linalg.eigvals(A)))               # spectral radius
    two = np.linalg.norm(A, 2)                               # largest singular value
    print(f"n={n}: rho(A) = {rho:.6f}, ||A||_2 = {two:.6f}")