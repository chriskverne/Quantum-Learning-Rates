import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

u = np.array([[0, .5, 0, .5, 0],
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

Y = gibbs(u, np.array([1, -1, -1, 1, 1]), 5000, 1000)
np.save("ising_samples.npy", Y)

print("Mean:", Y.mean(axis=0))
print("Variance:", Y.var(axis=0, ddof=1))

C = np.corrcoef(Y.T)
print("Correlation:\n", C.round(3))

plt.imshow(C, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")
plt.xticks(range(5), [f"$y_{i+1}$" for i in range(5)])
plt.yticks(range(5), [f"$y_{i+1}$" for i in range(5)])
for i in range(5):
    for j in range(5):
        plt.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center")
plt.title("Sample Correlation Matrix")
# plt.savefig("ising_correlation.png", dpi=200, bbox_inches="tight")
plt.show()