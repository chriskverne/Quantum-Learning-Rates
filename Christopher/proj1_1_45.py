import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate  data
np.random.seed(42)
N = 1000
data = np.random.multivariate_normal(mean=[0, 0], cov=np.eye(2), size=N)
y = data[:, 0]
z = data[:, 1]

S_yy = np.sum(y**2)
S_zz = np.sum(z**2)
S_yz = np.sum(y * z)

def log_posterior(u):
    """Computes log P(u | data) up to an additive constant."""
    if np.abs(u) >= 1.0:
        return -np.inf
    log_prior_lik_norm = -0.5 * (N + 1) * np.log(1.0 - u**2)
    quad_form = -(S_yy - 2.0 * u * S_yz + S_zz) / (2.0 * (1.0 - u**2))
    return log_prior_lik_norm + quad_form


# Step 2: Random Walk Metropolis-Hastings (Burn-in)
burn_in = 10000
delta = 0.4

u_current = 0.1
m_running = np.zeros(burn_in)
c_running = np.zeros(burn_in)
m = u_current
c = 0.0

for n in range(1, burn_in + 1):
    # sample v
    v_star = np.random.uniform(u_current - delta, u_current + delta)
    
    # Compute acceptance probability
    log_alpha = log_posterior(v_star) - log_posterior(u_current)
    
    if np.log(np.random.uniform(0, 1)) < log_alpha:
        u_current = v_star  # Accept
    # Else: u_current stays the same (Reject)
    
    # Online updates
    if n == 1:
        m = u_current
        c = 0.0
    else:
        m_prev = m
        m = ((n - 1) * m_prev + u_current) / n
        c = ((n - 2) * c + (u_current - m)**2) / (n - 1)
        
    m_running[n - 1] = m
    c_running[n - 1] = c

# Step 3: Draw post-burn-in samples
post_samples = np.zeros(N)
for i in range(N):
    v_star = np.random.uniform(u_current - delta, u_current + delta)
    log_alpha = log_posterior(v_star) - log_posterior(u_current)
    if np.log(np.random.uniform(0, 1)) < log_alpha:
        u_current = v_star
    post_samples[i] = u_current

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

ax1.plot(m_running, color='tab:blue', lw=1.5)
ax1.set_ylabel("Running Mean $m^{(n)}$")
ax1.set_title("Burn-in Convergence Diagnostics (Step Size = 0.4)")
ax1.grid(True, alpha=0.3)

ax2.plot(c_running, color='tab:orange', lw=1.5)
ax2.set_xlabel("Iteration $n$")
ax2.set_ylabel("Running Variance $c^{(n)}$")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()