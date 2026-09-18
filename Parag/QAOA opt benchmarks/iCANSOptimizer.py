import pennylane as qml
from pennylane import numpy as np

class iCANSOptimizer:
    def __init__(self, step=0.1, L=1.0, s_min=10, N=10000, mu=0.99, b=1e-6, icans_type=1):
        """
        iCANS Optimizer for Measurement-Frugal Variational Algorithms.
        
        Args:
            alpha (float): Learning rate. Must be < 2/L.
            L (float): Lipschitz constant bound.
            s_min (int): Minimum number of shots per partial derivative estimation.
            N (int): Total number of shots allowed for the entire optimization.
            mu (float): Exponential moving average constant (0 < mu < 1).
            b (float): Small regularizing bias parameter to prevent division by zero.
            icans_type (int): 1 for iCANS1 (standard), 2 for iCANS2 (adaptive learning rate check).
        """
        self.alpha = step
        self.L = L
        self.s_min = s_min
        self.N = N
        self.mu = mu
        self.b = b
        self.icans_type = icans_type
        
        # Internal state variables
        self.s_tot = 0
        self.k = 0
        self.chi_prime = None
        self.xi_prime = None
        self.s = None

    def _iEvaluate(self, qnode, theta, s_array):
        """
        Evaluates the gradient and the variance of the gradient estimates 
        by simulating shot noise classically to bypass PennyLane's finite-shot variance limitations.
        """
        d = len(theta)
        g = np.zeros(d)
        S = np.zeros(d)
        
        for i in range(d):
            s_i = int(s_array[i])
            
            # Forward shift (+pi/2)
            theta_plus = np.copy(theta)
            theta_plus[i] += np.pi / 2
            
            # Execute exactly (shots=None bypasses the VarianceMP limitation)
            exact_expval_plus, exact_var_plus = qnode(theta_plus, shots=None)
            
            # Classically simulate the finite-shot expectation value
            expval_plus = np.random.normal(exact_expval_plus, np.sqrt(exact_var_plus / s_i))
            
            # Backward shift (-pi/2)
            theta_minus = np.copy(theta)
            theta_minus[i] -= np.pi / 2
            
            # Execute exactly
            exact_expval_minus, exact_var_minus = qnode(theta_minus, shots=None)
            
            # Classically simulate the finite-shot expectation value
            expval_minus = np.random.normal(exact_expval_minus, np.sqrt(exact_var_minus / s_i))
            
            # Gradient estimate via parameter shift rule[cite: 1, 2]
            g[i] = (expval_plus - expval_minus) / 2.0
            
            # Variance of the gradient estimate S_i
            S[i] = (exact_var_plus + exact_var_minus) / 4.0   
            
        return g, S

    def step(self, qnode, theta):
        """
        Performs one optimization step using the iCANS1/2 algorithm.
        """
        d = len(theta)
        
        # Initialize
        if self.s is None:
            self.s = np.full(d, self.s_min, dtype=float)
            self.chi_prime = np.zeros(d)
            self.xi_prime = np.zeros(d)
            
        if self.s_tot >= self.N:
            return theta

        # Evaluate gradient and variances
        g, S = self._iEvaluate(qnode, theta, self.s)
        
        # Update total shots
        self.s_tot += 2 * np.sum(self.s)
        
        # Update smoothed estimators
        self.xi_prime = self.mu * self.xi_prime + (1 - self.mu) * S
        self.chi_prime = self.mu * self.chi_prime + (1 - self.mu) * g
        xi = self.xi_prime / (1 - self.mu**(self.k + 1))
        chi = self.chi_prime / (1 - self.mu**(self.k + 1))
        
        theta_new = np.copy(theta)
        gamma = np.zeros(d)
        s_next = np.zeros(d)
        
        # Iterate over each parameter dimension
        for i in range(d):
            # iCANS1
            if self.icans_type == 1:
                theta_new[i] -= self.alpha * g[i]
                
            # iCANS2
            elif self.icans_type == 2:
                threshold = (g[i]**2) / (self.L * (g[i]**2 + S[i]/self.s[i] + self.b * self.mu**self.k))
                if self.alpha <= threshold:
                    theta_new[i] -= self.alpha * g[i]
                else:
                    alpha_prime = threshold
                    theta_new[i] -= alpha_prime * g[i]
                    
            # Compute recommended shots
            factor = (2 * self.L * self.alpha) / (2 - self.L * self.alpha)
            s_next_i = np.ceil(factor * xi[i] / (chi[i]**2 + self.b * self.mu**self.k))
            s_next[i] = s_next_i
            
            # Compute expected gain per shot
            term1 = (self.alpha - self.L * self.alpha**2 / 2) * chi[i]**2
            term2 = (self.L * self.alpha**2 / (2 * s_next_i)) * xi[i]
            gamma[i] = (1 / s_next_i) * (term1 - term2)
            
        # Cap max shots and clip
        i_max = np.argmax(gamma)
        s_max = s_next[i_max]
        self.s = np.clip(s_next, self.s_min, s_max)
        
        # Increment iteration counter
        self.k += 1
        
        return theta_new