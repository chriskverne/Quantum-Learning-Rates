import pennylane as qml
from pennylane import numpy as np

class DecSPS:
    def __init__(self, l_star=-3.0, gamma_b=2.0, c0=1.0):
        self.l_star = l_star
        self.gamma_b = gamma_b
        self.c0 = c0
        self.k = 0
        self.gamma_prev = gamma_b
        self.c_prev = c0

    def step(self, objective_fn, params):
        loss = objective_fn(params)
        gradients = qml.grad(objective_fn)(params)
        
        grad_flat = np.hstack([np.array(g).flatten() for g in gradients])
        grad_norm_sq = np.sum(grad_flat ** 2)
        c_k = np.sqrt(self.k + 1)
        
        if grad_norm_sq < 1e-8:
            self.k += 1
            return params
            
        current_batch_calc = (loss - self.l_star) / grad_norm_sq
        memory_anchor = self.c_prev * self.gamma_prev
        
        stepsize = (1.0 / c_k) * min(current_batch_calc, memory_anchor)
        
        self.c_prev = c_k
        self.gamma_prev = stepsize
        self.k += 1
        
        return params - stepsize * gradients