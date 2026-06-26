import pennylane as qml
from pennylane import numpy as np
# =====================================================================
# 1. THE DecSPS OPTIMIZER IMPLEMENTATION
# =====================================================================
class DecSPS:
    def __init__(self, l_star=-3.0, gamma_b=2.0, c0=1.0):
        """
        Decreasing Stochastic Polyak Stepsize (DecSPS) tailored for QML.
        
        l_star:  A safe lower bound for the Hamiltonian (must be <= true minimum energy).
        gamma_b: Initial maximum stepsize limit (gamma_minus_1).
        c0:      Starting sequence controller (c_minus_1).
        """
        self.l_star = l_star
        self.gamma_b = gamma_b
        self.c0 = c0
        
        # Internal optimizer states
        self.k = 0
        self.gamma_prev = gamma_b
        self.c_prev = c0

    def step(self, objective_fn, params):
        # 1. Compute current cost value
        loss = objective_fn(params)
        
        # 2. Compute gradients using PennyLane's autograd engine
        grad_fn = qml.grad(objective_fn)
        gradients = grad_fn(params)
        
        # Flatten parameters/gradients to calculate the norm accurately
        grad_flat = np.hstack([np.array(g).flatten() for g in gradients])
        grad_norm_sq = np.sum(grad_flat ** 2)
        
        # Update sequence controller for step k
        c_k = np.sqrt(self.k + 1)
        
        # Safe escape if gradient is perfectly flat to prevent division by zero
        if grad_norm_sq < 1e-8:
            self.k += 1
            return params
            
        # Left side calculation: (Current Loss - Safe Baseline) / ||grad||^2
        current_batch_calc = (loss - self.l_star) / grad_norm_sq
        
        # Right side calculation: Historical Memory Anchor
        memory_anchor = self.c_prev * self.gamma_prev
        
        # Core DecSPS logic: Apply the dampening ceiling
        stepsize = (1.0 / c_k) * min(current_batch_calc, memory_anchor)
        
        # Store historical parameters for the next iteration (k + 1)
        self.c_prev = c_k
        self.gamma_prev = stepsize
        self.k += 1
        
        # FIX: Perform a clean element-wise array update to preserve shape
        new_params = params - stepsize * gradients
        return new_params


# =====================================================================
# 2. QUANTUM ARCHITECTURE DEFINITION
# =====================================================================
num_qubits = 2
num_layers = 2
dev = qml.device("default.qubit", wires=num_qubits)

# Task Layout: Rx, Ry then linear CNOT gates per layer
def ansatz(p):
    for layer in range(num_layers):
        # Rotation Sub-Layer
        for qubit in range(num_qubits):
            qml.RX(p[layer, qubit, 0], wires=qubit)
            qml.RY(p[layer, qubit, 1], wires=qubit)
        
        # Linear Entangling CNOT Sub-Layer
        for qubit in range(num_qubits - 1):
            qml.CNOT(wires=[qubit, qubit + 1])

# Objective Function: H = - sum(Z_i Z_{i+1}) - sum(X_i)
obs = []
coeffs = []
for i in range(num_qubits - 1):
    obs.append(qml.PauliZ(i) @ qml.PauliZ(i + 1))
    coeffs.append(-1.0)
for i in range(num_qubits):
    obs.append(qml.PauliX(i))
    coeffs.append(-1.0)

Hamiltonian = qml.Hamiltonian(coeffs, obs)

@qml.qnode(dev)
def cost_function(params):
    ansatz(params)
    return qml.expval(Hamiltonian)


# =====================================================================
# 3. BENCHMARK COMPARISON EXECUTION
# =====================================================================
# Generate baseline initial parameter weights
np.random.seed(42)
init_params = np.random.uniform(0, 2 * np.pi, size=(num_layers, num_qubits, 2), requires_grad=True)

# Exact classical ground state calculation for tracking reference
H_matrix = qml.matrix(Hamiltonian)
true_optimal_f = float(min(np.linalg.eigvalsh(H_matrix)))
print(f"Target Theoretical Ground Energy: {true_optimal_f:.6f}\n")

# Run DecSPS
decsps_params = np.array(init_params.copy(), requires_grad=True)
decsps_optimizer = DecSPS(l_star=-3.0, gamma_b=2.0)

print("--- Running DecSPS Optimization ---")
for step in range(101):
    decsps_params = decsps_optimizer.step(cost_function, decsps_params)
    if step % 20 == 0:
        print(f"Iteration {step:3d} | Current Energy Cost: {cost_function(decsps_params):.6f}")

# Run Standard Adam for baseline verification
adam_params = np.array(init_params.copy(), requires_grad=True)
adam_optimizer = qml.AdamOptimizer(stepsize=0.1)

print("\n--- Running Adam Baseline ---")
for step in range(101):
    adam_params = adam_optimizer.step(cost_function, adam_params)
    if step % 20 == 0:
        print(f"Iteration {step:3d} | Current Energy Cost: {cost_function(adam_params):.6f}")