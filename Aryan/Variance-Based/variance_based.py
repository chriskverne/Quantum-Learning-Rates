import os
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# 1. Global Configurations
# =====================================================================
qubit_sizes = [2, 4, 6]  # Qubit counts to benchmark (Supports 2 to 10)
num_layers = 2           # Depth of the hardware-efficient ansatz
num_steps = 80           # Optimization steps per run

# Fixed learning rates for standard optimizers
lr_sgd = 0.1
lr_adam = 0.1
lr_qng = 0.05

# Output directory for exported figures
output_dir = "vqe_benchmarks"
os.makedirs(output_dir, exist_ok=True)

# =====================================================================
# 2. Optimization Routines
# =====================================================================

def run_variance_based(params_init, num_qubits, steps, expval_qnode, expval_sq_qnode):
    """Optimizes VQE using the adaptive Variance-Based Step Size."""
    params = np.array(params_init, requires_grad=True)
    energy_history = []
    grad_fn = qml.grad(expval_qnode)
    
    for _ in range(steps):
        energy = expval_qnode(params)
        energy_sq = expval_sq_qnode(params)
        variance = max(0.0, energy_sq - (energy ** 2))
        
        grad = grad_fn(params)
        grad_norm_sq = np.sum(grad ** 2)
        
        # eta = sqrt(Var(H)) / ||grad||^2
        if grad_norm_sq > 1e-8:
            eta = np.sqrt(variance) / grad_norm_sq
        else:
            eta = 1e-3  # Soft fallback
            
        params = params - eta * grad
        energy_history.append(energy)
        
    return energy_history


def run_sgd(params_init, steps, expval_qnode, lr):
    """Optimizes VQE using standard Gradient Descent (SGD)."""
    params = np.array(params_init, requires_grad=True)
    opt = qml.GradientDescentOptimizer(stepsize=lr)
    energy_history = []
    
    for _ in range(steps):
        params, energy = opt.step_and_cost(expval_qnode, params)
        energy_history.append(energy)
        
    return energy_history


def run_adam(params_init, steps, expval_qnode, lr):
    """Optimizes VQE using the classical Adam Optimizer."""
    params = np.array(params_init, requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=lr)
    energy_history = []
    
    for _ in range(steps):
        params, energy = opt.step_and_cost(expval_qnode, params)
        energy_history.append(energy)
        
    return energy_history


def run_qng(params_init, steps, expval_qnode, lr):
    """Optimizes VQE using Quantum Natural Gradient (QNG)."""
    params = np.array(params_init, requires_grad=True)
    opt = qml.QNGOptimizer(stepsize=lr)
    energy_history = []
    
    for _ in range(steps):
        # QNG dynamically calculates the Fubini-Study metric tensor
        params, energy = opt.step_and_cost(expval_qnode, params)
        energy_history.append(energy)
        
    return energy_history

# =====================================================================
# 3. Hardware-Efficient Ansatz
# =====================================================================

def hardware_efficient_ansatz(params, num_qubits, layers):
    """Constructs a Hardware-Efficient Ansatz (HEA)."""
    params = np.reshape(params, (layers, num_qubits, 2))
    for layer in range(layers):
        for i in range(num_qubits):
            qml.RY(params[layer, i, 0], wires=i)
            qml.RZ(params[layer, i, 1], wires=i)
        for i in range(num_qubits - 1):
            qml.CNOT(wires=[i, i + 1])

# =====================================================================
# 4. Main Multi-Qubit Benchmarking Loop
# =====================================================================

for n_qubits in qubit_sizes:
    print(f"\n==========================================")
    print(f"RUNNING BENCHMARK FOR {n_qubits} QUBITS")
    print(f"==========================================")
    
    # --- 4a. Construct Hamiltonian (H and H^2) ---
    obs_list = []
    coeffs_list = []
    
    # Z_i Z_{i+1} term (periodic boundaries)
    for i in range(n_qubits):
        next_qubit = (i + 1) % n_qubits
        obs_list.append(qml.PauliZ(i) @ qml.PauliZ(next_qubit))
        coeffs_list.append(-1.0)
        
    # X_i term
    for i in range(n_qubits):
        obs_list.append(qml.PauliX(i))
        coeffs_list.append(-1.0)
        
    # Algebraically compute H^2 safely
    op_terms = [qml.s_prod(c, o) for c, o in zip(coeffs_list, obs_list)]
    H_operator = qml.sum(*op_terms)
    H_squared_operator = qml.simplify(H_operator @ H_operator)
    hq_coeffs, hq_ops = H_squared_operator.terms()
    
    H = qml.Hamiltonian(coeffs_list, obs_list)
    H_squared = qml.Hamiltonian(hq_coeffs, hq_ops)
    
    # --- 4b. Setup Simulator & QNodes ---
    dev = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(dev)
    def expval_H_qnode(params):
        hardware_efficient_ansatz(params, n_qubits, num_layers)
        return qml.expval(H)
        
    @qml.qnode(dev)
    def expval_H_squared_qnode(params):
        hardware_efficient_ansatz(params, n_qubits, num_layers)
        return qml.expval(H_squared)
        
    # --- 4c. Initialize Parameters (Shared for a fair test) ---
    total_params = num_layers * n_qubits * 2
    np.random.seed(101)  # Fixed seed per qubit run
    init_params = np.random.uniform(0, 2 * np.pi, total_params)
    
    # --- 4d. Compute Exact Ground State Energy classically ---
    h_matrix = qml.matrix(H)
    exact_eigvals = np.linalg.eigvalsh(h_matrix)
    exact_ground_energy = exact_eigvals[0]
    print(f"Exact Ground State Energy: {exact_ground_energy:.6f}")
    
    # --- 4e. Run Optimizations ---
    print("Running Variance-Based optimizer...")
    hist_var = run_variance_based(init_params, n_qubits, num_steps, expval_H_qnode, expval_H_squared_qnode)
    
    print("Running SGD optimizer...")
    hist_sgd = run_sgd(init_params, num_steps, expval_H_qnode, lr_sgd)
    
    print("Running Adam optimizer...")
    hist_adam = run_adam(init_params, num_steps, expval_H_qnode, lr_adam)
    
    print("Running Quantum Natural Gradient (QNG)...")
    hist_qng = run_qng(init_params, num_steps, expval_H_qnode, lr_qng)
    
    # --- 4f. Plot and Automatically Export ---
    plt.figure(figsize=(10, 6))
    plt.plot(hist_var, label="Variance-Based Step Size", color="crimson", lw=2.5, zorder=4)
    plt.plot(hist_sgd, label=f"SGD (lr={lr_sgd})", color="gray", lw=1.5, linestyle="--")
    plt.plot(hist_adam, label=f"Adam (lr={lr_adam})", color="royalblue", lw=2, linestyle="-.")
    plt.plot(hist_qng, label=f"QNG (lr={lr_qng})", color="forestgreen", lw=2, linestyle=":")
    
    plt.axhline(exact_ground_energy, color="black", linestyle="-", alpha=0.7, label="Exact Ground Energy")
    
    plt.title(f"VQE Optimization Comparison ({n_qubits} Qubits)", fontsize=14, fontweight="bold")
    plt.xlabel("Optimization Step", fontsize=12)
    plt.ylabel("Energy $\\langle H \\rangle$", fontsize=12)
    plt.legend(fontsize=10, loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.6)
    
    # Save fig dynamically based on qubit size
    filename = os.path.join(output_dir, f"vqe_comparison_{n_qubits}_qubits.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close() # Closes graph memory immediately to run smoothly in headless environments
    print(f"Successfully saved performance graph to: {filename}")

print("\nAll benchmarks finished! Check the 'vqe_benchmarks' folder for the generated plots.")