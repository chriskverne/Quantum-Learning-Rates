import os
import json
import pennylane as qml
from pennylane import numpy as np
import numpy as std_np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 1. Setup Results Directory
results_dir = "./results"
os.makedirs(results_dir, exist_ok=True)

# 2. System & Hamiltonian Configuration
num_qubits = 6
h_field = 1.0

coeffs = []
observables = []
for i in range(num_qubits - 1):
    coeffs.append(-1.0)
    observables.append(qml.PauliZ(i) @ qml.PauliZ(i + 1))
for i in range(num_qubits):
    coeffs.append(-h_field)
    observables.append(qml.PauliX(i))

H = qml.Hamiltonian(coeffs, observables)

# Compute exact matrix and ground state energy lambda_0
H_mat = std_np.real_if_close(qml.matrix(H, wire_order=range(num_qubits)))
eigenvalues = std_np.linalg.eigvalsh(H_mat)
lambda_0 = float(eigenvalues[0])

# Precompute H^2 matrix to evaluate energy variance <H^2> - <H>^2 efficiently
H2_mat = H_mat @ H_mat
H2_obs = qml.Hermitian(H2_mat, wires=range(num_qubits))

print(f"Exact Ground State Energy (lambda_0): {lambda_0:.6f}")

# 3. Circuit Factory
dev = qml.device("default.qubit", wires=num_qubits)

def create_qnodes(num_layers):
    def ansatz(params):
        for layer in range(num_layers):
            for q in range(num_qubits):
                qml.RX(params[layer, q, 0], wires=q)
                qml.RY(params[layer, q, 1], wires=q)
            for q in range(num_qubits - 1):
                qml.CNOT(wires=[q, q + 1])

    @qml.qnode(dev, interface="autograd")
    def energy_qnode(params):
        ansatz(params)
        return qml.expval(H)

    @qml.qnode(dev, interface="autograd")
    def h2_qnode(params):
        ansatz(params)
        return qml.expval(H2_obs)

    return energy_qnode, h2_qnode


def find_ansatz_e_star(energy_qnode, num_layers, lambda_0, num_restarts=3):
    """
    Finds the minimum energy E* achievable by the ansatz with `num_layers`
    using L-BFGS-B. For sufficient layers, E* converges to lambda_0.
    """
    shape = (num_layers, num_qubits, 2)
    grad_fn = qml.grad(energy_qnode)

    def loss_and_grad(flat_params):
        p = np.tensor(flat_params.reshape(shape), requires_grad=True)
        e = float(energy_qnode(p))
        g = std_np.array(grad_fn(p)).flatten()
        return e, g

    best_e = float("inf")
    for seed in range(num_restarts):
        rng = std_np.random.default_rng(seed + 100)
        init_flat = rng.uniform(0, 2 * std_np.pi, size=std_np.prod(shape))
        res = minimize(loss_and_grad, init_flat, method="L-BFGS-B", jac=True, options={"maxiter": 500, "ftol": 1e-12})
        if res.fun < best_e:
            best_e = float(res.fun)
        if abs(best_e - lambda_0) < 1e-6:
            break

    # Ensure numerical precision doesn't dip below exact lambda_0
    return max(best_e, lambda_0)


# 4. Run Layer Ablation Experiment
layer_ablation_list = [2, 4, 6, 8, 10]
num_steps = 400
lr = 0.01
eps = 1e-12  # Prevent division by zero near convergence

all_results = {
    "metadata": {
        "num_qubits": num_qubits,
        "h_field": h_field,
        "lambda_0": lambda_0,
        "learning_rate": lr,
        "num_steps": num_steps,
        "layers_ablation": layer_ablation_list,
    },
    "runs": {}
}

for num_layers in layer_ablation_list:
    print(f"\n--- Running VQE for num_layers = {num_layers} ---")
    energy_qnode, h2_qnode = create_qnodes(num_layers)
    grad_fn = qml.grad(energy_qnode)

    # Determine empirical E* for this circuit depth
    e_star_ansatz = find_ansatz_e_star(energy_qnode, num_layers, lambda_0)
    # When enough layers are added, E* = lambda_0.
    # Here we use lambda_0 as E* directly (and also store e_star_ansatz in JSON)
    e_star = lambda_0
    print(f"Ansatz E*: {e_star_ansatz:.6f} | Gap (E* - lambda_0): {e_star_ansatz - lambda_0:.2e}")

    # Initialize parameters deterministically
    np.random.seed(42)
    params = np.random.uniform(0, np.pi, size=(num_layers, num_qubits, 2), requires_grad=True)

    energies = []
    polyak_steps = []
    weinstein_steps = []
    ratios = []
    grad_norms_sq = []
    std_devs = []

    for step in range(num_steps):
        # Compute Current Energy and Gradient
        current_e = float(energy_qnode(params))
        grads = grad_fn(params)
        grad_norm_sq = float(np.sum(grads ** 2))

        # Compute <H^2> and Weinstein standard deviation sigma = sqrt(<H^2> - <H>^2)
        current_h2 = float(h2_qnode(params))
        variance = max(0.0, current_h2 - current_e ** 2)
        sigma = float(std_np.sqrt(variance))

        # Compute exact Polyak step and Weinstein step
        polyak_step = (current_e - e_star) / (grad_norm_sq + eps)
        weinstein_step = sigma / (grad_norm_sq + eps)
        ratio = weinstein_step / (polyak_step + eps)

        # Record metrics
        energies.append(current_e)
        polyak_steps.append(float(polyak_step))
        weinstein_steps.append(float(weinstein_step))
        ratios.append(float(ratio))
        grad_norms_sq.append(grad_norm_sq)
        std_devs.append(sigma)

        # Standard SGD update
        params = params - lr * grads

        if step % 100 == 0 or step == num_steps - 1:
            print(
                f"Step {step:3d} | E: {current_e:.5f} | "
                f"Polyak: {polyak_step:.4f} | Weinstein: {weinstein_step:.4f} | "
                f"Ratio (W/P): {ratio:.4f}"
            )

    all_results["runs"][str(num_layers)] = {
        "num_layers": num_layers,
        "e_star_ansatz": e_star_ansatz,
        "e_star_used": e_star,
        "energy": energies,
        "polyak_step": polyak_steps,
        "weinstein_step": weinstein_steps,
        "ratio": ratios,
        "grad_norm_sq": grad_norms_sq,
        "energy_std": std_devs,
    }

    # --- Plot Individual Layer Run ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    steps_axis = range(num_steps)

    # Subplot 1: Energy Evolution
    axes[0].plot(steps_axis, energies, label=r"$E(\theta^{(t)})$", color="tab:blue", lw=2)
    axes[0].axhline(lambda_0, color="black", linestyle="--", label=r"$\lambda_0$ (Exact Ground)")
    if abs(e_star_ansatz - lambda_0) > 1e-3:
        axes[0].axhline(e_star_ansatz, color="tab:red", linestyle=":", label=r"$E^*$ (Ansatz Min)")
    axes[0].set_title(f"Energy Evolution (L={num_layers})")
    axes[0].set_xlabel("SGD Step")
    axes[0].set_ylabel("Energy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Subplot 2: Step Sizes (Weinstein vs Polyak)
    axes[1].plot(steps_axis, polyak_steps, label="Polyak Step", color="tab:orange", lw=2)
    axes[1].plot(steps_axis, weinstein_steps, label="Weinstein Step", color="tab:green", lw=2, linestyle="--")
    axes[1].set_title(f"Polyak vs. Weinstein Step Size (L={num_layers})")
    axes[1].set_xlabel("SGD Step")
    axes[1].set_ylabel("Step Size $\eta^{(t)}$")
    axes[1].set_yscale("log")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Subplot 3: Ratio r = Weinstein / Polyak
    axes[2].plot(steps_axis, ratios, label=r"$r^{(t)} = \eta_W / \eta_P$", color="tab:purple", lw=2)
    axes[2].axhline(1.0, color="gray", linestyle=":", label="Ratio = 1.0")
    axes[2].set_title(f"Step Size Ratio (L={num_layers})")
    axes[2].set_xlabel("SGD Step")
    axes[2].set_ylabel("Ratio (Weinstein / Polyak)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"vqe_metrics_L{num_layers}.png"), dpi=300)
    plt.close(fig)


# 5. Save JSON Results
json_path = os.path.join(results_dir, "vqe_polyak_weinstein_results.json")
with open(json_path, "w") as f:
    json.dump(all_results, f, indent=4)
print(f"\nSaved complete JSON data to {json_path}")


# 6. Combined Layer Ablation Comparison Plots
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
steps_axis = range(num_steps)

for num_layers in layer_ablation_list:
    run_data = all_results["runs"][str(num_layers)]
    label = f"L = {num_layers} (E* - $\lambda_0$ = {run_data['e_star_ansatz'] - lambda_0:.1e})"

    # Top-Left: Energy Error E(t) - lambda_0
    energy_gap = [e - lambda_0 for e in run_data["energy"]]
    axes[0, 0].plot(steps_axis, energy_gap, label=label, lw=1.8)

    # Top-Right: Polyak Step Sizes
    axes[0, 1].plot(steps_axis, run_data["polyak_step"], label=f"L = {num_layers}", lw=1.8)

    # Bottom-Left: Weinstein Step Sizes
    axes[1, 0].plot(steps_axis, run_data["weinstein_step"], label=f"L = {num_layers}", lw=1.8)

    # Bottom-Right: Step Ratio r = Weinstein / Polyak
    axes[1, 1].plot(steps_axis, run_data["ratio"], label=f"L = {num_layers}", lw=1.8)

axes[0, 0].set_title(r"Energy Optimality Gap: $E(\theta^{(t)}) - \lambda_0$")
axes[0, 0].set_xlabel("SGD Step")
axes[0, 0].set_ylabel(r"$E(\theta^{(t)}) - \lambda_0$")
axes[0, 0].set_yscale("log")
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].set_title("Exact Polyak Step Size Across Layers")
axes[0, 1].set_xlabel("SGD Step")
axes[0, 1].set_ylabel(r"$\eta_{\mathrm{Polyak}}^{(t)}$")
axes[0, 1].set_yscale("log")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].set_title("Weinstein Step Size Across Layers")
axes[1, 0].set_xlabel("SGD Step")
axes[1, 0].set_ylabel(r"$\eta_{\mathrm{Weinstein}}^{(t)}$")
axes[1, 0].set_yscale("log")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].set_title(r"Step Size Ratio $r^{(t)} = \eta_{\mathrm{Weinstein}} / \eta_{\mathrm{Polyak}}$")
axes[1, 1].set_xlabel("SGD Step")
axes[1, 1].set_ylabel("Ratio $r^{(t)}$")
axes[1, 1].axhline(1.0, color="black", linestyle="--", alpha=0.6)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
ablation_plot_path = os.path.join(results_dir, "ablation_comparison.png")
plt.savefig(ablation_plot_path, dpi=300)
plt.close(fig)
print(f"Saved layer ablation plots to {ablation_plot_path}")