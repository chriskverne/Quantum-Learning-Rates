import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import numpy as std_np

std_np.random.seed(42)
np.random.seed(42)

num_qubits = 6
num_layers = 3
num_params = num_layers * num_qubits * 2
n_steps = 40
n_runs = 1
h_field = 1.0

# 1. Use lightning.gpu for fast adjoint gradients and expectations
dev = qml.device("default.qubit", wires=num_qubits)

# Build unscaled H
coeffs, observables = [], []
for i in range(num_qubits - 1):
    coeffs.append(-1.0)
    observables.append(qml.PauliZ(i) @ qml.PauliZ(i + 1))
for i in range(num_qubits):
    coeffs.append(-h_field)
    observables.append(qml.PauliX(i))

H = qml.Hamiltonian(coeffs, observables)
gse = float(std_np.min(std_np.linalg.eigvalsh(qml.matrix(H, wire_order=range(num_qubits)))))
print(f"Ground State Energy: {gse:.6f}")


def ansatz(params):
    for layer in range(num_layers):
        for q in range(num_qubits):
            qml.RX(params[layer, q, 0], wires=q)
            qml.RY(params[layer, q, 1], wires=q)
        for q in range(num_qubits - 1):
            qml.CNOT(wires=[q, q + 1])


def square_pauli_only(op):
    """Square Hamiltonian and strip the Identity constant so lightning.gpu never sees qml.Identity."""
    ps = qml.pauli.pauli_sentence(op)
    ps2 = ps @ ps
    ps2.simplify()
    id_word = qml.pauli.PauliWord({})
    const = float(np.real(ps2.pop(id_word, 0.0)))

    coeffs_sq, ops_sq = [], []
    for pw, c in ps2.items():
        coeffs_sq.append(float(np.real(c)))
        ops_sq.append(pw.operation(wire_order=range(num_qubits)))
    return qml.Hamiltonian(coeffs_sq, ops_sq), const


def make_fns(Ham_pauli):
    Ham_pauli_sq, const_sq = square_pauli_only(Ham_pauli)

    # Coordinate Lipschitz bound: L_coord <= sum(|a_k|) -> lr_coord = 1 / sum(|a_k|)
    # Global Lipschitz bound:     L_global <= P * sum(|a_k|) -> lr_global = 1 / (P * sum(|a_k|))
    l1_norm = float(std_np.sum(std_np.abs(Ham_pauli.coeffs)))
    L_coord = l1_norm
    L_global = num_params * l1_norm

    @qml.qnode(dev, diff_method="adjoint")
    def cost_fn(params):
        ansatz(params)
        return qml.expval(Ham_pauli)

    @qml.qnode(dev, diff_method=None)
    def pauli_sq_fn(params):
        ansatz(params)
        return qml.expval(Ham_pauli_sq)

    def energy_and_std(params):
        e_pauli = float(cost_fn(params))
        e2_pauli = float(pauli_sq_fn(params)) + const_sq
        var = max(e2_pauli - e_pauli**2, 1e-16)
        return e_pauli, float(std_np.sqrt(var))

    return {
        "cost_fn": cost_fn,
        "energy_and_std": energy_and_std,
        "grad_fn": qml.grad(cost_fn),
        "lr_global": 1.0 / L_global,
        "lr_coord": 1.0 / L_coord,
    }


prob = make_fns(H)
print(f"lr_global = {prob['lr_global']:.6e} | lr_coord = {prob['lr_coord']:.6e}")


# 2. Separate QNode with diff_method=None for metric_tensor so it doesn't try to run adjoint on probs
@qml.qnode(dev, diff_method=None)
def metric_qnode(params):
    ansatz(params)
    return qml.probs(wires=range(num_qubits))

metric_fn = qml.metric_tensor(metric_qnode, approx="block-diag")


def get_nat_grad(params, grad):
    g = metric_fn(params)
    g_mat = std_np.reshape(g, (params.size, params.size))
    g_inv = std_np.linalg.pinv(g_mat + 1e-6 * std_np.eye(params.size))
    nat_grad = g_inv @ std_np.reshape(grad, (-1,))
    return np.array(nat_grad.reshape(params.shape), requires_grad=True)


def run_opt(prob, init_params, method):
    params = init_params.copy()
    hist = []

    for _ in range(n_steps):
        if method in ("Weinstein GD", "Weinstein QNG"):
            energy, std = prob["energy_and_std"](params)
            hist.append(energy)
            grad = prob["grad_fn"](params)
            direction = get_nat_grad(params, grad) if "QNG" in method else grad
            step_size = std / (float(np.sum(grad * direction)) + 1e-16)
            params = params - step_size * direction

        elif method in ("Global Lip GD", "Global Lip QNG"):
            energy = float(prob["cost_fn"](params))
            hist.append(energy)
            grad = prob["grad_fn"](params)
            direction = get_nat_grad(params, grad) if "QNG" in method else grad
            params = params - prob["lr_global"] * direction

        elif method in ("Coord Lip GD", "Coord Lip QNG"):
            energy = float(prob["cost_fn"](params))
            hist.append(energy)

            # Full sweep: sequentially update all P parameters in 1 step
            for idx in range(params.size):
                grad = prob["grad_fn"](params)
                direction = get_nat_grad(params, grad) if "QNG" in method else grad

                multi_idx = std_np.unravel_index(idx, params.shape)
                new_params = std_np.array(params, dtype=float)
                new_params[multi_idx] -= prob["lr_coord"] * float(direction[multi_idx])
                params = np.array(new_params, requires_grad=True)

    return hist


methods = [
    # "Weinstein GD",   # Commented out since we already have these runs saved
    # "Weinstein QNG",  # Commented out since we already have these runs saved
    "Global Lip GD",
    "Global Lip QNG",
    "Coord Lip GD",
    "Coord Lip QNG",
]
hists = {m: [] for m in methods}

for run in range(n_runs):
    init_params = np.random.uniform(0, 2 * np.pi, (num_layers, num_qubits, 2), requires_grad=True)
    for m in methods:
        hists[m].append(run_opt(prob, init_params, m))
    print(f"Completed run {run + 1}/{n_runs}")

print("\n# --- PLOTTED DATA ARRAYS (MEAN OVER RUNS) ---")
print(f"gse = {gse}")
print("plotted_curves = {")
for m in methods:
    mean_curve = std_np.round(std_np.mean(hists[m], axis=0), 6).tolist()
    print(f"    '{m}': {mean_curve},")
print("}")

print("\n# --- RAW RUNS (SHAPE: n_runs x n_steps) ---")
for m in methods:
    raw_arr = std_np.round(std_np.array(hists[m]), 6).tolist()
    print(f"raw_{m.replace(' ', '_')} = {raw_arr}")

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
fig, ax = plt.subplots(1, 1, figsize=(7, 5))

for i, m in enumerate(methods):
    mean_curve = std_np.mean(hists[m], axis=0)
    ax.plot(mean_curve, color=colors[i], linewidth=2, label=m)

ax.axhline(gse, color="k", linestyle=":", label="Ground State Energy")
ax.set_title("6-Qubit TFIM (Unscaled): Global vs. Coordinate Lipschitz")
ax.set_xlabel("Step")
ax.set_ylabel("Energy")
ax.set_ylim(gse - 0.5, 2.0)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("tfim_lipschitz_unscaled.png", dpi=150)
plt.show()