import os
os.environ["OMP_NUM_THREADS"] = "1"  # Prevent C++ thread contention in multi-processing

import json
import shutil
import networkx as nx
import matplotlib.pyplot as plt
import numpy as std_np
import pennylane as qml
from pennylane import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# Setup Output Directory
results_dir = "./results_mis_er"
os.makedirs(results_dir, exist_ok=True)


# 1. Vectorized Exact Diagonal Ground State Spectrum Solver (MIS: linear + quadratic terms)
def compute_exact_diagonal_spectrum(num_qubits, h, quad_weights, const_offset):
    # H = const_offset + sum_i h[i] Z_i + sum_(i,j) quad_weights[(i,j)] Z_i Z_j
    num_states = 1 << num_qubits
    idx = std_np.arange(num_states)[:, None]
    bit_shifts = std_np.arange(num_qubits - 1, -1, -1)
    bits = (idx >> bit_shifts) & 1
    spins = 1 - 2 * bits

    diag_energies = std_np.full(num_states, const_offset, dtype=float)
    for i, hi in h.items():
        diag_energies += hi * spins[:, i]
    for (i, j), w in quad_weights.items():
        diag_energies += w * spins[:, i] * spins[:, j]

    unique_eigvals = std_np.unique(diag_energies)
    unique_eigvals.sort()
    gse = float(unique_eigvals[0])
    delta = float(unique_eigvals[1] - unique_eigvals[0]) if len(unique_eigvals) > 1 else 1.0
    return gse, delta


# 2. Parallel Worker Function (Executes 1 Independent Seed Run)
def run_single_seed(seed_idx, num_qubits, num_layers, h, quad_weights, H, delta, n_steps=200):
    dev = qml.device("lightning.qubit", wires=num_qubits)

    def ansatz(params):
        gammas = params[:num_layers]
        betas = params[num_layers:]

        for q in range(num_qubits):
            qml.Hadamard(wires=q)

        for layer in range(num_layers):
            # Quadratic (edge) part of the MIS Hamiltonian: exp(-i*gamma*w_ij*Z_i Z_j)
            for (i, j), w in quad_weights.items():
                qml.CNOT(wires=[i, j])
                qml.RZ(2 * gammas[layer] * w, wires=j)
                qml.CNOT(wires=[i, j])

            # Linear (single-qubit) part of the MIS Hamiltonian: exp(-i*gamma*h_i*Z_i)
            # This block did not exist for pure MaxCut -- it's required because MIS
            # has single-qubit Z terms from both the -sum(Z_i) objective and the
            # expanded (1+Z_i)(1+Z_j)/4 penalty terms.
            for i, hi in h.items():
                if hi != 0.0:
                    qml.RZ(2 * gammas[layer] * hi, wires=i)

            for q in range(num_qubits):
                qml.RX(2 * betas[layer], wires=q)

    # 1. Fast QNode for standard gradients (Adjoint execution)
    @qml.qnode(dev, diff_method="adjoint")
    def cost_fn(params):
        ansatz(params)
        return qml.expval(H)

    # 2. Dedicated QNode for QNG metric tensor support (Parameter-Shift)
    @qml.qnode(dev, diff_method="parameter-shift")
    def cost_fn_qng(params):
        ansatz(params)
        return qml.expval(H)

    @qml.qnode(dev)
    def var_fn(params):
        ansatz(params)
        return qml.var(H)

    grad_fn = qml.grad(cost_fn)

    # Initial random parameters for this seed
    T = 1.0  # total annealing time -- may need tuning per Hamiltonian scale
    layer_idx = std_np.arange(1, num_layers + 1)
    gammas_init = (layer_idx / num_layers) * (T / num_layers)
    betas_init = (1 - layer_idx / num_layers) * (T / num_layers)
    init_p = std_np.concatenate([gammas_init, betas_init])
    params = np.array(init_p, requires_grad=True)

    initial_params = params.copy()

    sgd_lrs = [0.001, 0.01, 0.1, 0.5]
    adam_lrs = [0.001, 0.01, 0.1, 0.5]
    qng_lrs = [0.01, 0.1]

    # --- 1. Polyak -> GD ---
    costs = []
    regimes = []
    best_cost = np.inf
    best_params = params.copy()
    no_improve = 0
    patience = 5
    refining = False
    lr_const = None
    switch = None

    for step in range(n_steps):
        cost = cost_fn(params)
        costs.append(float(cost))

        if cost < best_cost:
            best_cost = cost
            best_params = params.copy()
            no_improve = 0
        else:
            no_improve += 1

        if not refining and no_improve >= patience:
            refining = True
            params = best_params.copy()
            lr_const = 0.1
            no_improve = 0
            switch = step + 1

        grad = grad_fn(params)

        if not refining:
            var = var_fn(params)
            std = np.sqrt(var)
            grad_norm_sq = np.sum(grad**2)
            regimes.append("Weinstein" if std >= delta else "Temple")
            lr = std / (grad_norm_sq + 1e-8)
        else:
            regimes.append("Constant")
            lr = lr_const

        params = params - lr * grad

        if (step + 1) % 10 == 0:
            print(f"[Seed {seed_idx + 1}] [Polyak-GD] Step {step + 1}/{n_steps} complete.", flush=True)

    # --- 2. SGD Baselines ---
    sgd_results = {}
    for lr in sgd_lrs:
        sgd_params = initial_params.copy()
        sgd_costs = []
        for step in range(n_steps):
            sgd_cost = cost_fn(sgd_params)
            sgd_costs.append(float(sgd_cost))
            sgd_grad = grad_fn(sgd_params)
            sgd_params = sgd_params - lr * sgd_grad

            if (step + 1) % 10 == 0:
                print(f"[Seed {seed_idx + 1}] [SGD lr={lr}] Step {step + 1}/{n_steps} complete.", flush=True)

        sgd_results[lr] = sgd_costs

    # --- 3. Adam Baselines ---
    adam_results = {}
    for lr in adam_lrs:
        adam_params = initial_params.copy()
        adam_costs = []
        adam_opt = qml.AdamOptimizer(stepsize=lr)
        for step in range(n_steps):
            adam_cost = cost_fn(adam_params)
            adam_costs.append(float(adam_cost))
            adam_params = adam_opt.step(cost_fn, adam_params)

            if (step + 1) % 10 == 0:
                print(f"[Seed {seed_idx + 1}] [Adam lr={lr}] Step {step + 1}/{n_steps} complete.", flush=True)

        adam_results[lr] = adam_costs

    # --- 4. QNG Baselines ---
    qng_results = {}
    for lr in qng_lrs:
        qng_params = initial_params.copy()
        qng_costs = []
        qng_opt = qml.QNGOptimizer(stepsize=lr, approx="diag")
        for step in range(n_steps):
            qng_cost = cost_fn_qng(qng_params)
            qng_costs.append(float(qng_cost))
            qng_params = qng_opt.step(cost_fn_qng, qng_params)

            if (step + 1) % 10 == 0:
                print(f"[Seed {seed_idx + 1}] [QNG lr={lr}] Step {step + 1}/{n_steps} complete.", flush=True)

        qng_results[lr] = qng_costs

    # --- 5. Polyak -> QNG ---
    qng_pc_params = initial_params.copy()
    qng_pc_costs = []
    best_cost_qng = np.inf
    best_params_qng = qng_pc_params.copy()
    no_improve_qng = 0
    refining_qng = False
    switch_qng = None
    qng_pc_opt = qml.QNGOptimizer(stepsize=0.1, approx="diag")

    for step in range(n_steps):
        cost = cost_fn_qng(qng_pc_params)
        qng_pc_costs.append(float(cost))

        if cost < best_cost_qng:
            best_cost_qng = cost
            best_params_qng = qng_pc_params.copy()
            no_improve_qng = 0
        else:
            no_improve_qng += 1

        if not refining_qng and no_improve_qng >= patience:
            refining_qng = True
            qng_pc_params = best_params_qng.copy()
            no_improve_qng = 0
            switch_qng = step + 1

        if not refining_qng:
            var = var_fn(qng_pc_params)
            qng_pc_opt.stepsize = 1 / (4 * var + 1e-8)
        else:
            qng_pc_opt.stepsize = 0.1

        qng_pc_params = qng_pc_opt.step(cost_fn_qng, qng_pc_params)

        if (step + 1) % 10 == 0:
            print(f"[Seed {seed_idx + 1}] [Polyak-QNG] Step {step + 1}/{n_steps} complete.", flush=True)

    return {
        "seed_idx": seed_idx,
        "polyak_gd": costs,
        "polyak_gd_best": std_np.minimum.accumulate(costs).tolist(),
        "polyak_gd_switch": switch,
        "regimes": regimes,
        "polyak_qng": qng_pc_costs,
        "polyak_qng_best": std_np.minimum.accumulate(qng_pc_costs).tolist(),
        "polyak_qng_switch": switch_qng,
        "sgd": sgd_results,
        "adam": adam_results,
        "qng": qng_results,
    }


# 3. Master Execution Runner Function
def run_erdos_renyi_mis_benchmark(num_qubits, num_layers, p_edge=0.5, penalty=2.0, n_steps=200, num_seeds=1):
    print(f"\n======================================================================")
    print(f"  OPTIMIZED QAOA (MIS): {num_qubits} Qubits, {num_layers} Layers, p={p_edge} ({n_steps} Steps)")
    print(f"======================================================================\n")

    graph_G = nx.erdos_renyi_graph(n=num_qubits, p=p_edge, seed=42)
    edges = list(graph_G.edges())
    print(f"Generated Erdős–Rényi Graph G({num_qubits}, {p_edge}) with {len(edges)} total edges.")

    # --- Build MIS Hamiltonian ---
    # H = -sum_i Z_i + P * sum_(i,j) in E (1+Z_i)(1+Z_j)/4
    #   = const_offset + sum_i h_i Z_i + sum_(i,j) quad_coeff * Z_i Z_j
    degrees = dict(graph_G.degree())
    quad_coeff = penalty / 4.0
    h = {i: -1.0 + quad_coeff * degrees.get(i, 0) for i in range(num_qubits)}
    quad_weights = {edge: quad_coeff for edge in edges}
    const_offset = quad_coeff * len(edges)

    coeffs = []
    observables = []
    # Constant offset from expanding (1+Z_i)(1+Z_j)/4 -- included directly in H
    # (via Identity) so that cost_fn's qml.expval(H) and the exact solver's GSE
    # refer to the exact same operator. Previously this offset was added only
    # in compute_exact_diagonal_spectrum, which silently shifted GSE relative
    # to every optimized cost curve by a constant amount.
    if const_offset != 0.0:
        coeffs.append(const_offset)
        observables.append(qml.Identity(0))
    for i, hi in h.items():
        if hi != 0.0:
            coeffs.append(hi)
            observables.append(qml.PauliZ(i))
    for (i, j), w in quad_weights.items():
        coeffs.append(w)
        observables.append(qml.PauliZ(i) @ qml.PauliZ(j))
    H = qml.Hamiltonian(coeffs, observables)

    gse_val, delta = compute_exact_diagonal_spectrum(num_qubits, h, quad_weights, const_offset)
    print(f"Exact GSE = {gse_val:.6f} | Spectral Gap = {delta:.6f}")
    print(f"Executing {num_seeds} independent seeds in parallel across CPU threads...\n")

    # Run seeds concurrently across CPU cores
    seed_outputs = []
    with ProcessPoolExecutor(max_workers=num_seeds) as executor:
        futures = [
            executor.submit(run_single_seed, s, num_qubits, num_layers, h, quad_weights, H, delta, n_steps)
            for s in range(num_seeds)
        ]
        for future in as_completed(futures):
            res = future.result()
            seed_outputs.append(res)
            print(f" -> Completed Seed {res['seed_idx'] + 1}/{num_seeds}\n", flush=True)

    seed_outputs.sort(key=lambda x: x["seed_idx"])

    # Aggregate Averaged Metrics
    all_best = [s["polyak_gd_best"] for s in seed_outputs]
    all_best_qng = [s["polyak_qng_best"] for s in seed_outputs]
    best_curve = std_np.mean(all_best, axis=0)
    best_curve_qng = std_np.mean(all_best_qng, axis=0)

    sgd_lrs = [0.001, 0.01, 0.1, 0.5]
    adam_lrs = [0.001, 0.01, 0.1, 0.5]
    qng_lrs = [0.01]

    sgd_curves = {lr: std_np.mean([s["sgd"][lr] for s in seed_outputs], axis=0) for lr in sgd_lrs}
    adam_curves = {lr: std_np.mean([s["adam"][lr] for s in seed_outputs], axis=0) for lr in adam_lrs}
    qng_curves = {lr: std_np.mean([s["qng"][lr] for s in seed_outputs], axis=0) for lr in qng_lrs}

    print(f"\n[Completed {num_qubits}q/{num_layers}l] Mean Best Cost = {best_curve[-1]:.6f} | GSE = {gse_val:.6f}")

    x = list(range(1, n_steps + 1))
    results = {
        "num_qubits": int(num_qubits),
        "num_layers": int(num_layers),
        "graph_type": "erdos_renyi",
        "p_edge": float(p_edge),
        "num_edges": len(edges),
        "problem": "max_independent_set",
        "penalty": float(penalty),
        "gse": gse_val,
        "spectral_gap": delta,
        "polyak_gd": {},
        "polyak_qng": {},
        "sgd": {},
        "adam": {},
        "qng": {},
    }

    for name, lrs, data_key in [("sgd", sgd_lrs, "sgd"), ("adam", adam_lrs, "adam"), ("qng", qng_lrs, "qng")]:
        for lr in lrs:
            d = {}
            for r, s in enumerate(seed_outputs):
                i = r + 1
                d[f"x{i}"] = x
                d[f"loss{i}"] = s[data_key][lr]
            results[name][f"lr_{lr}"] = d

    for r, s in enumerate(seed_outputs):
        i = r + 1
        results["polyak_gd"][f"x{i}"] = x
        results["polyak_gd"][f"loss{i}"] = s["polyak_gd"]
        results["polyak_gd"][f"best_loss{i}"] = s["polyak_gd_best"]
        results["polyak_gd"][f"weinstein_vs_constant{i}"] = s["polyak_gd_switch"]
        results["polyak_qng"][f"x{i}"] = x
        results["polyak_qng"][f"loss{i}"] = s["polyak_qng"]
        results["polyak_qng"][f"best_loss{i}"] = s["polyak_qng_best"]
        results["polyak_qng"][f"weinstein_vs_constant{i}"] = s["polyak_qng_switch"]

    save_path = os.path.join(results_dir, f"{num_qubits}q_{num_layers}l_mis.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)

    # Plot Averaged Trajectories
    plt.figure(figsize=(12, 7))
    plt.plot(best_curve, marker="o", markersize=3, color="black", linewidth=4, label="Polyak -> GD")
    plt.plot(best_curve_qng, marker="d", markersize=3, color="tab:cyan", linewidth=3, label="Polyak -> QNG")

    styles = [":", "--", "-.", (0, (5, 1))]
    for j, lr in enumerate(sgd_lrs):
        plt.plot(sgd_curves[lr], marker="s", markersize=3, linestyle=styles[j], color="tab:red", linewidth=2, label=f"GD (LR={lr})")
    for j, lr in enumerate(adam_lrs):
        plt.plot(adam_curves[lr], marker="^", markersize=3, linestyle=styles[j], color="tab:purple", linewidth=2, label=f"Adam (LR={lr})")
    for j, lr in enumerate(qng_lrs):
        plt.plot(qng_curves[lr], marker="*", markersize=3, linestyle=styles[j], color="tab:brown", linewidth=2, label=f"QNG (LR={lr})")

    plt.axhline(gse_val, color="black", linestyle=":", linewidth=4, label="GSE (MIS optimum)")

    regimes = seed_outputs[0]["regimes"]
    regimes = ["Weinstein" if r == "Temple" else r for r in regimes]
    colors = {"Weinstein": "red", "Constant": "green"}
    labeled = set()
    start = 0
    for i in range(1, len(regimes) + 1):
        if i == len(regimes) or regimes[i] != regimes[start]:
            r = regimes[start]
            plt.axvspan(start, i, color=colors[r], alpha=0.15, label=r if r not in labeled else None)
            labeled.add(r)
            if i < len(regimes):
                plt.axvline(i, color="k", linestyle="--", linewidth=1.5)
            start = i

    plt.xlabel("Optimization Step", fontsize=16)
    plt.ylabel("Expectation Value (Cost)", fontsize=16)
    plt.title(f"MIS on Erdős–Rényi Graph ({num_qubits} Qubits, {num_layers} Layers, p={p_edge})", fontsize=18)
    plt.tick_params(labelsize=14)
    plt.legend(fontsize=11, framealpha=1, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.0)
    plt.grid(True, linewidth=2.5)
    plt.tight_layout()

    # Save plot directly to disk
    plot_path = os.path.join(results_dir, f"plot_{num_qubits}q_{num_layers}l_mis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {plot_path}")


# 4. Main Execution Loop Across Scale Configurations
if __name__ == "__main__":
    benchmark_configs = [
        (4, 2)  # High-depth scale configuration target
    ]

    for qubits, layers in benchmark_configs:
        run_erdos_renyi_mis_benchmark(num_qubits=qubits, num_layers=layers, p_edge=0.5, penalty=2.0, n_steps=200)

    # Archive All Output JSONs and Plots into Zip Package
    shutil.make_archive("erdos_renyi_qaoa_mis_results", "zip", results_dir)
    print("\nAll benchmark runs completed! Output package saved to 'erdos_renyi_qaoa_mis_results.zip'.")