import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import numpy as std_np
import json
import networkx as nx

configs = [(4, 4), (6, 6), (8, 6), (10, 9), (12, 10), (14, 11)]

for q, l in configs:
    num_qubits = q
    num_layers = l
    print(f"===== Starting {q} qubits {l} layers =====")
    n_steps = 50
    dev = qml.device("lightning.qubit", wires=num_qubits)

    # Define the MaxCut problem properly
    nx_graph = nx.cycle_graph(num_qubits)
    graph = list(nx_graph.edges())  

    coeffs = []
    observables = []
    for i, j in graph:
        coeffs.append(1.0)
        observables.append(qml.PauliZ(i) @ qml.PauliZ(j))
    H = qml.Hamiltonian(coeffs, observables)

    h1_norm = np.sum(np.abs(coeffs))

    def ansatz(params):
        # QAOA parameters are split into gammas and betas
        gammas = params[:num_layers]
        betas = params[num_layers:]
        
        # Initial state preparation: uniform superposition
        for q in range(num_qubits):
            qml.Hadamard(wires=q)
            
        for layer in range(num_layers):
            # Cost layer: exp(-i * gamma * Z_i Z_j)
            for i, j in graph:
                qml.CNOT(wires=[i, j])
                qml.RZ(2 * gammas[layer], wires=j)
                qml.CNOT(wires=[i, j])
                
            # Mixer layer: exp(-i * beta * X_i)
            for q in range(num_qubits):
                qml.RX(2 * betas[layer], wires=q)

    @qml.qnode(dev)
    def cost_fn(params):
        ansatz(params)
        return qml.expval(H)

    @qml.qnode(dev)
    def var_fn(params):
        ansatz(params)
        return qml.var(H)

    # Calculate eigenvalues to find the ground state and gap
    eigvals = np.linalg.eigvalsh(qml.matrix(H, wire_order=range(num_qubits)))
    # Handle degeneracy by finding the gap between unique eigenvalues
    unique_eigvals = np.unique(eigvals)
    delta = unique_eigvals[1] - unique_eigvals[0] if len(unique_eigvals) > 1 else 1.0

    grad_fn = qml.grad(cost_fn)

    sgd_lrs = [0.001, 0.01, 0.1, 0.5]
    adam_lrs = [0.001, 0.01, 0.1, 0.5]
    qng_lrs = [0.001, 0.01, 0.1, 0.5]

    all_costs, all_best, all_qng_pc, all_best_qng, all_switch, all_switch_qng = [], [], [], [], [], []
    all_sgd = {lr: [] for lr in sgd_lrs}
    all_adam = {lr: [] for lr in adam_lrs}
    all_qng = {lr: [] for lr in qng_lrs}

    for _ in range(5):
        # QAOA parameters are a 1D array of size 2 * num_layers
        params = np.random.normal(0, np.pi, (2 * num_layers,), requires_grad=True)
        initial_params = params.copy()

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
            costs.append(cost)
            if step % 10 == 0:
                print(f"Step {step:03d}: Cost = {cost:.6f}")

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
                regimes.append("Temple" if std < delta else "Weinstein")
                lr = std / (grad_norm_sq + 1e-8)
            else:
                regimes.append("Constant")
                lr = lr_const

            params = params - lr * grad

        ########################## Comparisons to default lr's ##################
        for lr in sgd_lrs:
            sgd_params = initial_params.copy()
            sgd_costs = []
            for step in range(n_steps):
                sgd_cost = cost_fn(sgd_params)
                sgd_costs.append(sgd_cost)
                sgd_grad = grad_fn(sgd_params)
                sgd_params = sgd_params - lr * sgd_grad
            all_sgd[lr].append([float(c) for c in sgd_costs])

        for lr in adam_lrs:
            adam_params = initial_params.copy()
            adam_costs = []
            adam_opt = qml.AdamOptimizer(stepsize=lr)
            for step in range(n_steps):
                adam_cost = cost_fn(adam_params)
                adam_costs.append(adam_cost)
                adam_params = adam_opt.step(cost_fn, adam_params)
            all_adam[lr].append([float(c) for c in adam_costs])

        for lr in qng_lrs:
            qng_params = initial_params.copy()
            qng_costs = []
            qng_opt = qml.QNGOptimizer(stepsize=lr)
            for step in range(n_steps):
                qng_cost = cost_fn(qng_params)
                qng_costs.append(qng_cost)
                qng_params = qng_opt.step(cost_fn, qng_params)
            all_qng[lr].append([float(c) for c in qng_costs])

        ######################## Our method ##########################
        qng_pc_params = initial_params.copy()
        qng_pc_costs = []
        best_cost_qng = np.inf
        best_params_qng = qng_pc_params.copy()
        no_improve_qng = 0
        refining_qng = False
        switch_qng = None
        qng_pc_opt = qml.QNGOptimizer(stepsize=0.1)

        for step in range(n_steps):
            cost = cost_fn(qng_pc_params)
            qng_pc_costs.append(cost)

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
                qng_pc_opt.stepsize = 1 / (4 * var + 1e-8) # Added 1e-8 for numerical stability
            else:
                qng_pc_opt.stepsize = 0.1

            qng_pc_params = qng_pc_opt.step(cost_fn, qng_pc_params)

        all_costs.append([float(c) for c in costs])
        all_best.append(std_np.minimum.accumulate(all_costs[-1]))
        all_qng_pc.append([float(c) for c in qng_pc_costs])
        all_best_qng.append(std_np.minimum.accumulate(all_qng_pc[-1]))
        all_switch.append(switch)
        all_switch_qng.append(switch_qng)

    best_curve = std_np.mean(all_best, axis=0)
    best_curve_qng = std_np.mean(all_best_qng, axis=0)
    sgd_curves = {lr: std_np.mean(all_sgd[lr], axis=0) for lr in sgd_lrs}
    adam_curves = {lr: std_np.mean(all_adam[lr], axis=0) for lr in adam_lrs}
    qng_curves = {lr: std_np.mean(all_qng[lr], axis=0) for lr in qng_lrs}

    print(f"Average Best cost = {best_curve[-1]:.6f}  (GSE = {eigvals[0]:.6f})")

    x = list(range(1, n_steps + 1))
    results = {"num_qubits": int(num_qubits), "num_layers": int(num_layers),
            "polyak_gd": {}, "polyak_qng": {}, "sgd": {}, "adam": {}, "qng": {}}
    for name, lrs, data in [("sgd", sgd_lrs, all_sgd), ("adam", adam_lrs, all_adam), ("qng", qng_lrs, all_qng)]:
        for lr in lrs:
            d = {}
            for r in range(5):
                i = r + 1
                d[f"x{i}"] = x
                d[f"loss{i}"] = [float(c) for c in data[lr][r]]
            results[name][f"lr_{lr}"] = d
    for r in range(5):
        i = r + 1
        results["polyak_gd"][f"x{i}"] = x
        results["polyak_gd"][f"loss{i}"] = [float(c) for c in all_costs[r]]
        results["polyak_gd"][f"best_loss{i}"] = [float(c) for c in all_best[r]]
        results["polyak_gd"][f"weinstein_vs_constant{i}"] = all_switch[r]
        results["polyak_qng"][f"x{i}"] = x
        results["polyak_qng"][f"loss{i}"] = [float(c) for c in all_qng_pc[r]]
        results["polyak_qng"][f"best_loss{i}"] = [float(c) for c in all_best_qng[r]]
        results["polyak_qng"][f"weinstein_vs_constant{i}"] = all_switch_qng[r]

    save_path = f"./results/cycle/{num_qubits}q_{num_layers}l.json"
    with open(save_path, "w") as f:
        json.dump(results, f)

    plt.figure(figsize=(12, 7)) # Increased figure size
    plt.plot(best_curve, marker='o', markersize=3, linestyle='-', color='black', linewidth=4, label="Polyak -> GD (best so far)")
    plt.plot(best_curve_qng, marker='d', markersize=3, linestyle='-', color='tab:cyan', linewidth=3, label="Polyak -> QNG (best so far)")
    styles = [':', '--', '-.', (0, (5, 1))]
    for j, lr in enumerate(sgd_lrs):
        plt.plot(sgd_curves[lr], marker='s', markersize=3, linestyle=styles[j], color='tab:red', linewidth=2, label=f"GD (LR={lr})")
    for j, lr in enumerate(adam_lrs):
        plt.plot(adam_curves[lr], marker='^', markersize=3, linestyle=styles[j], color='tab:purple', linewidth=2, label=f"Adam (LR={lr})")
    for j, lr in enumerate(qng_lrs):
        plt.plot(qng_curves[lr], marker='*', markersize=3, linestyle=styles[j], color='tab:brown', linewidth=2, label=f"QNG (LR={lr})")
    plt.axhline(eigvals[0], color='black', linestyle=':', linewidth=4, label="GSE")
    regimes = ["Weinstein" if r == "Temple" else r for r in regimes]
    colors = {"Weinstein": "red", "Constant": "green"}
    labeled = set()
    start = 0
    for i in range(1, len(regimes) + 1):
        if i == len(regimes) or regimes[i] != regimes[start]:
            r = regimes[start]
            plt.axvspan(start, i, color=colors[r], alpha=0.15,
                        label=r if r not in labeled else None)
            labeled.add(r)
            if i < len(regimes):
                plt.axvline(i, color='k', linestyle='--', linewidth=1.5)
            start = i
    plt.xlabel("Optimization Step", fontsize=16)
    plt.ylabel("Expectation Value (Cost)", fontsize=16)
    plt.tick_params(labelsize=14)

    # Move legend outside the plot to the right and adjust font size
    plt.legend(fontsize=12, framealpha=1, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

    plt.grid(True, linewidth=2.5)
    plt.tight_layout() # Ensures the legend doesn't get cut off
    plt.savefig(f'./results/cycle/{num_qubits}q_{num_layers}l.png', bbox_inches='tight', dpi=600)