import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import numpy as std_np
import json
import networkx as nx


from iCANSOptimizer import iCANSOptimizer
from gCANSOptimizer import gCANSOptimizer


configs = [(4, 2), (4, 4), (6, 4), (6, 6), (8, 4), (8, 6), (10, 5), (10, 9), (12, 6), (12, 10), (14, 7), (14, 11)]

for q, l in configs:
    num_qubits = q
    num_layers = l
    print(f"===== Starting {q} qubits {l} layers =====")
    n_steps = 50
    dev = qml.device("lightning.qubit", wires=num_qubits)

    # # Use a cycle graph for the MaxCut problem
    # nx_graph = nx.cycle_graph(num_qubits)
    # graph = list(nx_graph.edges())

    # Use a random regular graph for the MaxCut problem
    nx_graph = nx.random_regular_graph(d=3, n=num_qubits, seed=42)
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
        return qml.expval(H), qml.var(H)

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

    ican_lrs = [0.001, 0.01, 0.1, 0.5]
    gcan_lrs = [0.001, 0.01, 0.1, 0.5]

    # all_costs, all_best, all_qng_pc, all_best_qng, all_switch, all_switch_qng = [], [], [], [], [], []
    all_ican = {lr: [] for lr in ican_lrs}
    all_gcan = {lr: [] for lr in gcan_lrs}

    # Estimate L for kown eigenspectrum
    L_est = delta / 2.0
    # print(f"Estimated L: {L_est}, Ground State Energy: {eigvals[0]}, Gap: {delta}")

    for _ in range(5):
        # QAOA parameters are a 1D array of size 2 * num_layers
        params = np.random.uniform(0, np.pi, size=2 * num_layers)
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

        ########################## ICAN / GCAN comparisons ##################
        for lr in ican_lrs:
            ican_params = initial_params.copy()
            ican_costs = []
            ican_opt = iCANSOptimizer(step=lr, L=L_est)
            for step in range(n_steps):
                ican_cost = cost_fn(ican_params)
                ican_costs.append(ican_cost)
                ican_params = ican_opt.step(cost_fn, ican_params)
            all_ican[lr].append([float(c[0]) for c in ican_costs])

        for lr in gcan_lrs:
            gcan_params = initial_params.copy()
            gcan_costs = []
            gcan_opt = gCANSOptimizer(step=lr, L=L_est)
            for step in range(n_steps):
                gcan_cost = cost_fn(gcan_params)
                gcan_costs.append(gcan_cost)
                gcan_params = gcan_opt.step(cost_fn, gcan_params)
            all_gcan[lr].append([float(c[0]) for c in gcan_costs])


    ican_curves = {lr: std_np.mean(all_ican[lr], axis=0) for lr in ican_lrs}
    gcan_curves = {lr: std_np.mean(all_gcan[lr], axis=0) for lr in gcan_lrs}

    x = list(range(1, n_steps + 1))
    results = {"num_qubits": int(num_qubits), "num_layers": int(num_layers),
            "polyak_gd": {}, "polyak_qng": {}, "ican": {}, "gcan": {}}
    for name, lrs, data in [("ican", ican_lrs, all_ican), ("gcan", gcan_lrs, all_gcan)]:
        for lr in lrs:
            d = {}
            for r in range(5):
                i = r + 1
                d[f"x{i}"] = x
                d[f"loss{i}"] = [float(c) for c in data[lr][r]]
            results[name][f"lr_{lr}"] = d

    save_path = f"./results/random_CAN/{num_qubits}q_{num_layers}l.json"
    with open(save_path, "w") as f:
        json.dump(results, f)

    plt.figure(figsize=(12, 7)) # Increased figure size


    styles = [':', '--', '-.', (0, (5, 1))]
    for j, lr in enumerate(ican_lrs):
        plt.plot(ican_curves[lr], marker='s', markersize=3, linestyle=styles[j], color='tab:red', linewidth=2, label=f"ICAN (LR={lr})")
    for j, lr in enumerate(gcan_lrs):
        plt.plot(gcan_curves[lr], marker='^', markersize=3, linestyle=styles[j], color='tab:purple', linewidth=2, label=f"GCAN (LR={lr})")
    plt.axhline(eigvals[0], color='black', linestyle=':', linewidth=4, label="GSE")

    colors = {"Weinstein": "red", "Constant": "green"}
    labeled = set()
    start = 0

    plt.xlabel("Optimization Step", fontsize=16)
    plt.ylabel("Expectation Value (Cost)", fontsize=16)
    plt.tick_params(labelsize=14)

    # Move legend outside the plot to the right and adjust font size
    plt.legend(fontsize=12, framealpha=1, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

    plt.grid(True, linewidth=2.5)
    plt.tight_layout() # Ensures the legend doesn't get cut off
    plt.savefig(f'./results/random_CAN/{num_qubits}q_{num_layers}l.png', bbox_inches='tight', dpi=600)
    plt.close() # Avoid accumulating figures across the 12 configs