# import json

# example = "./QAOA/Cycle/4q_2l.json"

# with open(example, "r") as f:
#     data = json.load(f)

# # Top-level keys only
# print(list(data.keys()))

# folder_path = './QAOA'
# folders = ['Cycle', 'Maxcut', 'Reyni_Random']

# configs = ['4q_2l', '6q_4l', '8q_6l', '10q_8l', '12q_10l', '14q_10l']

# for folder in folders:
#     folder_path = f'./QAOA/{folder}'
#     for config in configs:
#         config_path = f'{folder_path}/{config}'

#         # Read JSON file here and process data / plot it (table is prefferd)

import json
import os
import numpy as np
import pandas as pd

# Switch base_dir to: "./QAOA/Cycle", "./QAOA/Maxcut", or "./QAOA/Reyni_Random"
base_dir = "./QAOA/Reyni_Random"
configs = ["4q_2l", "6q_4l", "8q_6l", "10q_8l", "12q_10l", "14q_10l"]
n_steps = 50
n_runs = 5

MAXCUT_GSE = {4: -2, 6: -5, 8: -8, 10: -11, 12: -13, 14: -15}


def compute_metrics(name, runs_matrix, e_min, e_max):
    best_runs = np.minimum.accumulate(runs_matrix, axis=1)

    # 1. Final Energy
    final_energies = best_runs[:, -1]
    mean_final, std_final = np.mean(final_energies), np.std(final_energies)

    # 2. Steps to 95% threshold
    target_energy = e_max - 0.95 * (e_max - e_min)
    steps = [
        (np.where(best_runs[r] <= target_energy)[0][0] + 1)
        if np.any(best_runs[r] <= target_energy)
        else n_steps
        for r in range(n_runs)
    ]

    # 3. Regret AUC
    auc_runs = np.sum(best_runs - e_min, axis=1)

    return {
        "Optimizer": name,
        "Final Energy": f"{mean_final:.3f} ± {std_final:.2f}",
        "Steps to 95%": f"{np.mean(steps):.1f} ± {np.std(steps):.1f}",
        "Regret AUC": f"{np.mean(auc_runs):.1f} ± {np.std(auc_runs):.1f}",
    }


pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

for cfg in configs:
    json_path = os.path.join(base_dir, f"{cfg}.json")
    if not os.path.exists(json_path):
        json_path = os.path.join(base_dir, cfg, "results.json")
    if not os.path.exists(json_path):
        continue

    with open(json_path, "r") as f:
        data = json.load(f)

    num_qubits = data["num_qubits"]
    dir_name = base_dir.lower()

    # --- 3-WAY BRANCH FOR GSE & E_MAX ---
    if "cycle" in dir_name:
        e_min = -num_qubits if num_qubits % 2 == 0 else -(num_qubits - 2)
        e_max = float(num_qubits)

    elif "maxcut" in dir_name:
        e_min = float(MAXCUT_GSE.get(num_qubits, 4 - 1.5 * num_qubits))
        e_max = float(1.5 * num_qubits)

    elif "reyni" in dir_name or "renyi" in dir_name:
        e_min = float(data["gse"])
        # Uses total edge count |E| as e_max (consistent with Cycle and MaxCut3)
        e_max = float(data.get("num_edges", num_qubits))

    else:
        e_min = float(data.get("gse", -num_qubits))
        e_max = float(data.get("num_edges", num_qubits))

    rows = []

    # 1. Polyak methods
    for opt_key, label in [
        ("polyak_gd", "Polyak -> GD"),
        ("polyak_qng", "Polyak -> QNG"),
    ]:
        if opt_key in data:
            runs = np.array(
                [data[opt_key][f"loss{i}"] for i in range(1, n_runs + 1)]
            )
            rows.append(compute_metrics(label, runs, e_min, e_max))

    # 2. Baseline optimizers (ignoring 'spsa' and 'rotosolve')
    for opt in ["sgd", "adam", "qng"]:
        if opt in data:
            for lr_key in sorted(
                data[opt].keys(), key=lambda k: float(k.replace("lr_", ""))
            ):
                lr_val = lr_key.replace("lr_", "")
                runs = np.array(
                    [
                        data[opt][lr_key][f"loss{i}"]
                        for i in range(1, n_runs + 1)
                    ]
                )
                rows.append(
                    compute_metrics(
                        f"{opt.upper()} (lr={lr_val})", runs, e_min, e_max
                    )
                )

    df_cfg = pd.DataFrame(rows)
    print(f"\n{'='*38} Config: {cfg} (GSE = {e_min:.3f}) {'='*38}")
    print(df_cfg.to_string(index=False))