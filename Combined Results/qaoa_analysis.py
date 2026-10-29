import json
import os
import numpy as np
import pandas as pd

# Switch base_dir to: "./QAOA/Cycle", "./QAOA/Maxcut", or "./QAOA/Reyni_Random"
base_dir = "./QAOA/Cycle"
configs = ["4q_2l", "6q_4l", "8q_6l", "10q_8l", "12q_10l", "14q_10l"]
n_steps = 50
n_runs = 5

# Fixed learning-rate columns (same as the VQE table)
lrs = ["0.001", "0.01", "0.1", "0.5"]

MAXCUT_GSE = {4: -2, 6: -5, 8: -8, 10: -11, 12: -13, 14: -15}


def compute_metrics(runs_matrix, e_min):
    if runs_matrix is None or len(runs_matrix) == 0:
        return "-", "-", "-"

    best_runs = np.minimum.accumulate(runs_matrix, axis=1)

    # 1. Final Energy
    final_energies = best_runs[:, -1]
    mean_final, std_final = np.mean(final_energies), np.std(final_energies)

    # 2. Steps to 95% of OWN progress
    steps = []
    for r in range(len(runs_matrix)):
        e_start = runs_matrix[r, 0]
        e_best = final_energies[r]

        if np.isclose(e_start, e_best):
            steps.append(n_steps)
        else:
            target_energy = e_start - 0.95 * (e_start - e_best)
            step_idx = np.where(best_runs[r] <= target_energy)[0][0]
            steps.append(step_idx + 1)

    # 3. Regret AUC
    auc_runs = np.sum(best_runs - e_min, axis=1)

    # Order: AUC, Energy, Steps
    auc_str = f"${np.mean(auc_runs):.1f} \\pm {np.std(auc_runs):.1f}$"
    eng_str = f"${mean_final:.3f} \\pm {std_final:.2f}$"
    step_str = f"${np.mean(steps):.1f} \\pm {np.std(steps):.1f}$"

    return auc_str, eng_str, step_str


def extract_runs(data, key1, key2=None):
    try:
        if key2 is None:
            runs = [data[key1][f"loss{i}"] for i in range(1, n_runs + 1)]
        else:
            runs = [data[key1][key2][f"loss{i}"] for i in range(1, n_runs + 1)]
        return np.array(runs)
    except (KeyError, TypeError):
        return None


# =======================================================
# Build the LaTeX string
# =======================================================
problem_name = os.path.basename(os.path.normpath(base_dir))

latex_lines = [
    "\\begin{table}[htpb]",
    "\\centering",
    "\\footnotesize",
    "\\setlength{\\tabcolsep}{3pt} % Compress spacing slightly to fit 15 columns",
    "\\resizebox{\\textwidth}{!}{%",
    "\\begin{tabular}{l cc cccc cccc cccc}",
    "\\toprule",
    " & \\multicolumn{2}{c}{\\textbf{Polyak}} & \\multicolumn{4}{c}{\\textbf{SGD}} & \\multicolumn{4}{c}{\\textbf{ADAM}} & \\multicolumn{4}{c}{\\textbf{QNG}} \\\\",
    "\\cmidrule(lr){2-3} \\cmidrule(lr){4-7} \\cmidrule(lr){8-11} \\cmidrule(lr){12-15}",
    "\\textbf{Metric} & GD & QNG & 0.001 & 0.01 & 0.1 & 0.5 & 0.001 & 0.01 & 0.1 & 0.5 & 0.001 & 0.01 & 0.1 & 0.5 \\\\",
]

for cfg in configs:
    json_path = os.path.join(base_dir, f"{cfg}.json")
    if not os.path.exists(json_path):
        json_path = os.path.join(base_dir, cfg, "results.json")
    if not os.path.exists(json_path):
        print(f"% Skipped {cfg} (Not found)")
        continue

    with open(json_path, "r") as f:
        data = json.load(f)

    num_qubits = data["num_qubits"]
    num_layers = data.get("num_layers", "N/A")
    dir_name = base_dir.lower()

    # --- 3-WAY BRANCH FOR GSE & E_MAX (unchanged) ---
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

    e_min = float(e_min)

    # Span row for configuration
    latex_lines.append("\\midrule")
    latex_lines.append(
        f"\\multicolumn{{15}}{{l}}{{\\textbf{{{num_qubits} Qubits, {num_layers} Layers \\quad (GSE = ${e_min:.3f}$)}}}} \\\\"
    )
    latex_lines.append("\\midrule")

    row_auc = ["Regret AUC"]
    row_eng = ["Final Energy"]
    row_step = ["Steps to 95\\%"]

    # 1. Polyak methods
    for key in ["polyak_gd", "polyak_qng"]:
        runs = extract_runs(data, key)
        auc, eng, step = compute_metrics(runs, e_min)
        row_auc.append(auc)
        row_eng.append(eng)
        row_step.append(step)

    # 2. Baseline optimizers (SGD, ADAM, QNG; ignoring 'spsa' and 'rotosolve')
    for opt in ["sgd", "adam", "qng"]:
        for lr in lrs:
            runs = extract_runs(data, opt, f"lr_{lr}")
            auc, eng, step = compute_metrics(runs, e_min)
            row_auc.append(auc)
            row_eng.append(eng)
            row_step.append(step)

    latex_lines.append(" & ".join(row_auc) + " \\\\")
    latex_lines.append(" & ".join(row_eng) + " \\\\")
    latex_lines.append(" & ".join(row_step) + " \\\\")

# Finish table
latex_lines.extend([
    "\\bottomrule",
    "\\end{tabular}%",
    "}",
    f"\\caption{{QAOA ({problem_name}) Optimization Benchmark Results.}}",
    f"\\label{{tab:qaoa_{problem_name.lower()}_benchmark_wide}}",
    "\\end{table}",
])

print("\n".join(latex_lines))
