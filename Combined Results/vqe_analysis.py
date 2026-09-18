import json

example = "./VQE/TFIM_ISING/4q_2l.json"
configs = ["4q_2l", "6q_3l", "8q_4l", "10q_5l", "12q_6l", "14q_7l"]

"""
TFIM_ISING
4q: -4.75877048
6q: -7.29622981
8q: -9.83795145
10q: -12.38149000
12q: -14.92597111
14q: -17.47100405

XY
-4.47213595
-6.98791841
-9.51754097
-12.05334837
-14.59245962
-17.13354447
"""

# with open(example, "r") as f:
#     data = json.load(f)

# # Top-level keys only
# print(list(data.keys()))
import json
import os
import numpy as np

# 1. Verify this path matches your directory exactly
base_dir = "./VQE/XY"
configs = ["4q_2l", "6q_3l", "8q_4l", "10q_5l", "12q_6l", "14q_7l"]
n_steps = 50
n_runs = 5

# Define the exact learning rates we expect to find for the baselines
lrs = ["0.001", "0.01", "0.1", "0.5"]

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

    # Note the specific order required: AUC, Energy, Steps
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
    "\\textbf{Metric} & GD & QNG & 0.001 & 0.01 & 0.1 & 0.5 & 0.001 & 0.01 & 0.1 & 0.5 & 0.001 & 0.01 & 0.1 & 0.5 \\\\"
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

    num_qubits = data.get("num_qubits", "N/A")
    num_layers = data.get("num_layers", "N/A")
    e_min = float(data["gse"])
    
    # Span row for configuration
    latex_lines.append("\\midrule")
    latex_lines.append(f"\\multicolumn{{15}}{{l}}{{\\textbf{{{num_qubits} Qubits, {num_layers} Layers \\quad (GSE = ${e_min:.3f}$)}}}} \\\\")
    latex_lines.append("\\midrule")

    # Arrays to hold the row data
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

    # 2. Baseline optimizers (SGD, ADAM, QNG)
    for opt in ["sgd", "adam", "qng"]:
        for lr in lrs:
            runs = extract_runs(data, opt, f"lr_{lr}")
            auc, eng, step = compute_metrics(runs, e_min)
            row_auc.append(auc)
            row_eng.append(eng)
            row_step.append(step)

    # Join and append to table
    latex_lines.append(" & ".join(row_auc) + " \\\\")
    latex_lines.append(" & ".join(row_eng) + " \\\\")
    latex_lines.append(" & ".join(row_step) + " \\\\")

# Finish table
latex_lines.extend([
    "\\bottomrule",
    "\\end{tabular}%",
    "}",
    "\\caption{VQE Optimization Benchmark Results.}",
    "\\label{tab:vqe_benchmark_wide}",
    "\\end{table}"
])

# Print final result
print("\n".join(latex_lines))