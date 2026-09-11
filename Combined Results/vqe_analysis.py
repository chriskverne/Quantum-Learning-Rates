import json

example = "./VQE/TFIM_ISING/4q_2l.json"
configs = ["4q_2l", "6q_3l", "8q_4l", "10q_5l", "12q_6l", "14q_7l"]

"""
TFIM_ISING
4q: −4.75877048
6q: −7.29622981
8q: −9.83795145
10q: −12.38149000
12q: −14.92597111
14q: −17.47100405

XY
−4.47213595
−6.98791841
−9.51754097
−12.05334837
−14.59245962
−17.13354447
"""

# with open(example, "r") as f:
#     data = json.load(f)

# # Top-level keys only
# print(list(data.keys()))

import json
import os
import numpy as np
import pandas as pd

# 1. Verify this path matches your directory exactly
base_dir = "./VQE/H2"
configs = ["4q_2l", "6q_3l", "8q_4l", "10q_5l", "12q_6l", "14q_7l"]
n_steps = 50
n_runs = 5

def compute_metrics(runs_matrix, e_min):
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

    return {
        "Final Energy": f"${mean_final:.3f} \\pm {std_final:.2f}$",
        "Steps to 95\\%": f"${np.mean(steps):.1f} \\pm {np.std(steps):.1f}$",
        "Regret AUC": f"${np.mean(auc_runs):.1f} \\pm {np.std(auc_runs):.1f}$",
    }

# Dictionary to hold the pivoted data
# Format: table_data[metric][optimizer][config_header] = value
metrics_list = ["Final Energy", "Steps to 95\\%", "Regret AUC"]
table_data = {m: {} for m in metrics_list}
config_headers = []

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
    
    # Create a nice multi-line LaTeX header for this column using \makecell
    header = f"\\makecell{{\\textbf{{{num_qubits}Q, {num_layers}L}} \\\\ GSE=${e_min:.3f}$}}"
    config_headers.append(header)

    runs_dict = {}

    # 1. Polyak methods
    for opt_key, label in [("polyak_gd", "Polyak GD"), ("polyak_qng", "Polyak QNG")]:
        if opt_key in data:
            runs_dict[label] = np.array([data[opt_key][f"loss{i}"] for i in range(1, n_runs + 1)])

    # 2. Baseline optimizers
    for opt in ["sgd", "adam", "qng"]:
        if opt in data:
            for lr_key in sorted(data[opt].keys(), key=lambda k: float(k.replace("lr_", ""))):
                lr_val = lr_key.replace("lr_", "")
                label = f"{opt.upper()} (lr={lr_val})"
                runs_dict[label] = np.array([data[opt][lr_key][f"loss{i}"] for i in range(1, n_runs + 1)])

    # Compute and store
    for opt_name, runs in runs_dict.items():
        res = compute_metrics(runs, e_min)
        for m in metrics_list:
            if opt_name not in table_data[m]:
                table_data[m][opt_name] = {}
            table_data[m][opt_name][header] = res[m]

# --- Custom LaTeX Table Generator ---
def generate_latex_table(metric_name, data_dict, columns):
    df = pd.DataFrame.from_dict(data_dict, orient='index')
    # Filter only available columns and set order
    cols_present = [c for c in columns if c in df.columns]
    df = df[cols_present]
    df.index.name = 'Optimizer'
    df.reset_index(inplace=True)

    col_format = "l" + "c" * (len(cols_present))
    
    lines = []
    lines.append("\\begin{table}[htpb]")
    lines.append("\\centering")
    # \resizebox ensures the table fits horizontally on the page
    lines.append("\\resizebox{\\textwidth}{!}{%") 
    lines.append(f"\\begin{{tabular}}{{{col_format}}}")
    lines.append("\\toprule")
    
    # Headers
    head_str = " & ".join([f"\\textbf{{{c}}}" if c == 'Optimizer' else c for c in df.columns])
    lines.append(head_str + " \\\\")
    lines.append("\\midrule")
    
    # Rows
    for _, row in df.iterrows():
        # Replace missing data with a hyphen if an optimizer didn't run for a specific config
        row_vals = [str(row[c]) if pd.notna(row[c]) else "-" for c in df.columns]
        lines.append(" & ".join(row_vals) + " \\\\")
        
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append(f"\\caption{{VQE Benchmark Results: \\textbf{{{metric_name}}}}}")
    
    # Clean label name
    clean_label = metric_name.lower().replace(" ", "_").replace("\\", "").replace("%", "")
    lines.append(f"\\label{{tab:vqe_{clean_label}}}")
    lines.append("\\end{table}\n")
    
    return "\n".join(lines)

# Print the final tables
for metric in metrics_list:
    print(f"% {'='*50}")
    print(f"% Table for {metric}")
    print(f"% {'='*50}\n")
    print(generate_latex_table(metric, table_data[metric], config_headers))