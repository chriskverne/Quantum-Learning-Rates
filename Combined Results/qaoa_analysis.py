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

<<<<<<< HEAD
import json
import os
import numpy as np
import pandas as pd

# Switch base_dir to: "./QAOA/Cycle", "./QAOA/Maxcut", or "./QAOA/Reyni_Random"
base_dir = "./QAOA/Reyni_Random"
=======







# import json
# import os
# import numpy as np

# # Switch base_dir to: "./QAOA/Cycle", "./QAOA/Maxcut", or "./QAOA/Reyni_Random"
# base_dir = "./QAOA/Cycle"
# configs = ["4q_2l", "6q_4l", "8q_6l", "10q_8l", "12q_10l", "14q_10l"]
# n_steps = 50
# n_runs = 5

# # Fixed learning-rate columns (same as the VQE table)
# lrs = ["0.001", "0.01", "0.1", "0.5"]

# MAXCUT_GSE = {4: -2, 6: -5, 8: -8, 10: -11, 12: -13, 14: -15}


# def compute_metrics(runs_matrix, e_min):
#     if runs_matrix is None or len(runs_matrix) == 0:
#         return "-", "-", "-"

#     best_runs = np.minimum.accumulate(runs_matrix, axis=1)

#     # 1. Final Energy
#     final_energies = best_runs[:, -1]
#     mean_final, std_final = np.mean(final_energies), np.std(final_energies)

#     # 2. Steps to 95% of OWN progress
#     steps = []
#     for r in range(len(runs_matrix)):
#         e_start = runs_matrix[r, 0]
#         e_best = final_energies[r]

#         if np.isclose(e_start, e_best):
#             steps.append(n_steps)
#         else:
#             target_energy = e_start - 0.95 * (e_start - e_best)
#             step_idx = np.where(best_runs[r] <= target_energy)[0][0]
#             steps.append(step_idx + 1)

#     # 3. Regret AUC
#     auc_runs = np.sum(best_runs - e_min, axis=1)

#     # Order: AUC, Energy, Steps
#     auc_str = f"${np.mean(auc_runs):.1f} \\pm {np.std(auc_runs):.1f}$"
#     eng_str = f"${mean_final:.3f} \\pm {std_final:.2f}$"
#     step_str = f"${np.mean(steps):.1f} \\pm {np.std(steps):.1f}$"

#     return auc_str, eng_str, step_str


# def extract_runs(data, key1, key2=None):
#     try:
#         if key2 is None:
#             runs = [data[key1][f"loss{i}"] for i in range(1, n_runs + 1)]
#         else:
#             runs = [data[key1][key2][f"loss{i}"] for i in range(1, n_runs + 1)]
#         return np.array(runs)
#     except (KeyError, TypeError):
#         return None


# # =======================================================
# # Build the LaTeX string
# # =======================================================
# problem_name = os.path.basename(os.path.normpath(base_dir))

# latex_lines = [
#     "\\begin{table}[htpb]",
#     "\\centering",
#     "\\footnotesize",
#     "\\setlength{\\tabcolsep}{3pt} % Compress spacing slightly to fit 15 columns",
#     "\\resizebox{\\textwidth}{!}{%",
#     "\\begin{tabular}{l cc cccc cccc cccc}",
#     "\\toprule",
#     " & \\multicolumn{2}{c}{\\textbf{Polyak}} & \\multicolumn{4}{c}{\\textbf{SGD}} & \\multicolumn{4}{c}{\\textbf{ADAM}} & \\multicolumn{4}{c}{\\textbf{QNG}} \\\\",
#     "\\cmidrule(lr){2-3} \\cmidrule(lr){4-7} \\cmidrule(lr){8-11} \\cmidrule(lr){12-15}",
#     "\\textbf{Metric} & GD & QNG & 0.001 & 0.01 & 0.1 & 0.5 & 0.001 & 0.01 & 0.1 & 0.5 & 0.001 & 0.01 & 0.1 & 0.5 \\\\",
# ]

# for cfg in configs:
#     json_path = os.path.join(base_dir, f"{cfg}.json")
#     if not os.path.exists(json_path):
#         json_path = os.path.join(base_dir, cfg, "results.json")
#     if not os.path.exists(json_path):
#         print(f"% Skipped {cfg} (Not found)")
#         continue

#     with open(json_path, "r") as f:
#         data = json.load(f)

#     num_qubits = data["num_qubits"]
#     num_layers = data.get("num_layers", "N/A")
#     dir_name = base_dir.lower()

#     # --- 3-WAY BRANCH FOR GSE & E_MAX (unchanged) ---
#     if "cycle" in dir_name:
#         e_min = -num_qubits if num_qubits % 2 == 0 else -(num_qubits - 2)
#         e_max = float(num_qubits)

#     elif "maxcut" in dir_name:
#         e_min = float(MAXCUT_GSE.get(num_qubits, 4 - 1.5 * num_qubits))
#         e_max = float(1.5 * num_qubits)

#     elif "reyni" in dir_name or "renyi" in dir_name:
#         e_min = float(data["gse"])
#         # Uses total edge count |E| as e_max (consistent with Cycle and MaxCut3)
#         e_max = float(data.get("num_edges", num_qubits))

#     else:
#         e_min = float(data.get("gse", -num_qubits))
#         e_max = float(data.get("num_edges", num_qubits))

#     e_min = float(e_min)

#     # Span row for configuration
#     latex_lines.append("\\midrule")
#     latex_lines.append(
#         f"\\multicolumn{{15}}{{l}}{{\\textbf{{{num_qubits} Qubits, {num_layers} Layers \\quad (GSE = ${e_min:.3f}$)}}}} \\\\"
#     )
#     latex_lines.append("\\midrule")

#     row_auc = ["Regret AUC"]
#     row_eng = ["Final Energy"]
#     row_step = ["Steps to 95\\%"]

#     # 1. Polyak methods
#     for key in ["polyak_gd", "polyak_qng"]:
#         runs = extract_runs(data, key)
#         auc, eng, step = compute_metrics(runs, e_min)
#         row_auc.append(auc)
#         row_eng.append(eng)
#         row_step.append(step)

#     # 2. Baseline optimizers (SGD, ADAM, QNG; ignoring 'spsa' and 'rotosolve')
#     for opt in ["sgd", "adam", "qng"]:
#         for lr in lrs:
#             runs = extract_runs(data, opt, f"lr_{lr}")
#             auc, eng, step = compute_metrics(runs, e_min)
#             row_auc.append(auc)
#             row_eng.append(eng)
#             row_step.append(step)

#     latex_lines.append(" & ".join(row_auc) + " \\\\")
#     latex_lines.append(" & ".join(row_eng) + " \\\\")
#     latex_lines.append(" & ".join(row_step) + " \\\\")

# # Finish table
# latex_lines.extend([
#     "\\bottomrule",
#     "\\end{tabular}%",
#     "}",
#     f"\\caption{{QAOA ({problem_name}) Optimization Benchmark Results.}}",
#     f"\\label{{tab:qaoa_{problem_name.lower()}_benchmark_wide}}",
#     "\\end{table}",
# ])

# print("\n".join(latex_lines))








import json
import os
import numpy as np

# Switch base_dir to: "./QAOA/Cycle", "./QAOA/Maxcut", or "./QAOA/Reyni_Random"
base_dir = "./QAOA/Reyni2"
>>>>>>> 40990d12dff0e8d233e1a4d21814d80bbe32904c
configs = ["4q_2l", "6q_4l", "8q_6l", "10q_8l", "12q_10l", "14q_10l"]
n_steps = 50
n_runs = 5

<<<<<<< HEAD
MAXCUT_GSE = {4: -2, 6: -5, 8: -8, 10: -11, 12: -13, 14: -15}


def compute_metrics(name, runs_matrix, e_min, e_max):
=======
# Fixed learning-rate columns (same as the VQE table)
lrs = ["0.001", "0.01", "0.1", "0.5"]

MAXCUT_GSE = {4: -2, 6: -5, 8: -8, 10: -11, 12: -13, 14: -15}


def compute_metrics(runs_matrix, e_min):
    if runs_matrix is None or len(runs_matrix) == 0:
        return "-", "-", "-", np.nan, np.nan

>>>>>>> 40990d12dff0e8d233e1a4d21814d80bbe32904c
    best_runs = np.minimum.accumulate(runs_matrix, axis=1)

    # 1. Final Energy
    final_energies = best_runs[:, -1]
    mean_final, std_final = np.mean(final_energies), np.std(final_energies)

<<<<<<< HEAD
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
=======
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
    mean_auc = np.mean(auc_runs)

    # Order: AUC, Energy, Steps
    auc_str = f"\({mean_auc:.1f} \\pm {np.std(auc_runs):.1f}\)"
    eng_str = f"\({mean_final:.3f} \\pm {std_final:.2f}\)"
    step_str = f"\({np.mean(steps):.1f} \\pm {np.std(steps):.1f}\)"

    return auc_str, eng_str, step_str, mean_auc, mean_final


def extract_runs(data, key1, key2=None):
    try:
        if key2 is None:
            runs = [data[key1][f"loss{i}"] for i in range(1, n_runs + 1)]
        else:
            runs = [data[key1][key2][f"loss{i}"] for i in range(1, n_runs + 1)]
        return np.array(runs)
    except (KeyError, TypeError):
        return None


def get_ranks(values):
    """
    Computes 1-based ranks (lowest metric value gets rank 1).
    Handles ties by assigning the average rank.
    """
    arr = np.array(values, dtype=float)
    arr_clean = np.where(np.isnan(arr), np.inf, arr)

    order = np.argsort(arr_clean)
    ranks = np.empty(len(arr), dtype=float)
    ranks[order] = np.arange(1, len(arr) + 1)

    # Resolve ties to average rank
    for val in np.unique(arr_clean):
        mask = arr_clean == val
        ranks[mask] = np.mean(ranks[mask])

    return ranks


def format_rank_row(mean_ranks, sgd_indices, qng_indices):
    """
    Formats the average rank row:
    - Best SGD performer in bold blue
    - Best QNG performer in bold red
    - Overall 1st place in bold (if not already SGD/QNG)
    """
    min_overall = np.min(mean_ranks)
    min_sgd = np.min(mean_ranks[sgd_indices])
    min_qng = np.min(mean_ranks[qng_indices])

    formatted = []
    for i, rank in enumerate(mean_ranks):
        val_str = f"{rank:.2f}"
        is_best_sgd = (i in sgd_indices) and np.isclose(rank, min_sgd)
        is_best_qng = (i in qng_indices) and np.isclose(rank, min_qng)
        is_best_overall = np.isclose(rank, min_overall)

        if is_best_sgd:
            formatted.append(f"\\textbf{{\\textcolor{{blue}}{{{val_str}}}}}")
        elif is_best_qng:
            formatted.append(f"\\textbf{{\\textcolor{{red}}{{{val_str}}}}}")
        elif is_best_overall:
            formatted.append(f"\\textbf{{{val_str}}}")
        else:
            formatted.append(val_str)

    return formatted


# =======================================================
# Build the LaTeX string for Main Table
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

# Track ranks across problems
all_auc_ranks = []
all_eng_ranks = []
>>>>>>> 40990d12dff0e8d233e1a4d21814d80bbe32904c

for cfg in configs:
    json_path = os.path.join(base_dir, f"{cfg}.json")
    if not os.path.exists(json_path):
        json_path = os.path.join(base_dir, cfg, "results.json")
    if not os.path.exists(json_path):
<<<<<<< HEAD
=======
        print(f"% Skipped {cfg} (Not found)")
>>>>>>> 40990d12dff0e8d233e1a4d21814d80bbe32904c
        continue

    with open(json_path, "r") as f:
        data = json.load(f)

    num_qubits = data["num_qubits"]
<<<<<<< HEAD
=======
    num_layers = data.get("num_layers", "N/A")
>>>>>>> 40990d12dff0e8d233e1a4d21814d80bbe32904c
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
<<<<<<< HEAD
        # Uses total edge count |E| as e_max (consistent with Cycle and MaxCut3)
=======
>>>>>>> 40990d12dff0e8d233e1a4d21814d80bbe32904c
        e_max = float(data.get("num_edges", num_qubits))

    else:
        e_min = float(data.get("gse", -num_qubits))
        e_max = float(data.get("num_edges", num_qubits))

<<<<<<< HEAD
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
=======
    e_min = float(e_min)

    # Span row for configuration
    latex_lines.append("\\midrule")
    latex_lines.append(
        f"\\multicolumn{{15}}{{l}}{{\\textbf{{{num_qubits} Qubits, {num_layers} Layers \\quad (GSE = \({e_min:.3f}\))}}}} \\\\"
    )
    latex_lines.append("\\midrule")

    row_auc = ["Regret AUC"]
    row_eng = ["Final Energy"]
    row_step = ["Steps to 95\\%"]

    cfg_auc_vals = []
    cfg_eng_vals = []

    # 1. Polyak methods (indices 0, 1)
    for key in ["polyak_gd", "polyak_qng"]:
        runs = extract_runs(data, key)
        auc, eng, step, raw_auc, raw_eng = compute_metrics(runs, e_min)
        row_auc.append(auc)
        row_eng.append(eng)
        row_step.append(step)
        cfg_auc_vals.append(raw_auc)
        cfg_eng_vals.append(raw_eng)

    # 2. Baseline optimizers: SGD (2..5), ADAM (6..9), QNG (10..13)
    for opt in ["sgd", "adam", "qng"]:
        for lr in lrs:
            runs = extract_runs(data, opt, f"lr_{lr}")
            auc, eng, step, raw_auc, raw_eng = compute_metrics(runs, e_min)
            row_auc.append(auc)
            row_eng.append(eng)
            row_step.append(step)
            cfg_auc_vals.append(raw_auc)
            cfg_eng_vals.append(raw_eng)

    latex_lines.append(" & ".join(row_auc) + " \\\\")
    latex_lines.append(" & ".join(row_eng) + " \\\\")
    latex_lines.append(" & ".join(row_step) + " \\\\")

    # Record ranks for this problem size
    all_auc_ranks.append(get_ranks(cfg_auc_vals))
    all_eng_ranks.append(get_ranks(cfg_eng_vals))

# Finish Main Table
latex_lines.extend([
    "\\bottomrule",
    "\\end{tabular}%",
    "}",
    f"\\caption{{QAOA ({problem_name}) Optimization Benchmark Results.}}",
    f"\\label{{tab:qaoa_{problem_name.lower()}_benchmark_wide}}",
    "\\end{table}",
])

# =======================================================
# Build the LaTeX string for Average Ranks Table
# =======================================================
rank_table_lines = []

if all_auc_ranks:
    mean_auc_ranks = np.mean(all_auc_ranks, axis=0)
    mean_eng_ranks = np.mean(all_eng_ranks, axis=0)

    # Column index groupings across the 14 optimizers:
    # 0: Polyak GD, 1: Polyak QNG
    # 2..5: SGD (0.001, 0.01, 0.1, 0.5)
    # 6..9: ADAM (0.001, 0.01, 0.1, 0.5)
    # 10..13: QNG (0.001, 0.01, 0.1, 0.5)
    sgd_indices = [0, 2, 3, 4, 5]
    qng_indices = [1, 10, 11, 12, 13]

    rank_row_auc = ["Regret AUC"] + format_rank_row(mean_auc_ranks, sgd_indices, qng_indices)
    rank_row_eng = ["Final Energy"] + format_rank_row(mean_eng_ranks, sgd_indices, qng_indices)

    rank_table_lines = [
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
        "\\midrule",
        " & ".join(rank_row_auc) + " \\\\",
        " & ".join(rank_row_eng) + " \\\\",
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        f"\\caption{{Average Optimizer Placements/Ranks across Problem Configurations (4--14 Qubits) for QAOA ({problem_name}). First place in \\textbf{{bold}}, best SGD performer in \\textbf{{\\textcolor{{blue}}{{bold blue}}}}, and best QNG performer in \\textbf{{\\textcolor{{red}}{{bold red}}}} (lower rank is better).}}",
        f"\\label{{tab:qaoa_{problem_name.lower()}_ranks}}",
        "\\end{table}",
    ]

# Print both tables
print("\n".join(latex_lines))
if rank_table_lines:
    print("\n" + "\n".join(rank_table_lines))
>>>>>>> 40990d12dff0e8d233e1a4d21814d80bbe32904c
