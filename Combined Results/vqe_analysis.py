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
<<<<<<< HEAD
=======




# import json
# import os
# import numpy as np

# # 1. Verify this path matches your directory exactly
# base_dir = "./VQE/XY"
# configs = ["4q_2l", "6q_3l", "8q_4l", "10q_5l", "12q_6l", "14q_7l"]
# n_steps = 50
# n_runs = 5

# # Define the exact learning rates we expect to find for the baselines
# lrs = ["0.001", "0.01", "0.1", "0.5"]

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

#     # Note the specific order required: AUC, Energy, Steps
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
#     "\\textbf{Metric} & GD & QNG & 0.001 & 0.01 & 0.1 & 0.5 & 0.001 & 0.01 & 0.1 & 0.5 & 0.001 & 0.01 & 0.1 & 0.5 \\\\"
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

#     num_qubits = data.get("num_qubits", "N/A")
#     num_layers = data.get("num_layers", "N/A")
#     e_min = float(data["gse"])
    
#     # Span row for configuration
#     latex_lines.append("\\midrule")
#     latex_lines.append(f"\\multicolumn{{15}}{{l}}{{\\textbf{{{num_qubits} Qubits, {num_layers} Layers \\quad (GSE = ${e_min:.3f}$)}}}} \\\\")
#     latex_lines.append("\\midrule")

#     # Arrays to hold the row data
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

#     # 2. Baseline optimizers (SGD, ADAM, QNG)
#     for opt in ["sgd", "adam", "qng"]:
#         for lr in lrs:
#             runs = extract_runs(data, opt, f"lr_{lr}")
#             auc, eng, step = compute_metrics(runs, e_min)
#             row_auc.append(auc)
#             row_eng.append(eng)
#             row_step.append(step)

#     # Join and append to table
#     latex_lines.append(" & ".join(row_auc) + " \\\\")
#     latex_lines.append(" & ".join(row_eng) + " \\\\")
#     latex_lines.append(" & ".join(row_step) + " \\\\")

# # Finish table
# latex_lines.extend([
#     "\\bottomrule",
#     "\\end{tabular}%",
#     "}",
#     "\\caption{VQE Optimization Benchmark Results.}",
#     "\\label{tab:vqe_benchmark_wide}",
#     "\\end{table}"
# ])

# # Print final result
# print("\n".join(latex_lines))


>>>>>>> 40990d12dff0e8d233e1a4d21814d80bbe32904c
import json
import os
import numpy as np

<<<<<<< HEAD
# 1. Verify this path matches your directory exactly
base_dir = "./VQE/XY"
=======
# Paths = "./VQE/XY", "./VQE/H2"
base_dir = "./VQE/TFIM_ISING"
>>>>>>> 40990d12dff0e8d233e1a4d21814d80bbe32904c
configs = ["4q_2l", "6q_3l", "8q_4l", "10q_5l", "12q_6l", "14q_7l"]
n_steps = 50
n_runs = 5

# Define the exact learning rates we expect to find for the baselines
lrs = ["0.001", "0.01", "0.1", "0.5"]

<<<<<<< HEAD
def compute_metrics(runs_matrix, e_min):
    if runs_matrix is None or len(runs_matrix) == 0:
        return "-", "-", "-"
=======

def compute_metrics(runs_matrix, e_min):
    if runs_matrix is None or len(runs_matrix) == 0:
        return "-", "-", "-", np.nan, np.nan
>>>>>>> 40990d12dff0e8d233e1a4d21814d80bbe32904c

    best_runs = np.minimum.accumulate(runs_matrix, axis=1)

    # 1. Final Energy
    final_energies = best_runs[:, -1]
    mean_final, std_final = np.mean(final_energies), np.std(final_energies)

    # 2. Steps to 95% of OWN progress
    steps = []
    for r in range(len(runs_matrix)):
        e_start = runs_matrix[r, 0]
        e_best = final_energies[r]
<<<<<<< HEAD
        
=======

>>>>>>> 40990d12dff0e8d233e1a4d21814d80bbe32904c
        if np.isclose(e_start, e_best):
            steps.append(n_steps)
        else:
            target_energy = e_start - 0.95 * (e_start - e_best)
            step_idx = np.where(best_runs[r] <= target_energy)[0][0]
            steps.append(step_idx + 1)

    # 3. Regret AUC
    auc_runs = np.sum(best_runs - e_min, axis=1)
<<<<<<< HEAD

    # Note the specific order required: AUC, Energy, Steps
    auc_str = f"${np.mean(auc_runs):.1f} \\pm {np.std(auc_runs):.1f}$"
    eng_str = f"${mean_final:.3f} \\pm {std_final:.2f}$"
    step_str = f"${np.mean(steps):.1f} \\pm {np.std(steps):.1f}$"

    return auc_str, eng_str, step_str
=======
    mean_auc = np.mean(auc_runs)

    # Note the specific order required: AUC, Energy, Steps
    auc_str = f"\({mean_auc:.1f} \\pm {np.std(auc_runs):.1f}\)"
    eng_str = f"\({mean_final:.3f} \\pm {std_final:.2f}\)"
    step_str = f"\({np.mean(steps):.1f} \\pm {np.std(steps):.1f}\)"

    return auc_str, eng_str, step_str, mean_auc, mean_final

>>>>>>> 40990d12dff0e8d233e1a4d21814d80bbe32904c

def extract_runs(data, key1, key2=None):
    try:
        if key2 is None:
            runs = [data[key1][f"loss{i}"] for i in range(1, n_runs + 1)]
        else:
            runs = [data[key1][key2][f"loss{i}"] for i in range(1, n_runs + 1)]
        return np.array(runs)
    except (KeyError, TypeError):
        return None

<<<<<<< HEAD
# =======================================================
# Build the LaTeX string
# =======================================================
=======

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
# Build LaTeX for Main Benchmark Table
# =======================================================
benchmark_name = os.path.basename(os.path.normpath(base_dir))

>>>>>>> 40990d12dff0e8d233e1a4d21814d80bbe32904c
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
<<<<<<< HEAD
    "\\textbf{Metric} & GD & QNG & 0.001 & 0.01 & 0.1 & 0.5 & 0.001 & 0.01 & 0.1 & 0.5 & 0.001 & 0.01 & 0.1 & 0.5 \\\\"
]

=======
    "\\textbf{Metric} & GD & QNG & 0.001 & 0.01 & 0.1 & 0.5 & 0.001 & 0.01 & 0.1 & 0.5 & 0.001 & 0.01 & 0.1 & 0.5 \\\\",
]

# Track placements across problem sizes
all_auc_ranks = []
all_eng_ranks = []

>>>>>>> 40990d12dff0e8d233e1a4d21814d80bbe32904c
for cfg in configs:
    json_path = os.path.join(base_dir, f"{cfg}.json")
    if not os.path.exists(json_path):
        json_path = os.path.join(base_dir, cfg, "results.json")
<<<<<<< HEAD
    
=======

>>>>>>> 40990d12dff0e8d233e1a4d21814d80bbe32904c
    if not os.path.exists(json_path):
        print(f"% Skipped {cfg} (Not found)")
        continue

    with open(json_path, "r") as f:
        data = json.load(f)

    num_qubits = data.get("num_qubits", "N/A")
    num_layers = data.get("num_layers", "N/A")
    e_min = float(data["gse"])
<<<<<<< HEAD
    
    # Span row for configuration
    latex_lines.append("\\midrule")
    latex_lines.append(f"\\multicolumn{{15}}{{l}}{{\\textbf{{{num_qubits} Qubits, {num_layers} Layers \\quad (GSE = ${e_min:.3f}$)}}}} \\\\")
    latex_lines.append("\\midrule")

    # Arrays to hold the row data
=======

    # Span row for configuration
    latex_lines.append("\\midrule")
    latex_lines.append(
        f"\\multicolumn{{15}}{{l}}{{\\textbf{{{num_qubits} Qubits, {num_layers} Layers \\quad (GSE = \({e_min:.3f}\))}}}} \\\\"
    )
    latex_lines.append("\\midrule")

    # Arrays to hold row data
>>>>>>> 40990d12dff0e8d233e1a4d21814d80bbe32904c
    row_auc = ["Regret AUC"]
    row_eng = ["Final Energy"]
    row_step = ["Steps to 95\\%"]

<<<<<<< HEAD
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
=======
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
>>>>>>> 40990d12dff0e8d233e1a4d21814d80bbe32904c

    # Join and append to table
    latex_lines.append(" & ".join(row_auc) + " \\\\")
    latex_lines.append(" & ".join(row_eng) + " \\\\")
    latex_lines.append(" & ".join(row_step) + " \\\\")

<<<<<<< HEAD
# Finish table
=======
    # Compute ranks for this configuration
    all_auc_ranks.append(get_ranks(cfg_auc_vals))
    all_eng_ranks.append(get_ranks(cfg_eng_vals))

# Finish Main Table
>>>>>>> 40990d12dff0e8d233e1a4d21814d80bbe32904c
latex_lines.extend([
    "\\bottomrule",
    "\\end{tabular}%",
    "}",
<<<<<<< HEAD
    "\\caption{VQE Optimization Benchmark Results.}",
    "\\label{tab:vqe_benchmark_wide}",
    "\\end{table}"
])

# Print final result
print("\n".join(latex_lines))
=======
    f"\\caption{{VQE ({benchmark_name}) Optimization Benchmark Results.}}",
    f"\\label{{tab:vqe_{benchmark_name.lower()}_benchmark_wide}}",
    "\\end{table}",
])

# =======================================================
# Build LaTeX for Average Ranks Table
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

    rank_row_auc = ["Regret AUC"] + format_rank_row(
        mean_auc_ranks, sgd_indices, qng_indices
    )
    rank_row_eng = ["Final Energy"] + format_rank_row(
        mean_eng_ranks, sgd_indices, qng_indices
    )

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
        f"\\caption{{Average Optimizer Placements/Ranks across Problem Configurations (4--14 Qubits) for VQE ({benchmark_name}). First place in \\textbf{{bold}}, best SGD performer in \\textbf{{\\textcolor{{blue}}{{bold blue}}}}, and best QNG performer in \\textbf{{\\textcolor{{red}}{{bold red}}}} (lower rank is better).}}",
        f"\\label{{tab:vqe_{benchmark_name.lower()}_ranks}}",
        "\\end{table}",
    ]

# Print both tables
print("\n".join(latex_lines))
if rank_table_lines:
    print("\n" + "\n".join(rank_table_lines))
>>>>>>> 40990d12dff0e8d233e1a4d21814d80bbe32904c
