import re
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
cmap = plt.get_cmap('nipy_spectral')

plot_directory = Path(__file__).resolve().parent
results_directory = plot_directory.parent

all_methods = ["Trotter", "Bliss_Pauli", "Bliss_THC"]
methods_legend = ["Trotter", "BLISS-Pauli", "BLISS-THC"]
all_mol_names = ["H2S", "CO2", "CH2O", "CH4"]

all_num_modes = [3, 4, 6, 9]
method_markers = {"Triangular":"s", "Pauli":"o", "Pauli_clean":"o", "Bliss_Pauli":"s", "Bliss_Pauli_clean":"v", "THC":"o", "THC_clean":"o", "Bliss_THC":"s", "Bliss_THC_clean":"v", "Trotter":"s"}
method_colors = {"Triangular":cmap(0.2), "Pauli":cmap(0.44), "Pauli_clean":cmap(0.62), "Bliss_Pauli":cmap(0.44), "Bliss_Pauli_clean":cmap(0.62), "THC":cmap(0.75), "THC_clean":cmap(0.95), "Bliss_THC":cmap(0.75), "Bliss_THC_clean":cmap(0.95), "Trotter":cmap(0.2)}

all_one_norms = {}
all_T_gates = {}


def path_extractor(method, mol_name):
    if method == "Bliss_Pauli":
        folder = results_directory / "Bliss_Pauli_params" / mol_name / "HO_basis"
        pattern = "cost_4M_2MC_eps_*.txt"

    elif method == "Bliss_THC":
        folder = results_directory / "Bliss_THC_params" / mol_name / "HO_basis"
        pattern = "cost_4M_hR_2MC_eps_*.txt"

    else:
        raise ValueError(f"Unknown method: {method}")

    matches = list(folder.glob(pattern))

    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one file matching '{pattern}' in '{folder}', found {len(matches)}."
        )

    return str(matches[0])


def extract_metrics(filepath):
    """
    Extracts the T-gate cost, 1-norm, and qubit cost for a given method
    and molecule from a specific text file.
    """

    number_pattern = r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?'

    toffoli_regex = re.compile(rf'Toffoli cost.*?({number_pattern})[^\d]*$', re.IGNORECASE)
    norm_regex = re.compile(rf'(?:1-norm|one norm):\s*({number_pattern})', re.IGNORECASE)
    qubit_regex = re.compile(rf'Qubit cost.*?({number_pattern})', re.IGNORECASE)

    metrics = {'T': None, 'norm': None, 'qubit': None}

    FilePath = Path(filepath)

    if not FilePath.is_file():
        print(f"Warning: File not found at {FilePath}")
        return metrics

    try:
        with FilePath.open('r', encoding='utf-8') as f:
            for line in f:
                toffoli_match = toffoli_regex.search(line)
                if toffoli_match:
                    metrics['T'] = 4 * float(toffoli_match.group(1))

                norm_match = norm_regex.search(line)
                if norm_match:
                    metrics['norm'] = float(norm_match.group(1))

                qubit_match = qubit_regex.search(line)
                if qubit_match:
                    metrics['qubit'] = float(qubit_match.group(1))

    except Exception as e:
        print(f"Error reading {filepath}: {e}")

    return metrics


results = {}
for _mol in all_mol_names:
    results[_mol] = {}
    for _method in all_methods:
        if _method != "Trotter":
            filepath = path_extractor(_method, _mol)
            results[_mol][_method] = extract_metrics(filepath)

trot_alpha = 5516
eps_QPE = 3
num_trot_calls = np.ceil(trot_alpha / (4 * eps_QPE))
QPE_qubits = 2 * np.ceil(np.log2(num_trot_calls))

results["H2S"]["Trotter"] = {'T': num_trot_calls * 2 * 1.71e4, 'qubit': QPE_qubits + 12}
results["CO2"]["Trotter"] = {'T': num_trot_calls * 3 * 2.58e4, 'qubit': QPE_qubits + 16}
results["CH2O"]["Trotter"] = {'T': num_trot_calls * 2 * 1.11e5, 'qubit': QPE_qubits + 24}
results["CH4"]["Trotter"] = {'T': num_trot_calls * 5 * 2.28e5, 'qubit': QPE_qubits + 36}


def plot_cost_metrics(results, all_num_modes, all_mol_names, method_markers, method_colors):
    """
    Creates two grouped bar charts side-by-side:
        1. T-gate cost
        2. Number of qubits
    """

    all_methods = ["Trotter", "Bliss_Pauli", "Bliss_THC"]
    qubit_methods = ["Trotter", "Bliss_Pauli", "Bliss_THC"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    x_positions = np.arange(len(all_mol_names))

    bar_width = 0.8 / len(all_methods)
    total_group_width = bar_width * len(all_methods)

    for i, method in enumerate(all_methods[::-1]):
        y_vals = []
        for mol in all_mol_names:
            val = results.get(mol, {}).get(method, {}).get("T")
            y_vals.append(val if val is not None else 0)

        offset = x_positions - total_group_width / 2 + bar_width / 2 + i * bar_width

        if any(y != 0 for y in y_vals):
            ax1.bar(
                offset,
                y_vals,
                bar_width,
                color=method_colors.get(method, "gray"),
                alpha=0.8,
                edgecolor="black",
                linewidth=0.5
            )

    bar_width = 0.8 / len(qubit_methods)
    total_group_width = bar_width * len(qubit_methods)

    for i, method in enumerate(qubit_methods):
        y_vals = []
        for mol in all_mol_names:
            val = results.get(mol, {}).get(method, {}).get("qubit")
            y_vals.append(val if val is not None else 0)

        offset = x_positions - total_group_width / 2 + bar_width / 2 + i * bar_width

        if any(y != 0 for y in y_vals):
            ax2.bar(
                offset,
                y_vals,
                bar_width,
                color=method_colors.get(method, "gray"),
                alpha=0.8,
                edgecolor="black",
                linewidth=0.5
            )

    all_mol_names_tex = ["H$_2$S", "CO$_2$", "CH$_2$O", "CH$_4$"]
    combined_labels = [
        f"{mol.upper()}"
        for size, mol in zip(all_num_modes, all_mol_names_tex)
    ]

    ax1.set_yscale("log")
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(combined_labels, rotation=0, ha="center", fontsize=18)
    ax1.tick_params(axis='y', labelsize=19)
    ax1.set_ylabel("T-gate cost", fontsize=18, fontweight="bold")
    ax1.set_xlabel("Molecules", fontsize=18, fontweight="bold")

    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(combined_labels, rotation=0, ha="center", fontsize=19)
    ax2.tick_params(axis='y', labelsize=19)
    ax2.set_ylabel("Qubit cost", fontsize=18, fontweight="bold")
    ax2.set_xlabel("Molecules", fontsize=18, fontweight="bold")

    legend_handles = []

    for i, method in enumerate(all_methods):
        handle = mpatches.Patch(
            facecolor=method_colors.get(method, "gray"),
            edgecolor="black",
            label=methods_legend[i]
        )
        legend_handles.append(handle)

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=len(all_methods) // 2 + 2,
        alignment="center",
        frameon=True,
        fontsize=16,
        title_fontsize=11
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    plt.savefig(plot_directory / "Trotter_vs_Qubitization.pdf", bbox_inches="tight")
    plt.show()


plot_cost_metrics(results, all_num_modes, all_mol_names, method_markers, method_colors)
