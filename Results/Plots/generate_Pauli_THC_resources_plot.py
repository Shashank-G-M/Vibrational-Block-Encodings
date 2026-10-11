import csv
import re
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import numpy as np


plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
cmap = plt.get_cmap('nipy_spectral')

# Toggle BLISS Toffoli ranges on or off. BLISS does not get separate qubit bars.
plot_bliss = True

all_mol_names = ["pah1", "pah2", "pah3", "pah4", "pah5", "pah6", "pah7"]
all_num_modes = [30, 48, 66, 84, 102, 120, 138]

method_markers = {
    "Triangular": "^",
    "Pauli": "o",
    "Bliss_Pauli": "s",
    "THC": "o",
    "Bliss_THC": "s",
}

# Pauli and THC use clean-QROAM colors; BLISS variants use distinct nearby colors.
method_colors = {
    "Triangular": cmap(0.2),
    "Pauli": cmap(0.44),
    "Bliss_Pauli": cmap(0.44),
    "THC": cmap(0.75),
    "Bliss_THC": cmap(0.75),
}

method_labels = {
    "Triangular": "Triangular",
    "Pauli": "Pauli",
    "Bliss_Pauli": "BLISS-Pauli",
    "THC": "THC",
    "Bliss_THC": "BLISS-THC",
}

method_directories = {
    "Pauli": "Pauli_params",
    "Bliss_Pauli": "Bliss_Pauli_params",
    "THC": "THC_params",
    "Bliss_THC": "Bliss_THC_params",
}

results_directory = Path(__file__).resolve().parent.parent


def extract_triangular_metrics(filepath):
    """Extract the Toffoli and qubit costs from a Triangular result file."""
    number_pattern = r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?'
    toffoli_regex = re.compile(rf'Toffoli cost.*?({number_pattern})[^\d]*$', re.IGNORECASE)
    qubit_regex = re.compile(rf'Qubit cost.*?({number_pattern})', re.IGNORECASE)
    metrics = {'Toffoli': None, 'qubit': None}
    filepath = Path(filepath)

    if not filepath.is_file():
        print(f"Warning: File not found at {filepath}")
        return metrics

    with filepath.open('r', encoding='utf-8') as input_file:
        for line in input_file:
            toffoli_match = toffoli_regex.search(line)
            if toffoli_match:
                metrics['Toffoli'] = float(toffoli_match.group(1))

            qubit_match = qubit_regex.search(line)
            if qubit_match:
                metrics['qubit'] = float(qubit_match.group(1))

    return metrics


def extract_qroam_metrics(filepath):
    """Extract the dirty baseline and full clean-QROAM cost ranges."""
    filepath = Path(filepath)
    if not filepath.is_file():
        print(f"Warning: File not found at {filepath}")
        return None

    with filepath.open('r', encoding='utf-8') as input_file:
        rows = list(csv.DictReader(input_file, delimiter='\t'))

    if not rows:
        print(f"Warning: No QROAM costs found in {filepath}")
        return None

    dirty_row = next((row for row in rows if row['QROAM_type'] == 'dirty'), None)
    if dirty_row is None:
        print(f"Warning: No dirty-QROAM baseline found in {filepath}")
        return None

    toffoli_costs = [float(row['Toffoli_cost']) for row in rows]
    qubit_costs = [float(row['qubit_cost']) for row in rows]

    return {
        'dirty_toffoli': float(dirty_row['Toffoli_cost']),
        'dirty_qubit': float(dirty_row['qubit_cost']),
        'min_toffoli': min(toffoli_costs),
        'max_toffoli': max(toffoli_costs),
        'max_qubit': max(qubit_costs),
    }


results = {}
for mol_name in all_mol_names:
    results[mol_name] = {
        "Triangular": extract_triangular_metrics(
            results_directory / "Triangular_params" / mol_name / "VSCF_basis"
            / "cost_5M_2MC_eps_1.txt"
        )
    }

    for method, directory in method_directories.items():
        filepath = (
            results_directory / directory / mol_name / "VSCF_basis"
            / "cost_QROAM_5M_hR_2MC_eps_1.txt"
        )
        results[mol_name][method] = extract_qroam_metrics(filepath)


def plot_cost_metrics(results, all_num_modes, all_mol_names, method_markers,
                      method_colors, include_bliss=True):
    """Plot Toffoli ranges and stacked select-copy/clean-ancilla qubit bars."""
    qroam_methods = ["Pauli", "THC"]
    if include_bliss:
        qroam_methods = ["Pauli", "Bliss_Pauli", "THC", "Bliss_THC"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7.5))

    # Small horizontal offsets prevent overlapping QROAM ranges at each molecule.
    range_offsets = dict(zip(qroam_methods, np.linspace(-4.0, 4.0, len(qroam_methods))))

    # Toffoli costs: Triangular is a point; each QROAM method spans its full range.
    for x_idx, mol_name in enumerate(all_mol_names):
        mode_count = all_num_modes[x_idx]
        triangular_cost = results[mol_name]["Triangular"].get('Toffoli')
        if triangular_cost is not None:
            ax1.scatter(
                mode_count, triangular_cost,
                color=method_colors["Triangular"], marker=method_markers["Triangular"],
                s=80, alpha=0.8, edgecolor='black', linewidth=0.5, zorder=3,
            )

        for method in qroam_methods:
            metrics = results[mol_name].get(method)
            if metrics is None:
                continue

            baseline = metrics['dirty_toffoli']
            lower_error = baseline - metrics['min_toffoli']
            upper_error = metrics['max_toffoli'] - baseline
            ax1.errorbar(
                mode_count + range_offsets[method], baseline,
                yerr=np.array([[lower_error], [upper_error]]),
                fmt='none', color=method_colors[method],
                elinewidth=2.5, capsize=0,
                alpha=0.85, zorder=2,
            )
            ax1.scatter(
                [mode_count + range_offsets[method]] * 2,
                [metrics['min_toffoli'], metrics['max_toffoli']],
                color=method_colors[method], marker=method_markers[method],
                s=65, edgecolor='black', linewidth=0.5, alpha=0.85, zorder=3,
            )

    # Qubit costs: one candle per factorization. The hatched section is the
    # extra clean-ancilla range above the dirty select-copy baseline.
    qubit_methods = ["Triangular", "THC", "Pauli"]
    qubit_color_methods = {
        "Triangular": "Triangular",
        "THC": "Bliss_THC",
        "Pauli": "Bliss_Pauli",
    }
    x_positions = np.arange(len(all_mol_names))
    bar_width = 0.22
    bar_offsets = {"Triangular": -bar_width, "THC": 0.0, "Pauli": bar_width}

    for method in qubit_methods:
        base_qubits = []
        additional_qubits = []
        for mol_name in all_mol_names:
            metrics = results[mol_name].get(method)
            if method == "Triangular":
                base = metrics.get('qubit') if metrics is not None else None
                maximum = base
            elif metrics is not None:
                base = metrics['dirty_qubit']
                maximum = metrics['max_qubit']
            else:
                base = None
                maximum = None

            base_qubits.append(base if base is not None else 0)
            additional_qubits.append(maximum - base if base is not None else 0)

        positions = x_positions + bar_offsets[method]
        color = method_colors[qubit_color_methods[method]]
        ax2.bar(
            positions, base_qubits, bar_width,
            color=color, alpha=0.8, edgecolor='black', linewidth=0.5,
        )

        if method != "Triangular":
            ax2.bar(
                positions, additional_qubits, bar_width, bottom=base_qubits,
                color=color, alpha=0.8, edgecolor='black', linewidth=0.5,
                hatch='/////',
            )

    combined_labels = [
        f"{mode_count}\n{mol_name.upper()}"
        for mode_count, mol_name in zip(all_num_modes, all_mol_names)
    ]

    ax1.set_yscale('log')
    ax1.set_xticks(all_num_modes)
    ax1.set_xticklabels(combined_labels, fontsize=19)
    ax1.set_ylabel('Toffoli cost', fontsize=20, fontweight='bold')
    ax1.set_xlabel('Number of modes', fontsize=20, fontweight='bold')
    ax1.tick_params(axis='y', labelsize=19)

    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(combined_labels, fontsize=19)
    ax2.set_yscale('log')
    ax2.set_ylabel('Qubit cost', fontsize=20, fontweight='bold')
    ax2.set_xlabel('Number of modes', fontsize=20, fontweight='bold')
    ax2.tick_params(axis='y', labelsize=19)

    legend_methods = ["Triangular"] + qroam_methods
    method_handles = []
    for method in legend_methods:
        if method == "Triangular":
            handle = mlines.Line2D(
                [], [], color='none', marker=method_markers[method],
                markerfacecolor=method_colors[method], markeredgecolor='black',
                markersize=8, label=method_labels[method],
            )
        else:
            handle = mlines.Line2D(
                [], [], color=method_colors[method], linewidth=3,
                marker=method_markers[method],
                markerfacecolor=method_colors[method], markeredgecolor='black',
                markersize=7, label=method_labels[method],
            )
        method_handles.append(handle)

    qubit_handles = [
        mpatches.Patch(
            facecolor='lightgray', edgecolor='black',
            label='Core qubits',
        ),
        mpatches.Patch(
            facecolor='lightgray', edgecolor='black', hatch='/////',
            label='Additional clean ancillas',
        ),
    ]

    method_legend = fig.legend(
        handles=method_handles, loc='lower center', bbox_to_anchor=(0.5, 0.13),
        ncol=len(method_handles), frameon=True, fontsize=21, title_fontsize=14,
        columnspacing=1.0, handletextpad=0.4,
        handler_map={
            handle: HandlerLine2D(numpoints=2)
            for handle in method_handles[1:]
        },
    )
    fig.add_artist(method_legend)
    fig.legend(
        handles=qubit_handles, loc='lower center', bbox_to_anchor=(0.5, 0.04),
        ncol=2, frameon=True, fontsize=21, title_fontsize=14,
        columnspacing=1.0, handletextpad=0.4,
    )

    #Commend out if saving is not needed
    plt.tight_layout(rect=[0, 0.22, 1, 1])
    plt.savefig("Toffoli_and_qubit_costs.pdf", bbox_inches='tight')
    plt.show()


plot_cost_metrics(
    results, all_num_modes, all_mol_names, method_markers, method_colors,
    include_bliss=plot_bliss,
)
