import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
cmap = plt.get_cmap("nipy_spectral")

all_mol_names = ["pah1", "pah2", "pah3", "pah4", "pah5", "pah6", "pah7"]
all_num_modes = [30, 48, 66, 84, 102, 120, 138]

method_markers = {
    "Triangular": "^",
    "Pauli": "o",
    "Bliss_Pauli": "s",
    "THC": "o",
    "Bliss_THC": "s",
}

method_colors = {
    "Triangular": cmap(0.2),
    "Pauli": cmap(0.44),
    "Bliss_Pauli": cmap(0.44),
    "THC": cmap(0.75),
    "Bliss_THC": cmap(0.75),
}

results_directory = Path(__file__).resolve().parent.parent
number_pattern = r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"
one_norm_regex = re.compile(
    rf"(?:1-norm|one norm):\s*({number_pattern})", re.IGNORECASE
)


def extract_one_norm(filepath):
    """Extract the LCU 1-norm from one resource-estimate file."""
    filepath = Path(filepath)
    if not filepath.is_file():
        raise FileNotFoundError(f"Required result file not found: {filepath}")

    with filepath.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            match = one_norm_regex.search(line)
            if match:
                return float(match.group(1))

    raise ValueError(f"No LCU 1-norm found in {filepath}")


def load_results():
    """Load only the 1-norm values required by the plot."""
    results = {}

    for mol_name in all_mol_names:
        basis_directory = Path(mol_name) / "VSCF_basis"
        results[mol_name] = {
            "Triangular": {
                "norm": extract_one_norm(
                    results_directory
                    / "Triangular_params"
                    / basis_directory
                    / "cost_5M_2MC_eps_1.txt"
                )
            },
            "Pauli": {
                "norm": extract_one_norm(
                    results_directory
                    / "Pauli_params"
                    / basis_directory
                    / "cost_sel_copy_dirty_5M_2MC_eps_1.txt"
                )
            },
            "Bliss_Pauli": {
                "norm": extract_one_norm(
                    results_directory
                    / "Bliss_Pauli_params"
                    / basis_directory
                    / "cost_sel_copy_dirty_5M_2MC_eps_1.txt"
                )
            },
            "THC": {
                "norm": extract_one_norm(
                    results_directory
                    / "THC_params"
                    / basis_directory
                    / "cost_sel_copy_dirty_5M_hR_2MC_eps_1.txt"
                )
            },
            "Bliss_THC": {
                "norm": extract_one_norm(
                    results_directory
                    / "Bliss_THC_params"
                    / basis_directory
                    / "cost_sel_copy_dirty_5M_hR_2MC_eps_1.txt"
                )
            },
        }

    return results


def plot_one_norm(results, all_num_modes, all_mol_names, method_markers, method_colors):
    """Plot LCU 1-norms and Triangular-to-method 1-norm ratios."""
    target_methods = ["Triangular", "Pauli", "Bliss_Pauli", "THC", "Bliss_THC"]
    target_method_labels = [
        "Triangular",
        "Pauli",
        "BLISS-Pauli",
        "THC",
        "BLISS-THC",
    ]

    fig, ax = plt.subplots(figsize=(10, 7))
    norm_data_per_method = {
        method: {"x": [], "y": []} for method in target_methods
    }
    norm_ratios = {
        method: {"x": [], "y": []}
        for method in target_methods
        if method != "Triangular"
    }

    for x_idx, mol_name in enumerate(all_mol_names):
        mode_count = all_num_modes[x_idx]
        for method in target_methods:
            norm = results[mol_name][method]["norm"]
            norm_data_per_method[method]["x"].append(mode_count)
            norm_data_per_method[method]["y"].append(norm)

        triangular_norm = results[mol_name]["Triangular"]["norm"]
        for method in target_methods[1:]:
            norm_ratios[method]["x"].append(mode_count)
            norm_ratios[method]["y"].append(
                triangular_norm / results[mol_name][method]["norm"]
            )

    for method_idx, method in enumerate(target_methods):
        data = norm_data_per_method[method]
        x_values = np.asarray(data["x"])
        y_values = np.asarray(data["y"])
        slope, intercept = np.polyfit(x_values, y_values, 1)
        x_fit = np.linspace(x_values.min(), x_values.max(), 100)
        y_fit = slope * x_fit + intercept
        label_with_slope = (
            f"{target_method_labels[method_idx]}: {slope / 1e4:.2f}"
        )

        ax.plot(
            x_fit,
            y_fit,
            color=method_colors.get(method, "gray"),
            linestyle="-",
            alpha=0.6,
            zorder=1,
        )
        ax.scatter(
            x_values,
            y_values,
            color=method_colors.get(method, "gray"),
            marker=method_markers.get(method, "o"),
            s=90,
            alpha=0.9,
            edgecolor="black",
            linewidth=0.8,
            zorder=2,
            label=label_with_slope,
        )

    axins = ax.inset_axes([0.075, 0.60, 0.33, 0.35])
    for method in target_methods[1:]:
        axins.plot(
            norm_ratios[method]["x"],
            norm_ratios[method]["y"],
            color=method_colors.get(method, "gray"),
            marker=method_markers.get(method, "o"),
            markersize=8,
            linestyle="-",
            alpha=0.9,
            markeredgecolor="black",
            linewidth=1.2,
            zorder=2,
        )

    axins.set_ylabel("Ratios", fontsize=15, fontweight="bold")
    axins.set_xticks(all_num_modes)
    axins.set_xticklabels([])
    axins.tick_params(axis="y", which="major", labelsize=14)

    combined_labels = [
        f"{mode_count}\n{mol_name.upper()}"
        for mode_count, mol_name in zip(all_num_modes, all_mol_names)
    ]
    ax.set_xticks(all_num_modes)
    ax.set_xticklabels(combined_labels, rotation=0, ha="center", fontsize=18)
    ax.set_ylabel("1-norm (cm$^{\\mathbf{-1}}$)", fontsize=18, fontweight="bold")
    ax.set_xlabel("Number of modes", fontsize=18, fontweight="bold")
    ax.tick_params(axis="y", labelsize=18)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=len(target_methods),
        fontsize=16,
        frameon=True,
        title="Fit Slopes (1e4 cm$^{-1}$)",
        title_fontsize=15,
    )

    output_path = Path(__file__).resolve().parent / "1-norms.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plot_one_norm(
        load_results(),
        all_num_modes,
        all_mol_names,
        method_markers,
        method_colors,
    )
