import pickle
from pathlib import Path

import matplotlib.pyplot as plt


PLOT_DIRECTORY = Path(__file__).resolve().parent
RESULTS_DIRECTORY = PLOT_DIRECTORY.parent


def load_pickle(relative_path):
    """Load one spectral-analysis result from the Results directory."""
    filepath = RESULTS_DIRECTORY / relative_path
    if not filepath.is_file():
        raise FileNotFoundError(f"Required spectral-analysis file not found: {filepath}")

    with filepath.open("rb") as input_file:
        results = pickle.load(input_file)

    required_keys = {"tensor errors", "energy errors"}
    missing_keys = required_keys.difference(results)
    if missing_keys:
        raise KeyError(f"{filepath} is missing keys: {sorted(missing_keys)}")

    return results


def load_spectral_results():
    """Load the Pauli, THC, and Triangular results for CH4 and C2H4."""
    ch4_pauli = load_pickle(
        Path("Pauli_params/CH4/VSCF_basis/spectral_analysis_5M_2MC.pkl")
    )
    ch4_thc = load_pickle(
        Path(
            "THC_params/CH4/VSCF_basis/"
            "spectral_analysis_5M_6.67R_2MC_eps_0.644901.pkl"
        )
    )
    ch4_triangular = load_pickle(
        Path("Triangular_params/CH4/VSCF_basis/spectral_analysis_5M_2MC.pkl")
    )

    c2h4_pauli = load_pickle(
        Path("Pauli_params/C2H4_new/VSCF_basis/spectral_analysis_5M_2MC.pkl")
    )
    c2h4_thc = load_pickle(
        Path(
            "THC_params/C2H4_new/VSCF_basis/"
            "spectral_analysis_5M_6.75R_2MC_eps_0.522931.pkl"
        )
    )
    c2h4_triangular = load_pickle(
        Path("Triangular_params/C2H4_new/VSCF_basis/spectral_analysis_5M_2MC.pkl")
    )

    return (
        ch4_triangular,
        ch4_pauli,
        ch4_thc,
        c2h4_triangular,
        c2h4_pauli,
        c2h4_thc,
    )


def plot_error_metrics(ch4_tri, ch4_pauli, ch4_thc,
                       c2h4_tri, c2h4_pauli, c2h4_thc):
    """Plot tensor errors against energy errors for three LCU methods."""
    plt.rcParams.update({
        "font.size": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 23,
        "ytick.labelsize": 23,
    })

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(18, 10),
        layout="constrained",
    )

    data_grid = [
        [ch4_tri, ch4_pauli, ch4_thc],
        [c2h4_tri, c2h4_pauli, c2h4_thc],
    ]
    column_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    column_labels = ["Triangular", "Pauli", "THC"]

    for row_index in range(2):
        for column_index in range(3):
            axis = axes[row_index, column_index]
            data = data_grid[row_index][column_index]

            filtered_points = [
                (tensor_error, energy_error)
                for tensor_error, energy_error in zip(
                    data["tensor errors"], data["energy errors"]
                )
                if 0.8 < tensor_error < 1e3 and energy_error > 0
            ]
            filtered_x = [point[0] for point in filtered_points]
            filtered_y = [point[1] for point in filtered_points]
            label = column_labels[column_index] if row_index == 0 else None

            axis.scatter(
                filtered_x,
                filtered_y,
                color=column_colors[column_index],
                s=120,
                alpha=0.75,
                edgecolor="black",
                linewidth=1,
                label=label,
            )
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_xlim(right=1e3)

            if column_index == 0:
                row_label = "CH$_4$" if row_index == 0 else "C$_2$H$_4$"
                axis.yaxis.set_label_position("left")
                axis.set_ylabel(
                    row_label,
                    rotation=90,
                    labelpad=25,
                    fontweight="bold",
                    fontsize=19,
                )

    fig.supxlabel("Tensor error (cm$^{\\mathbf{-1}}$)", fontweight="bold")
    fig.supylabel("Energy error (cm$^{\\mathbf{-1}}$)", fontweight="bold")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0), ncol=3)

    output_path = PLOT_DIRECTORY / "energy_vs_tensor_errors.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plot_error_metrics(*load_spectral_results())
