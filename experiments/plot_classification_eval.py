import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os

TITLE_FONT_SIZE = 24
LABEL_FONT_SIZE = 20
TICK_FONT_SIZE = 16
LEGEND_FONT_SIZE = 18
VALUE_LABEL_FONT_SIZE = 16

# Ensure Matplotlib defaults are scaled up consistently
plt.rcParams.update(
    {
        "font.size": LABEL_FONT_SIZE,
        "axes.titlesize": TITLE_FONT_SIZE,
        "axes.labelsize": LABEL_FONT_SIZE,
        "xtick.labelsize": TICK_FONT_SIZE,
        "ytick.labelsize": TICK_FONT_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE,
    }
)

# Configuration
RUN_DIR = "experiments/classification/classification_eval_Qwen3-8B_single_token"
DATA_DIR = RUN_DIR.split("/")[-1]

IMAGE_FOLDER = "images"
CLS_IMAGE_FOLDER = f"{IMAGE_FOLDER}/classification_eval"
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(CLS_IMAGE_FOLDER, exist_ok=True)

OUTPUT_PATH = f"{CLS_IMAGE_FOLDER}/classification_results_{DATA_DIR}.png"

# Filter out files containing any of these strings
FILTERED_FILENAMES = [
]

JSON_TO_LABEL = {
    # gemma 2 9b
    "checkpoints_cls_latentqa_only_addition_gemma-2-9b-it": "Classification + LatentQA",
    "checkpoints_latentqa_only_addition_gemma-2-9b-it": "LatentQA",
    "checkpoints_cls_only_addition_gemma-2-9b-it": "Classification",
    "checkpoints_latentqa_cls_past_lens_addition_gemma-2-9b-it": "Past Lens + LatentQA + Classification",

    # qwen3 8b

    "checkpoints_cls_latentqa_only_addition_Qwen3-8B": "Classification + LatentQA",
    "checkpoints_latentqa_only_addition_Qwen3-8B": "LatentQA",
    "checkpoints_cls_only_addition_Qwen3-8B": "Classification",
    "checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B": "Past Lens + LatentQA + Classification",
    "checkpoints_cls_latentqa_sae_addition_Qwen3-8B": "SAE + LatentQA + Classification",
}

# Dataset groupings
IID_DATASETS = [
    "geometry_of_truth",
    "relations",
    "sst2",
    "md_gender",
    "snli",
    "ner",
    "tense",
]

OOD_DATASETS = [
    "ag_news",
    "language_identification",
    "singular_plural",
]


def calculate_accuracy(records, dataset_ids):
    """Calculate accuracy for specified datasets."""
    correct = 0
    total = 0

    for record in records:
        if record["dataset_id"] in dataset_ids:
            total += 1
            if record["ground_truth"] == record["target"]:
                correct += 1

    if total == 0:
        return 0.0, 0
    return correct / total, total


def calculate_confidence_interval(accuracy, n, confidence=0.95):
    """Calculate binomial confidence interval for accuracy."""
    if n == 0:
        return 0.0

    # Use normal approximation for binomial confidence interval
    z_score = 1.96  # 95% confidence
    se = np.sqrt(accuracy * (1 - accuracy) / n)
    margin = z_score * se

    return margin


def load_results_from_folder(folder_path):
    """Load all JSON results from folder and calculate accuracies."""
    folder = Path(folder_path)
    results = {}

    json_files = sorted(folder.glob("*.json"))

    # Filter out files based on FILTERED_FILENAMES
    if FILTERED_FILENAMES:
        json_files = [f for f in json_files if not any(filter_str in f.name for filter_str in FILTERED_FILENAMES)]

    # Print dictionary template for easy copy-paste
    print("Found JSON files:")
    file_dict = {f.name: "" for f in json_files}
    print(file_dict)
    print()

    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)

        # Use label from mapping, or filename if not mapped
        label = JSON_TO_LABEL.get(json_file.name, json_file.name)

        records = data.get("records", [])

        # Calculate accuracies and counts
        iid_acc, iid_count = calculate_accuracy(records, IID_DATASETS)
        ood_acc, ood_count = calculate_accuracy(records, OOD_DATASETS)

        # Calculate confidence intervals
        iid_ci = calculate_confidence_interval(iid_acc, iid_count)
        ood_ci = calculate_confidence_interval(ood_acc, ood_count)

        results[label] = {
            "iid_accuracy": iid_acc,
            "ood_accuracy": ood_acc,
            "iid_ci": iid_ci,
            "ood_ci": ood_ci,
            "iid_count": iid_count,
            "ood_count": ood_count,
        }

        print(f"{label}:")
        print(f"  IID Accuracy: {iid_acc:.2%} ± {iid_ci:.2%} (n={iid_count})")
        print(f"  OOD Accuracy: {ood_acc:.2%} ± {ood_ci:.2%} (n={ood_count})")

    return results


def plot_accuracies(results, output_path=None):
    """Create bar plots for IID and OOD accuracies."""
    labels = list(results.keys())
    iid_accs = [results[name]["iid_accuracy"] for name in labels]
    ood_accs = [results[name]["ood_accuracy"] for name in labels]
    iid_cis = [results[name]["iid_ci"] for name in labels]
    ood_cis = [results[name]["ood_ci"] for name in labels]

    # Generate distinct colors for each bar
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # IID plot
    x_pos = np.arange(len(labels))
    bars1 = ax1.bar(x_pos, iid_accs, color=colors, alpha=0.8, yerr=iid_cis, capsize=5)
    ax1.set_ylabel("Accuracy", fontsize=LABEL_FONT_SIZE, fontweight="bold")
    ax1.set_title("IID Dataset Accuracy", fontsize=TITLE_FONT_SIZE, fontweight="bold")
    ax1.set_xticks([])
    # ax1.set_ylim([0, 1])
    ax1.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    ax1.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1%}",
            ha="center",
            va="bottom",
            fontsize=VALUE_LABEL_FONT_SIZE,
        )

    # OOD plot
    bars2 = ax2.bar(x_pos, ood_accs, color=colors, alpha=0.8, yerr=ood_cis, capsize=5)
    ax2.set_ylabel("Accuracy", fontsize=LABEL_FONT_SIZE, fontweight="bold")
    ax2.set_title("OOD Dataset Accuracy", fontsize=TITLE_FONT_SIZE, fontweight="bold")
    ax2.set_xticks([])
    # ax2.set_ylim([0, 1])
    ax2.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    ax2.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1%}",
            ha="center",
            va="bottom",
            fontsize=VALUE_LABEL_FONT_SIZE,
        )

    # Shared legend centered beneath both plots
    legend_labels = [f"{label}" for label in labels]
    legend_columns = min(len(legend_labels), 1) or 1
    fig.legend(
        bars1,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=legend_columns,
        frameon=True,
        fontsize=LEGEND_FONT_SIZE,
    )

    plt.tight_layout(pad=2.0, rect=(0, 0.12, 1, 1))

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"\nPlot saved to: {output_path}")

    plt.show()


def main():
    print(f"Loading results from: {RUN_DIR}\n")
    results = load_results_from_folder(RUN_DIR)

    if not results:
        print("No JSON files found in the specified folder!")
        return

    print(f"\nGenerating plots...")
    plot_accuracies(results, OUTPUT_PATH)


if __name__ == "__main__":
    main()
