import json
import os
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

# Configuration
OUTPUT_JSON_DIR = "taboo_eval_results"


def load_results(json_dir):
    """Load all JSON files from the directory."""
    results_by_lora = defaultdict(list)

    json_dir = Path(json_dir)
    if not json_dir.exists():
        print(f"Directory {json_dir} does not exist!")
        return results_by_lora

    json_files = list(json_dir.glob("*.json"))
    print(f"Found {len(json_files)} JSON files")

    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)

        investigator_lora = data["meta"]["investigator_lora_path"]

        # Calculate accuracy for each record
        for record in data["records"]:
            # Each record has 10 full sequence responses
            num_correct = record["fullseq_yes_count"]
            total = len(record["full_sequence_responses"])
            accuracy = num_correct / total

            results_by_lora[investigator_lora].append(accuracy)

    return results_by_lora


def plot_results(results_by_lora):
    """Create a bar chart of average accuracy by investigator LoRA."""
    if not results_by_lora:
        print("No results to plot!")
        return

    # Calculate mean accuracy for each LoRA
    lora_names = []
    mean_accuracies = []

    for lora_path, accuracies in results_by_lora.items():
        # Extract a readable name from the path
        lora_name = lora_path.split("/")[-1]
        lora_names.append(lora_name)
        mean_acc = sum(accuracies) / len(accuracies)
        mean_accuracies.append(mean_acc)

        print(f"{lora_name}: {mean_acc:.3f} (n={len(accuracies)} records)")

    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(lora_names)), mean_accuracies, color="steelblue")

    ax.set_xlabel("Investigator LoRA", fontsize=12)
    ax.set_ylabel("Average Accuracy", fontsize=12)
    ax.set_title("Taboo Probe: Average Correct Answer Rate by Investigator", fontsize=14)
    ax.set_xticks(range(len(lora_names)))
    ax.set_xticklabels(lora_names, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, mean_accuracies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{acc:.3f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig("taboo_probe_results.png", dpi=300, bbox_inches="tight")
    print("\nPlot saved as 'taboo_probe_results.png'")
    plt.show()


def main():
    # Load results from all JSON files
    results_by_lora = load_results(OUTPUT_JSON_DIR)

    # Plot the results
    plot_results(results_by_lora)


if __name__ == "__main__":
    main()
