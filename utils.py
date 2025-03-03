"""
Utility functions for text preprocessing, retry handling, and evaluation visualization.
"""

import re
import os
import nltk
from nltk.corpus import words
from typing import Callable
import random
import time
import os
import pandas as pd
import matplotlib.pyplot as plt

def preprocess_text(text: str) -> str:
    """
    Preprocesses a string for better retrieval by filtering to English words only.

    This function:
      1. Removes non-alphabetic characters.
      2. Lowercases and splits on whitespace.
      3. Filters out tokens that are not in the NLTK English vocabulary.

    Args:
        text: Raw string to preprocess.

    Returns:
        A cleaned, space-joined string of valid English tokens.
    """
    english_vocab = set(w.lower() for w in words.words())
    cleaned = re.sub(r"[^A-Za-z\s]", "", text)  # Remove all non-alphabetic chars
    tokens = cleaned.lower().split()
    valid_tokens = [word for word in tokens if word in english_vocab]
    return " ".join(valid_tokens)


def retry_with_backoff(func: Callable, retries: int = 5, base_delay: float = 2.0):
    """
    Retries a function call with exponential backoff in case of failures.

    Args:
        func: The function to attempt.
        retries: Number of total retry attempts.
        base_delay: Starting delay in seconds before the first retry.

    Returns:
        The return value of the function call, if successful.

    Raises:
        RuntimeError: If all retries fail.
    """
    for attempt in range(retries):
        try:
            return func()
        except Exception as exc:
            wait_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"Retry {attempt + 1}/{retries} failed with error: {exc}. "
                  f"Retrying in {wait_time:.2f} seconds...")
            time.sleep(wait_time)
    raise RuntimeError("Max retries exceeded for the given function call.")


def visualize_evaluation_results(csv_path: str, output_dir: str) -> None:
    """
    Reads an evaluation CSV file, creates a bar chart for mean metric scores,
    and saves the resulting chart (with each bar uniquely colored and labeled) 
    to the specified directory.
    
    Args:
        csv_path: Path to the evaluation CSV file.
        output_dir: The directory where the chart will be saved.
    """
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Potential columns from Ragas or your custom metrics
    metrics = [
        "context_recall",
        "answer_relevancy",
        "llm_context_precision_with_reference",
        "faithfulness",
        "factual_correctness"
    ]

    # Keep only columns that exist in the DataFrame
    metrics = [m for m in metrics if m in df.columns]
    if not metrics:
        print("No recognized metric columns found in the given CSV.")
        return

    # Calculate mean values for each metric
    mean_values = df[metrics].mean()

    # Prepare for plotting
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "evaluation_metrics.png")

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 5))

    # Generate a distinct color for each bar using a colormap
    cmap = plt.cm.get_cmap("tab10", len(mean_values))
    colors = [cmap(i) for i in range(len(mean_values))]

    # Plot bars
    x_positions = range(len(mean_values))
    bars = ax.bar(x_positions, mean_values, color=colors)

    # Label each bar with its numeric value
    ax.bar_label(bars, fmt="%.2f", padding=3)

    # Configure axis
    ax.set_title("Evaluation Metrics (Average Scores)", fontsize=14, pad=10)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(mean_values.index, rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close(fig)

    print(f"Evaluation metrics bar chart saved at {plot_path}")
