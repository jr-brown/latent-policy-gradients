#!/usr/bin/env python3
"""Extract model metrics from log files and output as LaTeX table.

Usage:
    python scripts/extract_model_metrics.py local/outputs/tmux/ns_lbr_rkl_log.txt local/outputs/tmux/dqns_lbr_rkl_log.txt
"""

import argparse
import re
import sys
from pathlib import Path


def extract_metrics(log_path: Path) -> dict:
    """Extract key metrics from a model fitting log file."""
    content = log_path.read_text()

    metrics = {
        'file': log_path.name,
        'model_loss': None,
        'kfold_mean': None,
        'kfold_se': None,
        'single_to_multi_val': None,
        'no_dist_to_dist_val': None,
    }

    # 1. Model loss after full training
    # Look for: │ native_saliency_multi_kl model │ 0.06315 │
    # or similar pattern in Model Performance Summary
    model_loss_match = re.search(
        r'Model Performance Summary.*?│[^│]+model\s*│\s*([\d.]+)\s*│',
        content, re.DOTALL
    )
    if model_loss_match:
        metrics['model_loss'] = float(model_loss_match.group(1))

    # 2. K-fold validation loss with standard error
    # Look for: Mean validation loss: 0.064435 ± 0.001929
    kfold_match = re.search(
        r'Mean validation loss:\s*([\d.]+)\s*±\s*([\d.]+)',
        content
    )
    if kfold_match:
        metrics['kfold_mean'] = float(kfold_match.group(1))
        metrics['kfold_se'] = float(kfold_match.group(2))

    # 3. Single-stage to multi-stage validation loss
    # Look for: │ Validation loss (multi-stage) │      0.093094 │
    single_to_multi_match = re.search(
        r'Validation loss \(multi-stage\)\s*│\s*([\d.]+)',
        content
    )
    if single_to_multi_match:
        metrics['single_to_multi_val'] = float(single_to_multi_match.group(1))

    # 4. No-distractor to distractor validation loss
    # Look for: │ Validation loss (with distractors) │      0.075042 │
    no_dist_match = re.search(
        r'Validation loss \(with distractors\)\s*│\s*([\d.]+)',
        content
    )
    if no_dist_match:
        metrics['no_dist_to_dist_val'] = float(no_dist_match.group(1))

    return metrics


def format_latex_table(all_metrics: list[dict]) -> str:
    """Format metrics as a LaTeX table."""

    # Extract model name from filename
    def get_model_name(filename: str) -> str:
        name = filename.replace('_log.txt', '')
        # Make it more readable
        name_map = {
            'ns_lbr_rkl': 'Native Saliency',
            'ns_lbr_rkl_diagonal': 'Native Saliency (diagonal)',
            'ns_lbr_rkl_memoryless': 'Native Saliency (memoryless)',
            'ns_lbr_rkl_simultaneous': 'Native Saliency (simultaneous)',
            'dqns_lbr_rkl': 'Diag. Quad. Native Saliency',
            'ns_cs256_lbr_rkl': 'Native Saliency Contextual Sharded (256)',
            'ns_cs256_lbr_rkl_simultaneous': 'NS Contextual Sharded (256, simul.)',
            'ns_cs32_lbr_rkl_new': 'Native Saliency Contextual Sharded (32)',
        }
        return name_map.get(name, name)

    # Find best (minimum) values for each column
    valid_train = [m['model_loss'] for m in all_metrics if m['model_loss'] is not None]
    valid_kfold = [m['kfold_mean'] for m in all_metrics if m['kfold_mean'] is not None]
    valid_s2m = [m['single_to_multi_val'] for m in all_metrics if m['single_to_multi_val'] is not None]
    valid_d2d = [m['no_dist_to_dist_val'] for m in all_metrics if m['no_dist_to_dist_val'] is not None]

    best_train = min(valid_train) if valid_train else None
    best_kfold = min(valid_kfold) if valid_kfold else None
    best_s2m = min(valid_s2m) if valid_s2m else None
    best_d2d = min(valid_d2d) if valid_d2d else None

    def format_val(val, best, fmt=".4f"):
        """Format a value, bolding if it's the best."""
        if val is None:
            return '---'
        formatted = f"{val:{fmt}}"
        if best is not None and val == best:
            return f"\\textbf{{{formatted}}}"
        return formatted

    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{Model comparison across validation schemes}',
        r'\label{tab:model_comparison}',
        r'\begin{tabular}{lcccc}',
        r'\toprule',
        r'Model & Training Loss & K-Fold Val. & Single$\to$Multi & No Dist.$\to$Dist. \\',
        r'\midrule',
    ]

    for m in all_metrics:
        model_name = get_model_name(m['file'])

        # Format values with bolding for best
        train_loss = format_val(m['model_loss'], best_train)

        if m['kfold_mean'] is not None and m['kfold_se'] is not None:
            kfold_formatted = format_val(m['kfold_mean'], best_kfold)
            kfold = f"{kfold_formatted} $\\pm$ {m['kfold_se']:.4f}"
        else:
            kfold = '---'

        s2m = format_val(m['single_to_multi_val'], best_s2m)
        d2d = format_val(m['no_dist_to_dist_val'], best_d2d)

        lines.append(f'{model_name} & {train_loss} & {kfold} & {s2m} & {d2d} \\\\')

    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ])

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Extract model metrics from log files and output as LaTeX table.'
    )
    parser.add_argument(
        'log_files',
        nargs='+',
        type=Path,
        help='Log file(s) to process'
    )
    args = parser.parse_args()

    log_files = args.log_files

    # Validate files exist
    for f in log_files:
        if not f.exists():
            print(f"Error: File not found: {f}", file=sys.stderr)
            sys.exit(1)

    print("=" * 60)
    print("Extracted Metrics")
    print("=" * 60)

    all_metrics = []
    for log_file in log_files:
        metrics = extract_metrics(log_file)
        all_metrics.append(metrics)

        print(f"\n{log_file.name}:")
        print(f"  Training loss:           {metrics['model_loss']}")
        print(f"  K-fold validation:       {metrics['kfold_mean']} ± {metrics['kfold_se']}")
        print(f"  Single→Multi validation: {metrics['single_to_multi_val']}")
        print(f"  NoDist→Dist validation:  {metrics['no_dist_to_dist_val']}")

    print("\n" + "=" * 60)
    print("LaTeX Table")
    print("=" * 60 + "\n")

    print(format_latex_table(all_metrics))


if __name__ == '__main__':
    main()
