import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import re

def strip_markdown_syntax(code: str) -> str:
    """
    Extract Python code from markdown-like LLM responses.

    Behavior:
    - If one or more fenced Python blocks exist (```python, ```py, ```python3, ...),
      return only their contents (concatenated with blank lines).
    - Otherwise, preserve backwards compatibility by stripping a single outer fence
      if present, or returning the input stripped.
    
    Args:
        code: The text potentially containing markdown and explanations
        
    Returns:
        Clean Python code content
    """
    if not code:
        return ""

    text = code.strip()
    fence_pattern = re.compile(r"```\s*([^\n`]*)\n(.*?)```", re.DOTALL)

    python_blocks = []
    for match in fence_pattern.finditer(text):
        language = match.group(1).strip().lower()
        block_content = match.group(2).strip()

        if language in {"python", "py"} or language.startswith("python"):
            python_blocks.append(block_content)

    if python_blocks:
        return "\n\n".join(python_blocks).strip()

    # Backwards-compatible fallback: strip one outer fence if content is a single
    # fenced block without specific language filtering.
    if text.startswith("```") and text.endswith("```"):
        lines = text.split('\n')
        if len(lines) >= 2:
            return '\n'.join(lines[1:-1]).strip()

    return text

def save_text_to_file(text, file_path, verbose=True):
    """
    Saves the given text content to a file at the specified path.
    Creates directories if they don't exist.
    """
    text = strip_markdown_syntax(text) # Strip markdown syntax before saving

    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    with open(file_path, "w") as f:
        f.write(text)
    
    if verbose:
        tqdm.write(f"Saved content to {file_path}")

def parse_time_metrics(output):
    """Parses output from /usr/bin/time -v"""
    metrics = {}
    try:
        for line in output.splitlines():
            line = line.strip()
            if "User time (seconds):" in line:
                metrics["user_time"] = float(line.split(":")[-1].strip())
            elif "System time (seconds):" in line:
                metrics["sys_time"] = float(line.split(":")[-1].strip())
            elif "Maximum resident set size (kbytes):" in line:
                metrics["max_rss_kb"] = int(line.split(":")[-1].strip())
            elif "Percent of CPU this job got:" in line:
                metrics["cpu_percent"] = line.split(":")[-1].strip()
    except Exception as e:
        # Using print/tqdm.write might be noisy if this function is called frequently
        pass 
    return metrics

def clear_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def generate_summary_plot(stats_summary, output_dir):
    """
    Generates summary plots comparing performance metrics across models.
    Creates separate plot files for Validity, Time, and Cost in the output directory.
    """
    if not stats_summary:
        return

    os.makedirs(output_dir, exist_ok=True)

    models = [s['model'] for s in stats_summary]
    short_models = [m.split('/')[-1] for m in models]
    
    valid_percentages = [(s['valid_programs'] / s['total_programs']) * 100 for s in stats_summary]
    
    # Handle potentially 0 valid programs to avoid division by zero
    avg_times = []
    for s in stats_summary:
        if s['valid_programs'] > 0:
            avg_times.append(s['total_time'] / s['valid_programs'])
        else:
            avg_times.append(0)

    costs_per_valid = []
    for s in stats_summary:
        if s['valid_programs'] > 0:
            costs_per_valid.append(s['total_cost'] / s['valid_programs'])
        else:
            # If no valid programs, the cost per valid program is theoretically infinite.
            # We show total sunk cost here as a proxy, or could be 0.
            costs_per_valid.append(s['total_cost']) 

    avg_scores = [s.get('avg_quality_score', 0.0) for s in stats_summary]

    x = np.arange(len(models)) 

    # --- Plot 1: Validity ---
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(x, valid_percentages, color='skyblue')
    ax1.set_title('Validity (%)')
    ax1.set_ylabel('Percentage')
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_models, rotation=45, ha='right')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    fig1.tight_layout()
    plot_path1 = os.path.join(output_dir, "performance_validity.png")
    plt.savefig(plot_path1, bbox_inches='tight')
    tqdm.write(f"Validity plot saved to {plot_path1}")
    plt.close(fig1)

    # --- Plot 2: Time ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.bar(x, avg_times, color='lightgreen')
    ax2.set_title('Avg Time per Valid Program (s)')
    ax2.set_ylabel('Seconds')
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_models, rotation=45, ha='right')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    fig2.tight_layout()
    plot_path2 = os.path.join(output_dir, "performance_time.png")
    plt.savefig(plot_path2, bbox_inches='tight')
    tqdm.write(f"Time plot saved to {plot_path2}")
    plt.close(fig2)

    # --- Plot 3: Cost ---
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.bar(x, costs_per_valid, color='salmon')
    ax3.set_title('Cost per Valid Program ($)')
    ax3.set_ylabel('Dollars')
    ax3.set_xticks(x)
    ax3.set_xticklabels(short_models, rotation=45, ha='right')
    ax3.grid(axis='y', linestyle='--', alpha=0.7)

    fig3.tight_layout()
    plot_path3 = os.path.join(output_dir, "performance_cost.png")
    plt.savefig(plot_path3, bbox_inches='tight')
    tqdm.write(f"Cost plot saved to {plot_path3}")
    plt.close(fig3)

    # --- Plot 4: Quality Score ---
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    ax4.bar(x, avg_scores, color='violet')
    ax4.set_title('Avg Quality Score')
    ax4.set_ylabel('Score')
    ax4.set_xticks(x)
    ax4.set_xticklabels(short_models, rotation=45, ha='right')
    ax4.grid(axis='y', linestyle='--', alpha=0.7)

    fig4.tight_layout()
    plot_path4 = os.path.join(output_dir, "performance_quality.png")
    plt.savefig(plot_path4, bbox_inches='tight')
    tqdm.write(f"Quality Score plot saved to {plot_path4}")
    plt.close(fig4)

def generate_coverage_text_report(grouped_results, output_file):
    """Generates a structured text report of coverage results."""
    with open(output_file, 'w') as f:
        f.write("Coverage Analysis Report\n")
        f.write("========================\n\n")
        
        for group_name in sorted(grouped_results.keys()):
            files = grouped_results[group_name]
            f.write(f"=== Group: {group_name} ===\n")
            
            successful_coverages = []
            files_sorted = sorted(files, key=lambda x: x['file'])
            
            # List individual files
            for entry in files_sorted:
                fname = os.path.basename(entry['file'])
                if entry['success']:
                    f.write(f"  {fname}: {entry['coverage_percent']:.2f}%\n")
                    if entry.get("verbose_report"):
                        f.write("\n    Detailed Coverage:\n")
                        for line in entry["verbose_report"].splitlines():
                            f.write(f"    {line}\n")
                        f.write("\n")
                    successful_coverages.append(entry['coverage_percent'])
                else:
                    err_msg = entry['error'].replace('\n', ' ')[:100]
                    f.write(f"  {fname}: ERROR ({err_msg}...)\n")
            
            # Summary for the group
            if successful_coverages:
                avg = sum(successful_coverages) / len(successful_coverages)
                f.write(f"\n  Summary for {group_name}:\n")
                f.write(f"    Average Coverage: {avg:.2f}%\n")
                f.write(f"    Valid Programs: {len(successful_coverages)}/{len(files)}\n")
            else:
                f.write(f"\n  Summary for {group_name}:\n")
                f.write(f"    Average Coverage: N/A\n")
                f.write(f"    Valid Programs: 0/{len(files)}\n")
            
            f.write("\n" + "-"*40 + "\n\n")

def generate_complexity_scatter_plots(all_metrics, output_dir):
    """
    Generates only the requested complexity plots:
    - Coverage (%) vs Line Count (2D), colored by function count (blue=low, red=high)
    - Coverage (%) vs Execution Time (s) vs Compilation Time (s) (3D)

    all_metrics: list of dicts where each entry must include a 'metrics' key.
    The 'model' key is optional for this function and is currently ignored.
    """
    if not all_metrics:
        return

    os.makedirs(output_dir, exist_ok=True)

    records = []
    for entry in all_metrics:
        metrics_root = entry.get('metrics', {})
        execution_metrics = metrics_root.get('execution', {})
        compilation_metrics = metrics_root.get('compilation', {})

        # Legacy fallback where execution metrics may be at root
        if not execution_metrics and 'wall_time' in metrics_root and 'compilation' not in metrics_root:
            execution_metrics = metrics_root

        line_count = execution_metrics.get('line_count')
        function_count = execution_metrics.get('function_count')
        coverage_percent = execution_metrics.get('coverage_percent')
        execution_time = execution_metrics.get('wall_time')
        compilation_time = compilation_metrics.get('wall_time')

        if line_count is None or function_count is None or coverage_percent is None:
            continue

        records.append({
            'line_count': line_count,
            'function_count': function_count,
            'coverage_percent': coverage_percent,
            'execution_time': execution_time,
            'compilation_time': compilation_time,
        })

    if not records:
        return

    line_counts = [r['line_count'] for r in records]
    function_counts = [r['function_count'] for r in records]
    coverage_vals = [r['coverage_percent'] for r in records]

    # Plot 1: Coverage vs line count (2D) with function-count gradient (blue->red)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    scatter1 = ax1.scatter(
        line_counts,
        coverage_vals,
        c=function_counts,
        cmap='coolwarm',
        alpha=0.75,
        edgecolors='w',
        s=60,
    )
    cbar1 = fig1.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Number of Distinct Functions (blue=low, red=high)')
    ax1.set_xlabel('Line Count')
    ax1.set_ylabel('Coverage (%)')
    ax1.set_title('Coverage vs Line Count')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    fig1.tight_layout()
    plot_path1 = os.path.join(output_dir, 'complexity_coverage_vs_line_count.png')
    plt.savefig(plot_path1, bbox_inches='tight')
    tqdm.write(f"Saved plot: {plot_path1}")
    plt.close(fig1)

    # Plot 2-4: Three 2D perspectives for execution/compilation/coverage
    points_3d = [
        r for r in records
        if r['execution_time'] is not None and r['compilation_time'] is not None
    ]

    if points_3d:
        x_exec = [r['execution_time'] for r in points_3d]
        y_comp = [r['compilation_time'] for r in points_3d]
        z_cov = [r['coverage_percent'] for r in points_3d]
        fn_colors = [r['function_count'] for r in points_3d]

        # Perspective A: Coverage vs Execution Time
        fig2a, ax2a = plt.subplots(figsize=(10, 6))
        scatter2a = ax2a.scatter(
            x_exec,
            z_cov,
            c=fn_colors,
            cmap='coolwarm',
            alpha=0.8,
            edgecolors='w',
            s=60,
        )
        cbar2a = fig2a.colorbar(scatter2a, ax=ax2a)
        cbar2a.set_label('Number of Distinct Functions (blue=low, red=high)')
        ax2a.set_xlabel('Execution Time (s)')
        ax2a.set_ylabel('Coverage (%)')
        ax2a.set_title('Coverage vs Execution Time')
        ax2a.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        fig2a.tight_layout()
        plot_path2a = os.path.join(output_dir, 'complexity_coverage_vs_execution_time.png')
        plt.savefig(plot_path2a, bbox_inches='tight')
        tqdm.write(f"Saved plot: {plot_path2a}")
        plt.close(fig2a)

        # Perspective B: Coverage vs Compilation Time
        fig2b, ax2b = plt.subplots(figsize=(10, 6))
        scatter2b = ax2b.scatter(
            y_comp,
            z_cov,
            c=fn_colors,
            cmap='coolwarm',
            alpha=0.8,
            edgecolors='w',
            s=60,
        )
        cbar2b = fig2b.colorbar(scatter2b, ax=ax2b)
        cbar2b.set_label('Number of Distinct Functions (blue=low, red=high)')
        ax2b.set_xlabel('Compilation Time (s)')
        ax2b.set_ylabel('Coverage (%)')
        ax2b.set_title('Coverage vs Compilation Time')
        ax2b.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        fig2b.tight_layout()
        plot_path2b = os.path.join(output_dir, 'complexity_coverage_vs_compilation_time.png')
        plt.savefig(plot_path2b, bbox_inches='tight')
        tqdm.write(f"Saved plot: {plot_path2b}")
        plt.close(fig2b)

        # Perspective C: Execution Time vs Compilation Time
        fig2c, ax2c = plt.subplots(figsize=(10, 6))
        scatter2c = ax2c.scatter(
            x_exec,
            y_comp,
            c=fn_colors,
            cmap='coolwarm',
            alpha=0.8,
            edgecolors='w',
            s=60,
        )
        cbar2c = fig2c.colorbar(scatter2c, ax=ax2c)
        cbar2c.set_label('Number of Distinct Functions (blue=low, red=high)')
        ax2c.set_xlabel('Execution Time (s)')
        ax2c.set_ylabel('Compilation Time (s)')
        ax2c.set_title('Execution Time vs Compilation Time')
        ax2c.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        fig2c.tight_layout()
        plot_path2c = os.path.join(output_dir, 'complexity_execution_vs_compilation_time.png')
        plt.savefig(plot_path2c, bbox_inches='tight')
        tqdm.write(f"Saved plot: {plot_path2c}")
        plt.close(fig2c)