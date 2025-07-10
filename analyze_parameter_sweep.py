#!/usr/bin/env python3
"""
Parameter Sweep Analysis Script for EDG Experiments

This script analyzes parameter sweep results from EDG experiments to find
the optimal parameter combination based on final scores.

Usage:
    python analyze_parameter_sweep.py <results_directory> [output_directory]

Examples:
    python analyze_parameter_sweep.py results/sweep_ptp1b_20250709_172119
    python analyze_parameter_sweep.py /path/to/results my_analysis_output
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
from pathlib import Path
import re
from typing import Dict, Optional
import warnings

warnings.filterwarnings("ignore")

# Try to import optional dependencies
try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    print("Warning: seaborn not available, using matplotlib only")
    HAS_SEABORN = False

try:
    import yaml

    HAS_YAML = True
except ImportError:
    print("Warning: PyYAML not available, config file parsing disabled")
    HAS_YAML = False

# Set up plotting style
plt.style.use("default")
if HAS_SEABORN:
    sns.set_palette("husl")


class ParameterSweepAnalyzer:
    """Analyzer for parameter sweep results from EDG experiments."""

    def __init__(self, results_dir: str):
        """
        Initialize the analyzer with results directory.

        Args:
            results_dir: Path to the sweep results directory
        """
        self.results_dir = Path(results_dir)
        self.data = []
        self.df = None

    def extract_parameters_from_path(self, path: Path) -> Optional[Dict[str, float]]:
        """
        Extract parameters from directory path using multiple patterns.

        Args:
            path: Path to the experiment directory

        Returns:
            Dictionary of parameter values or None if parsing fails
        """
        # Try different patterns to extract parameters
        full_path = str(path)

        # Pattern 1: New descriptive directory format - boltz2_2.0A_200steps_3particles_adam_solver_lr0.0001
        # Extract from descriptive directory names first (most reliable for new structure)
        descriptive_pattern = r"boltz\d+_([\d.]+)A_(\d+)steps_\d+particles_(\w+)_solver_lr([\d.]+)"
        descriptive_match = re.search(descriptive_pattern, full_path)
        
        if descriptive_match:
            params = {
                "resolution": float(descriptive_match.group(1)),
                "num_steps": int(descriptive_match.group(2)),
                "solver_type": descriptive_match.group(3),
                "learning_rate": float(descriptive_match.group(4)),
            }
            
            # Try to extract max_guidance_denoising_ratio from config file
            config_file = path.parent / "experiment_config.yaml"
            if config_file.exists() and HAS_YAML:
                try:
                    with open(config_file, "r") as f:
                        config = yaml.safe_load(f)
                    if (
                        "density_guidance" in config
                        and "max_guidance_denoising_ratio" in config["density_guidance"]
                    ):
                        guidance_ratio = config["density_guidance"]["max_guidance_denoising_ratio"]
                        # Handle both direct values and nested schedules
                        if isinstance(guidance_ratio, dict) and "values" in guidance_ratio:
                            # Handle schedule case - take the first non-zero value
                            values = guidance_ratio["values"]
                            if isinstance(values, list):
                                for val in values:
                                    if isinstance(val, (int, float)) and val > 0:
                                        params["max_guidance_denoising_ratio"] = float(val)
                                        break
                        elif isinstance(guidance_ratio, (int, float)):
                            params["max_guidance_denoising_ratio"] = float(guidance_ratio)
                except Exception as e:
                    print(f"Warning: Could not parse config file {config_file}: {e}")
            
            return params

        # Pattern 2: Legacy format - lr{lr}_ratio{ratio}_steps{steps}
        pattern2 = r"lr([\d.]+)_ratio([\d.]+)_steps(\d+)"
        match2 = re.search(pattern2, full_path)

        if match2:
            return {
                "learning_rate": float(match2.group(1)),
                "max_guidance_denoising_ratio": float(match2.group(2)),
                "num_steps": int(match2.group(3)),
            }

        # Pattern 3: Extract from experiment config file if available
        config_file = path.parent / "experiment_config.yaml"
        if config_file.exists() and HAS_YAML:
            try:
                with open(config_file, "r") as f:
                    config = yaml.safe_load(f)

                params = {}
                # Extract learning rate
                if (
                    "adaptive_solver" in config
                    and "learning_rate" in config["adaptive_solver"]
                ):
                    params["learning_rate"] = float(
                        config["adaptive_solver"]["learning_rate"]
                    )

                # Extract max guidance denoising ratio
                if (
                    "density_guidance" in config
                    and "max_guidance_denoising_ratio" in config["density_guidance"]
                ):
                    params["max_guidance_denoising_ratio"] = float(
                        config["density_guidance"]["max_guidance_denoising_ratio"]
                    )

                # Extract num steps
                if "diffusion" in config and "num_steps" in config["diffusion"]:
                    params["num_steps"] = int(config["diffusion"]["num_steps"])

                if len(params) >= 2:  # At least 2 parameters found
                    return params

            except Exception as e:
                print(f"Warning: Could not parse config file {config_file}: {e}")

        # Pattern 4: Generic parameter extraction from directory names
        # Look for common parameter patterns in the full path
        params = {}

        # Learning rate patterns
        lr_patterns = [
            r"lr([\d.]+)",
            r"learning_rate([\d.]+)",
            r"adam_solver_lr([\d.]+)",
        ]
        for pattern in lr_patterns:
            match = re.search(pattern, full_path)
            if match:
                params["learning_rate"] = float(match.group(1))
                break

        # Steps patterns
        steps_patterns = [r"(\d+)steps", r"steps(\d+)", r"num_steps(\d+)"]
        for pattern in steps_patterns:
            match = re.search(pattern, full_path)
            if match:
                params["num_steps"] = int(match.group(1))
                break

        # Resolution patterns
        res_patterns = [r"(\d+\.?\d*)A", r"res([\d.]+)", r"resolution([\d.]+)"]
        for pattern in res_patterns:
            match = re.search(pattern, full_path)
            if match:
                params["resolution"] = float(match.group(1))
                break

        # Ratio/guidance patterns
        ratio_patterns = [r"ratio([\d.]+)", r"guidance([\d.]+)", r"weight([\d.]+)"]
        for pattern in ratio_patterns:
            match = re.search(pattern, full_path)
            if match:
                params["max_guidance_denoising_ratio"] = float(match.group(1))
                break

        return params if params else None

    def _extract_run_id(self, scores_file: Path) -> Optional[str]:
        """
        Extract run ID from path for multiple runs with same parameters.
        
        Args:
            scores_file: Path to the scores.csv file
            
        Returns:
            Run identifier (e.g., "_1", "_2") or None if not found
        """
        # Look for suffix patterns like _1, _2 in directory name
        parent_dir = scores_file.parent.name
        run_match = re.search(r"_(\d+)$", parent_dir)
        if run_match:
            return f"_{run_match.group(1)}"
        return None

    def load_scores(self, scores_file: Path) -> Optional[Dict[str, float]]:
        """
        Load and analyze scores from a CSV file.

        Args:
            scores_file: Path to the scores.csv file

        Returns:
            Dictionary of score metrics or None if loading fails
        """
        try:
            # Read the CSV file
            df = pd.read_csv(scores_file, header=0)

            # The file format appears to be: step,score with 3 comma-separated scores per line
            # Let's handle this properly
            scores_data = []
            for _, row in df.iterrows():
                # Each row has format: score1,score2,score3
                row_str = str(row.iloc[0])  # Get the first (and likely only) column
                if "," in row_str:
                    # Split by comma and convert to float
                    scores = [float(x.strip()) for x in row_str.split(",") if x.strip()]
                    scores_data.append(scores)
                else:
                    # Single score
                    try:
                        scores_data.append([float(row_str)])
                    except ValueError:
                        continue

            if not scores_data:
                return None

            # Convert to numpy array for easier manipulation
            scores_array = np.array(scores_data)

            # Calculate metrics
            final_scores = scores_array[-1]  # Last row
            best_final_score = np.min(final_scores)
            mean_final_score = np.mean(final_scores)
            std_final_score = np.std(final_scores)

            # Calculate convergence metrics
            if len(scores_array) > 10:
                # Look at improvement over last 10% of steps
                convergence_window = max(10, len(scores_array) // 10)
                recent_scores = scores_array[-convergence_window:]
                early_scores = scores_array[:convergence_window]

                improvement = np.mean(early_scores) - np.mean(recent_scores)
                convergence_rate = improvement / len(scores_array)
            else:
                convergence_rate = 0.0

            # Find the minimum score achieved during the run
            min_score_overall = np.min(scores_array)

            return {
                "best_final_score": best_final_score,
                "mean_final_score": mean_final_score,
                "std_final_score": std_final_score,
                "min_score_overall": min_score_overall,
                "convergence_rate": convergence_rate,
                "total_steps": len(scores_array),
                "scores_file": str(scores_file),
            }

        except Exception as e:
            print(f"Error loading scores from {scores_file}: {e}")
            return None

    def collect_data(self):
        """Collect all parameter sweep data from the results directory."""
        print(f"Collecting data from {self.results_dir}")

        # Detect directory structure (new shared input caching vs. old structure)
        shared_processing_dir = self.results_dir / "shared_boltz_processing"
        
        if shared_processing_dir.exists():
            print("Detected new shared input caching directory structure")
            # New structure: look for scores.csv in shared_boltz_processing subdirectories
            scores_files = list(shared_processing_dir.glob("*/scores.csv"))
        else:
            print("Detected legacy directory structure")
            # Legacy structure: recursive search for scores.csv
            scores_files = list(self.results_dir.rglob("scores.csv"))
            
        print(f"Found {len(scores_files)} scores.csv files")

        for scores_file in scores_files:
            # Extract parameters from path
            params = self.extract_parameters_from_path(scores_file)
            if params is None:
                print(f"Could not extract parameters from {scores_file}")
                continue

            # Load score metrics
            metrics = self.load_scores(scores_file)
            if metrics is None:
                print(f"Could not load scores from {scores_file}")
                continue

            # Add run identifier for multiple runs with same parameters
            run_id = self._extract_run_id(scores_file)
            if run_id:
                params["run_id"] = run_id

            # Combine parameters and metrics
            row = {**params, **metrics}
            self.data.append(row)

        if not self.data:
            raise ValueError("No valid data found in results directory")

        # Create DataFrame
        self.df = pd.DataFrame(self.data)
        print(f"Successfully loaded {len(self.df)} experiments")

        # Display basic statistics
        print("\nParameter ranges:")
        for param in ["learning_rate", "max_guidance_denoising_ratio", "num_steps", "resolution", "solver_type"]:
            if param in self.df.columns:
                if param == "solver_type":
                    print(f"  {param}: {self.df[param].unique().tolist()}")
                else:
                    print(f"  {param}: {self.df[param].min()} - {self.df[param].max()}")

    def find_best_configuration(self) -> Dict:
        """Find the best parameter configuration based on best final score."""
        if self.df is None:
            raise ValueError("No data loaded. Run collect_data() first.")

        # Sort by best final score (lower is better)
        best_row = self.df.loc[self.df["best_final_score"].idxmin()]

        print("\n" + "=" * 60)
        print("BEST PARAMETER CONFIGURATION")
        print("=" * 60)
        print(f"Learning Rate: {best_row['learning_rate']}")
        if 'max_guidance_denoising_ratio' in best_row:
            print(f"Max Guidance Denoising Ratio: {best_row['max_guidance_denoising_ratio']}")
        if 'resolution' in best_row:
            print(f"Resolution: {best_row['resolution']} Å")
        if 'solver_type' in best_row:
            print(f"Solver Type: {best_row['solver_type']}")
        print(f"Number of Steps: {best_row['num_steps']}")
        if 'run_id' in best_row:
            print(f"Run ID: {best_row['run_id']}")
        print(f"Best Final Score: {best_row['best_final_score']:.2f}")
        print(f"Mean Final Score: {best_row['mean_final_score']:.2f}")
        print(f"Min Score Overall: {best_row['min_score_overall']:.2f}")
        print(f"Convergence Rate: {best_row['convergence_rate']:.4f}")
        print(f"Results File: {best_row['scores_file']}")

        return best_row.to_dict()

    def create_visualizations(self, output_dir: str = "parameter_analysis"):
        """Create comprehensive visualizations of the parameter sweep results."""
        if self.df is None:
            raise ValueError("No data loaded. Run collect_data() first.")

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Set up the plotting style
        plt.rcParams["figure.figsize"] = (12, 8)

        # 1. Heatmap of performance across parameter combinations
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Create pivot tables for heatmaps
        metrics = [
            "best_final_score",
            "mean_final_score",
            "min_score_overall",
            "convergence_rate",
        ]

        for i, metric in enumerate(metrics):
            ax = axes[i // 2, i % 2]

            # Determine the best columns to use for pivot table
            if "max_guidance_denoising_ratio" in self.df.columns and len(self.df["max_guidance_denoising_ratio"].unique()) > 1:
                # Traditional 2D heatmap with guidance ratio
                pivot = self.df.pivot_table(
                    values=metric,
                    index="learning_rate",
                    columns="max_guidance_denoising_ratio",
                    aggfunc="mean",
                )
            elif "resolution" in self.df.columns and len(self.df["resolution"].unique()) > 1:
                # Alternative heatmap with resolution
                pivot = self.df.pivot_table(
                    values=metric,
                    index="learning_rate", 
                    columns="resolution",
                    aggfunc="mean",
                )
            else:
                # Fallback: single dimension plot
                grouped = self.df.groupby("learning_rate")[metric].mean()
                ax.plot(grouped.index, grouped.values, marker='o')
                ax.set_title(f"{metric} vs Learning Rate")
                ax.set_xlabel("Learning Rate")
                ax.set_ylabel(metric)
                continue

            # Plot heatmap if we have a valid pivot table
            if 'pivot' in locals() and pivot is not None and not pivot.empty:
                if HAS_SEABORN:
                    sns.heatmap(
                        pivot,
                        annot=True,
                        fmt=".2f",
                        cmap="viridis",
                        ax=ax,
                        cbar_kws={"label": metric},
                    )
                else:
                    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
                    ax.set_xticks(range(len(pivot.columns)))
                    ax.set_yticks(range(len(pivot.index)))
                    ax.set_xticklabels(pivot.columns)
                    ax.set_yticklabels(pivot.index)
                    plt.colorbar(im, ax=ax, label=metric)

                ax.set_title(f"{metric}")
                if "max_guidance_denoising_ratio" in self.df.columns:
                    ax.set_xlabel("Max Guidance Denoising Ratio")
                elif "resolution" in self.df.columns:
                    ax.set_xlabel("Resolution (Å)")
                ax.set_ylabel("Learning Rate")

        plt.tight_layout()
        plt.savefig(
            output_path / "parameter_heatmaps.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # 2. Box plots showing parameter effects
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # Choose available parameters dynamically
        available_params = [col for col in ["learning_rate", "max_guidance_denoising_ratio", "num_steps", "resolution"] 
                          if col in self.df.columns and len(self.df[col].unique()) > 1]
        
        for i, param in enumerate(available_params[:4]):  # Only plot first 4 parameters
            ax = axes[i // 2, i % 2]
            
            if param == "solver_type" and param in self.df.columns:
                # Handle categorical parameter
                if HAS_SEABORN:
                    sns.boxplot(data=self.df, x=param, y="best_final_score", ax=ax)
                else:
                    grouped = self.df.groupby(param)["best_final_score"].apply(list)
                    ax.boxplot(grouped.values, labels=grouped.index)
            else:
                # Handle numeric parameters
                if HAS_SEABORN:
                    sns.boxplot(data=self.df, x=param, y="best_final_score", ax=ax)
                else:
                    # Create boxplot manually
                    grouped = self.df.groupby(param)["best_final_score"].apply(list)
                    ax.boxplot(grouped.values, labels=grouped.index)

            ax.set_title(f"Score Distribution by {param}")
            ax.set_xlabel(param.replace("_", " ").title())
            ax.set_ylabel("Best Final Score")
            plt.setp(ax.get_xticklabels(), rotation=45)

        # Remove empty subplots
        for i in range(len(available_params), 4):
            fig.delaxes(axes[i // 2, i % 2])

        plt.tight_layout()
        plt.savefig(
            output_path / "parameter_boxplots.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # 3. 3D scatter plot of parameter space
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Choose 3 parameters for 3D plot
        x_param = "learning_rate"
        y_param = "max_guidance_denoising_ratio" if "max_guidance_denoising_ratio" in self.df.columns else "resolution"
        z_param = "num_steps"
        
        # Fallback if parameters are missing
        if y_param not in self.df.columns:
            y_param = available_params[1] if len(available_params) > 1 else "learning_rate"
        if z_param not in self.df.columns:
            z_param = available_params[2] if len(available_params) > 2 else "learning_rate"

        scatter = ax.scatter(
            self.df[x_param],
            self.df[y_param],
            self.df[z_param],
            c=self.df["best_final_score"],
            cmap="viridis",
            s=100,
            alpha=0.8,
        )

        ax.set_xlabel(x_param.replace("_", " ").title())
        ax.set_ylabel(y_param.replace("_", " ").title())
        ax.set_zlabel(z_param.replace("_", " ").title())
        ax.set_title("Parameter Space with Score Coloring")

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5)
        cbar.set_label("Best Final Score")

        plt.savefig(
            output_path / "parameter_space_3d.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # 4. Convergence comparison for top performers
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Get top 5 performers
        top_performers = self.df.nsmallest(5, "best_final_score")

        for _, row in top_performers.iterrows():
            # Load full score trajectory
            scores_file = Path(row["scores_file"])
            try:
                df_scores = pd.read_csv(scores_file, header=0)
                scores_data = []
                for _, score_row in df_scores.iterrows():
                    row_str = str(score_row.iloc[0])
                    if "," in row_str:
                        scores = [
                            float(x.strip()) for x in row_str.split(",") if x.strip()
                        ]
                        scores_data.append(np.min(scores))  # Best score at each step
                    else:
                        try:
                            scores_data.append(float(row_str))
                        except ValueError:
                            continue

                if scores_data:
                    # Create label with available parameters
                    label_parts = [f"lr={row['learning_rate']}"]
                    if 'max_guidance_denoising_ratio' in row:
                        label_parts.append(f"ratio={row['max_guidance_denoising_ratio']}")
                    if 'resolution' in row:
                        label_parts.append(f"res={row['resolution']}Å")
                    label_parts.append(f"steps={row['num_steps']}")
                    if 'run_id' in row:
                        label_parts.append(f"run{row['run_id']}")
                    
                    label = ", ".join(label_parts)
                    ax.plot(scores_data, label=label, linewidth=2)
            except Exception as e:
                print(f"Error plotting convergence for {scores_file}: {e}")

        ax.set_xlabel("Step")
        ax.set_ylabel("Best Score")
        ax.set_title("Convergence Comparison - Top 5 Performers")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            output_path / "convergence_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # 5. Summary statistics table
        # Group by available parameters
        groupby_params = ["learning_rate"]
        if "max_guidance_denoising_ratio" in self.df.columns:
            groupby_params.append("max_guidance_denoising_ratio")
        if "resolution" in self.df.columns:
            groupby_params.append("resolution")
        if "num_steps" in self.df.columns:
            groupby_params.append("num_steps")
        
        summary_stats = (
            self.df.groupby(groupby_params)
            .agg(
                {"best_final_score": ["mean", "std", "min"], "convergence_rate": "mean"}
            )
            .round(3)
        )

        # Save summary to CSV
        summary_stats.to_csv(output_path / "parameter_summary.csv")

        print(f"\nVisualizations saved to {output_path}/")
        print("Files created:")
        print("  - parameter_heatmaps.png")
        print("  - parameter_boxplots.png")
        print("  - parameter_space_3d.png")
        print("  - convergence_comparison.png")
        print("  - parameter_summary.csv")

    def generate_report(self, output_file: str = "parameter_sweep_report.txt"):
        """Generate a comprehensive text report of the analysis."""
        if self.df is None:
            raise ValueError("No data loaded. Run collect_data() first.")

        with open(output_file, "w") as f:
            f.write("PARAMETER SWEEP ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")

            # Dataset overview
            f.write("DATASET OVERVIEW\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total experiments: {len(self.df)}\n")
            f.write(
                f"Parameter combinations tested: {len(self.df.drop_duplicates(['learning_rate', 'max_guidance_denoising_ratio', 'num_steps']))}\n"
            )
            f.write(f"Results directory: {self.results_dir}\n\n")

            # Parameter ranges
            f.write("PARAMETER RANGES\n")
            f.write("-" * 20 + "\n")
            for param in ["learning_rate", "max_guidance_denoising_ratio", "num_steps", "resolution", "solver_type"]:
                if param in self.df.columns:
                    values = sorted(self.df[param].unique()) if param != "solver_type" else list(self.df[param].unique())
                    f.write(f"{param}: {values}\n")
            f.write("\n")

            # Best configuration
            best_config = self.find_best_configuration()
            f.write("BEST CONFIGURATION\n")
            f.write("-" * 20 + "\n")
            f.write(f"Learning Rate: {best_config['learning_rate']}\n")
            if 'max_guidance_denoising_ratio' in best_config:
                f.write(f"Max Guidance Denoising Ratio: {best_config['max_guidance_denoising_ratio']}\n")
            if 'resolution' in best_config:
                f.write(f"Resolution: {best_config['resolution']} Å\n")
            if 'solver_type' in best_config:
                f.write(f"Solver Type: {best_config['solver_type']}\n")
            f.write(f"Number of Steps: {best_config['num_steps']}\n")
            if 'run_id' in best_config:
                f.write(f"Run ID: {best_config['run_id']}\n")
            f.write(f"Best Final Score: {best_config['best_final_score']:.2f}\n")
            f.write(f"Mean Final Score: {best_config['mean_final_score']:.2f}\n")
            f.write(f"Min Score Overall: {best_config['min_score_overall']:.2f}\n\n")

            # Top 10 configurations
            f.write("TOP 10 CONFIGURATIONS\n")
            f.write("-" * 20 + "\n")
            top_10 = self.df.nsmallest(10, "best_final_score")
            for i, (_, row) in enumerate(top_10.iterrows(), 1):
                config_parts = [f"lr={row['learning_rate']}"]
                if 'max_guidance_denoising_ratio' in row:
                    config_parts.append(f"ratio={row['max_guidance_denoising_ratio']}")
                if 'resolution' in row:
                    config_parts.append(f"res={row['resolution']}Å")
                config_parts.append(f"steps={row['num_steps']}")
                if 'run_id' in row:
                    config_parts.append(f"run{row['run_id']}")
                config_parts.append(f"score={row['best_final_score']:.2f}")
                
                f.write(f"{i:2d}. {', '.join(config_parts)}\n")
            f.write("\n")

            # Parameter effect analysis
            f.write("PARAMETER EFFECT ANALYSIS\n")
            f.write("-" * 30 + "\n")
            for param in ["learning_rate", "max_guidance_denoising_ratio", "num_steps", "resolution", "solver_type"]:
                if param in self.df.columns:
                    grouped = self.df.groupby(param)["best_final_score"].agg(
                        ["mean", "std", "min"]
                    )
                    f.write(f"\n{param.upper().replace('_', ' ')}:\n")
                    for value, stats in grouped.iterrows():
                        f.write(
                            f"  {value}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, min={stats['min']:.2f}\n"
                        )

        print(f"\nReport saved to {output_file}")


def main():
    """Main function to run the parameter sweep analysis."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Analyze parameter sweep results from EDG experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python analyze_parameter_sweep.py results/sweep_ptp1b_20250709_172119
    python analyze_parameter_sweep.py /path/to/results --output my_analysis
    python analyze_parameter_sweep.py results/ --metric min_score_overall
        """,
    )

    parser.add_argument(
        "results_dir", help="Path to the parameter sweep results directory"
    )

    parser.add_argument(
        "--output",
        "-o",
        default="parameter_analysis",
        help="Output directory for analysis results (default: parameter_analysis)",
    )

    parser.add_argument(
        "--metric",
        "-m",
        default="best_final_score",
        choices=[
            "best_final_score",
            "mean_final_score",
            "min_score_overall",
            "convergence_rate",
        ],
        help="Metric to use for finding best configuration (default: best_final_score)",
    )

    parser.add_argument(
        "--report",
        "-r",
        default="parameter_sweep_report.txt",
        help="Output file for text report (default: output_dir/parameter_sweep_report.txt)",
    )

    # Handle case where no arguments are provided
    if len(sys.argv) == 1:
        # Check if default directory exists
        default_dir = "results/sweep"
        if Path(default_dir).exists():
            print(f"No arguments provided, using default directory: {default_dir}")
            results_dir = default_dir
            output_dir = "parameter_analysis"
            report_file = f"{output_dir}/parameter_sweep_report.txt"
        else:
            parser.print_help()
            return
    else:
        args = parser.parse_args()
        results_dir = args.results_dir
        output_dir = args.output
        report_file = f"{output_dir}/{args.report}"

    # Check if results directory exists
    if not Path(results_dir).exists():
        print(f"Error: Results directory '{results_dir}' does not exist.")
        print("Please specify the correct path to your parameter sweep results.")
        return

    # Initialize analyzer
    analyzer = ParameterSweepAnalyzer(results_dir)

    try:
        # Collect data
        analyzer.collect_data()

        if analyzer.df is None or len(analyzer.df) == 0:
            print(
                "No valid experiment data found. Please check the results directory structure."
            )
            return

        # Find best configuration
        analyzer.find_best_configuration()

        # Create visualizations
        analyzer.create_visualizations(output_dir)

        # Generate report
        analyzer.generate_report(report_file)

        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        print("Check the following files for detailed results:")
        print(f"- {output_dir}/ (directory with visualizations)")
        print(f"- {report_file} (comprehensive text report)")

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback

        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
