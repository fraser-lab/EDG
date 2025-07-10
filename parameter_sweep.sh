#!/bin/bash

# Parameter sweep script for EDG experiments
# 
# This script runs parameter sweeps over:
# - Adaptive solver learning rates
# - Max guidance denoising ratios  
# - Number of diffusion steps
#
# Usage: ./parameter_sweep.sh <config_file> [--dry-run]
# 
# Examples:
#   ./parameter_sweep.sh configs/ptp1b.yaml                    # Run full sweep
#   ./parameter_sweep.sh configs/ptp1b.yaml --dry-run          # Validate only
#
# Features:
# - Uses new direct CLI parameters (--learning-rate, --max-guidance-denoising-ratio, --num-steps)
# - Supports dry-run mode for validation
# - Creates organized output directories with logs
# - Generates summary reports

set -e

# Parse command line arguments
DRY_RUN=false
CONFIG_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -*)
            echo "Unknown option $1"
            exit 1
            ;;
        *)
            if [ -z "$CONFIG_FILE" ]; then
                CONFIG_FILE="$1"
            else
                echo "Too many arguments"
                exit 1
            fi
            shift
            ;;
    esac
done

# Check if config file is provided
if [ -z "$CONFIG_FILE" ]; then
    echo "Usage: $0 <config_file> [--dry-run]"
    echo "Example: $0 configs/ptp1b.yaml"
    echo "Example: $0 configs/ptp1b.yaml --dry-run"
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found"
    exit 1
fi

# Extract base name for output directory structure
BASE_NAME=$(basename "$CONFIG_FILE" .yaml)

# Parameter arrays - customize these values as needed
LEARNING_RATES=(0.01 0.02 0.05 0.1 0.2 0.4)           # Adaptive solver learning rates
MAX_GUIDANCE_RATIOS=(0.1 0.2 0.4)         # Maximum guidance to denoising ratios
NUM_STEPS=(200 400)                       # Number of diffusion steps

echo "Starting parameter sweep for $CONFIG_FILE"
if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN MODE - will only validate configurations"
fi
echo "Learning rates: ${LEARNING_RATES[*]}"
echo "Max guidance denoising ratios: ${MAX_GUIDANCE_RATIOS[*]}"
echo "Number of steps: ${NUM_STEPS[*]}"
echo "Total combinations: $((${#LEARNING_RATES[@]} * ${#MAX_GUIDANCE_RATIOS[@]} * ${#NUM_STEPS[@]}))"
echo ""

# Create results directory
RESULTS_DIR="results/sweep_${BASE_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Counter for progress tracking
total_runs=$((${#LEARNING_RATES[@]} * ${#MAX_GUIDANCE_RATIOS[@]} * ${#NUM_STEPS[@]}))
current_run=0

# Log file for the sweep
LOG_FILE="$RESULTS_DIR/sweep_log.txt"
echo "Parameter sweep started at $(date)" > "$LOG_FILE"
echo "Config file: $CONFIG_FILE" >> "$LOG_FILE"
echo "Results directory: $RESULTS_DIR" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Run parameter sweep
for lr in "${LEARNING_RATES[@]}"; do
    for ratio in "${MAX_GUIDANCE_RATIOS[@]}"; do
        for steps in "${NUM_STEPS[@]}"; do
            current_run=$((current_run + 1))
            
            # Create output directory name
            output_dir="$RESULTS_DIR/lr${lr}_ratio${ratio}_steps${steps}"
            
            echo "[$current_run/$total_runs] Running: lr=$lr, ratio=$ratio, steps=$steps"
            echo "Output: $output_dir"
            
            # Log the run
            echo "Run $current_run/$total_runs: lr=$lr, ratio=$ratio, steps=$steps" >> "$LOG_FILE"
            echo "Started at: $(date)" >> "$LOG_FILE"
            
            # Build command
            cmd="pixi run python -m edg --config \"$CONFIG_FILE\" --learning-rate $lr --max-guidance-denoising-ratio $ratio --num-steps $steps --output-dir \"$output_dir\""
            
            # Add validation-only flag for dry run
            if [ "$DRY_RUN" = true ]; then
                cmd="$cmd --validate-only"
            fi
            
            # Run the experiment
            if eval "$cmd 2>&1 | tee \"$output_dir.log\""; then
                
                if [ "$DRY_RUN" = true ]; then
                    echo "✓ Configuration validated" >> "$LOG_FILE"
                    echo "✓ Configuration validated"
                else
                    echo "✓ Completed successfully" >> "$LOG_FILE"
                    echo "✓ Run completed successfully"
                fi
            else
                echo "✗ Failed with exit code $?" >> "$LOG_FILE"
                echo "✗ Run failed - check $output_dir.log for details"
            fi
            
            echo "Finished at: $(date)" >> "$LOG_FILE"
            echo "" >> "$LOG_FILE"
            echo ""
        done
    done
done

echo "Parameter sweep completed!"
echo "Results saved in: $RESULTS_DIR"
echo "Summary log: $LOG_FILE"

# Create a summary of all runs
SUMMARY_FILE="$RESULTS_DIR/sweep_summary.txt"
echo "Parameter Sweep Summary" > "$SUMMARY_FILE"
echo "======================" >> "$SUMMARY_FILE"
echo "Config file: $CONFIG_FILE" >> "$SUMMARY_FILE"
echo "Total runs: $total_runs" >> "$SUMMARY_FILE"
echo "Started: $(head -1 "$LOG_FILE" | cut -d' ' -f4-)" >> "$SUMMARY_FILE"
echo "Completed: $(date)" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Count successful runs
successful_runs=$(grep -c "✓ Completed successfully" "$LOG_FILE" || echo "0")
echo "Successful runs: $successful_runs/$total_runs" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

echo "Run directories:" >> "$SUMMARY_FILE"
ls -1 "$RESULTS_DIR" | grep -E "^lr.*_ratio.*_steps.*$" | sort >> "$SUMMARY_FILE"

echo ""
echo "Summary saved in: $SUMMARY_FILE"